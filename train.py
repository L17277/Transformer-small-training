import sys
import os
import re
import json
import time
import logging
import threading
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{time.strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. 修复函数
def fix_vocab_size_mismatch(model_dir, actual_vocab_size):
    """修正配置文件中词汇表大小不匹配的问题"""
    config_path = os.path.join(model_dir, "config.json")
    
    try:
        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新词汇表大小
        if config.get('vocab_size', 0) != actual_vocab_size:
            logger.info(f"修复词汇表大小: {config.get('vocab_size')} -> {actual_vocab_size}")
            config['vocab_size'] = actual_vocab_size
            
            # 确保所有相关配置一致
            if 'vocab_size' in config:
                config['vocab_size'] = actual_vocab_size
            if 'embed_dim' in config:
                config['embedding_dim'] = config['embed_dim']
            
            # 保存更新后的配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
    except Exception as e:
        logger.error(f"修复词汇表时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    return False

def rename_model_keys(state_dict):
    """转换transformer.*前缀为decoder.*以匹配模型结构"""
    new_state_dict = {}
    replacements = [
        ("transformer.layers", "decoder.layers"),
        ("fc_out.weight", "fc.weight"),
        ("fc_out.bias", "fc.bias"),
        ("position_embedding", "pos_embed")
    ]
    
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            if old in new_key:
                new_key = new_key.replace(old, new)
                break
        
        # 特殊处理：确保所有层前缀一致
        if "decoder.layers" in new_key and "transformer" in new_key:
            new_key = new_key.replace("transformer", "decoder")
        
        new_state_dict[new_key] = value
    
    return new_state_dict

# 2. 模型定义 - 使用decoder前缀确保兼容性
class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词汇表大小和嵌入维度
        self.vocab_size = config['vocab_size']
        self.embed_dim = config.get('embed_dim', config.get('embedding_dim', 384))
        
        # 词嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # 位置编码
        max_seq_len = config.get('max_seq_len', config.get('seq_length', 256))
        self.pos_embed = nn.Embedding(max_seq_len, self.embed_dim)
        
        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True  # 添加批处理优先
        )
        
        # Transformer 解码器 - 使用decoder前缀
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['num_layers']
        )
        
        # 输出层
        self.fc = nn.Linear(self.embed_dim, self.vocab_size)
        
        # 层归一化
        self.ln = nn.LayerNorm(self.embed_dim)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
        nn.init.xavier_uniform_(self.pos_embed.weight)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 创建位置索引
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 词嵌入 + 位置嵌入
        x_emb = self.embedding(x) + self.pos_embed(positions)
        x_emb = self.ln(x_emb)
        
        # 创建目标掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer 解码器
        # 注意: 使用batch_first=True后不需要permute
        output = self.decoder(
            tgt=x_emb, 
            memory=x_emb, 
            tgt_mask=tgt_mask
        )
        
        # 输出层
        logits = self.fc(output)
        
        return logits

# 3. 数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length=256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text = self.load_and_preprocess_text()
        
        # 编码文本
        encoded = self.tokenizer.encode(self.text)
        self.encoded_text = encoded.ids
        self.pad_id = tokenizer.token_to_id("[PAD]")
        
        logger.info(f"加载数据集完成，总token数: {len(self.encoded_text)}")
        
    def load_and_preprocess_text(self):
        """加载并预处理文本"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 简单的文本清理
            text = re.sub(r'\s+', ' ', text)  # 替换多个空白字符为单个空格
            text = text.strip()
            return text
        except Exception as e:
            logger.error(f"加载文本文件失败: {str(e)}")
            raise
        
    def __len__(self):
        """返回数据集长度"""
        if len(self.encoded_text) <= self.seq_length:
            return 1
        return len(self.encoded_text) // self.seq_length
    
    def __getitem__(self, idx):
        """获取单个样本"""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 用于创建目标序列
        
        # 提取输入和目标
        segment = self.encoded_text[start_idx:end_idx]
        
        # 处理序列不足的情况
        if len(segment) < self.seq_length + 1:
            # 填充不足的序列
            padding = [self.pad_id] * (self.seq_length + 1 - len(segment))
            segment.extend(padding)
        
        input_ids = segment[:self.seq_length]
        target_ids = segment[1:self.seq_length+1]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

# 4. 训练器类
class ModelTrainer:
    def __init__(self, config, train_file, model_dir, device='auto', app=None):
        self.config = config
        self.train_file = train_file
        self.model_dir = Path(model_dir)
        self.stop_training = False
        self.app = app  # 对GUI应用的引用
        self.current_epoch = 0
        self.total_epochs = config["epochs"]
        
        # 确保模型目录存在
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置并添加模型类型
        self.config["model_type"] = "decoder"  # 强制设置为decoder类型
        self._save_config()
        
        # 设备选择
        self.device = self._select_device(device)
        
        # 检查设备内存
        self._check_device_memory()
        
        # 初始化分词器并获取实际词表大小
        self.tokenizer = self._init_tokenizer()
        actual_vocab_size = len(self.tokenizer.get_vocab())
        
        # 修复词汇表大小
        fix_vocab_size_mismatch(self.model_dir, actual_vocab_size)
        self.config["vocab_size"] = actual_vocab_size  # 更新配置中的词汇表大小
        
        # 初始化模型
        self.model = self._init_model().to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("[PAD]"))
        
        # 训练状态
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info(f"模型训练器初始化完成 (设备: {self.device}, 词表大小: {actual_vocab_size})")
        if self.app:
            self.app.log_message(f"模型训练器初始化完成 (设备: {self.device}, 词表大小: {actual_vocab_size})")
    
    def _check_device_memory(self):
        """检查设备内存是否足够"""
        if self.device.type == 'cuda':
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            free_mem = total_mem - torch.cuda.memory_allocated(self.device)
            
            # 估计模型需要的内存 (粗略估计)
            estimated_mem = self.config["batch_size"] * self.config["seq_length"] * 1024 * 1024 * 2
            
            if free_mem < estimated_mem:
                warning = (f"警告: 设备内存可能不足!\n"
                          f"可用内存: {free_mem / (1024**2):.2f} MB\n"
                          f"估计需要: {estimated_mem / (1024**2):.2f} MB")
                logger.warning(warning)
                if self.app:
                    self.app.log_message(warning, level="warning")
    
    def _select_device(self, device):
        """智能选择设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _init_tokenizer(self):
        """初始化分词器并自动确定词表大小"""
        tokenizer_dir = self.model_dir / "vocab"
        tokenizer_dir.mkdir(exist_ok=True)
        
        vocab_file = tokenizer_dir / "vocab.json"
        merges_file = tokenizer_dir / "merges.txt"
        
        # 如果分词器不存在，则训练一个新的
        if not vocab_file.exists() or not merges_file.exists():
            if self.app:
                self.app.log_message("训练新的分词器...")
            logger.info("训练新的分词器...")
            
            # 创建ByteLevelBPETokenizer实例
            tokenizer = ByteLevelBPETokenizer()
            
            # 训练分词器 - 使用训练数据自动确定词表大小
            tokenizer.train(
                files=[self.train_file],
                vocab_size=30000,  # 初始值，实际大小由数据决定
                min_frequency=2,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            )
            
            # 保存分词器
            tokenizer.save_model(str(tokenizer_dir))
            logger.info(f"分词器已保存到: {tokenizer_dir}")
            if self.app:
                self.app.log_message(f"分词器已保存到: {tokenizer_dir}")
        
        # 加载分词器
        tokenizer = ByteLevelBPETokenizer(
            str(vocab_file),
            str(merges_file)
        )
        
        # 添加特殊token
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for token in special_tokens:
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token])
        
        # 记录实际词表大小
        actual_vocab_size = len(tokenizer.get_vocab())
        logger.info(f"实际词表大小: {actual_vocab_size}")
        if self.app:
            self.app.log_message(f"实际词表大小: {actual_vocab_size}")
        
        return tokenizer
    
    def _init_model(self):
        """初始化模型"""
        model_path = self.model_dir / "model.pth"
        if model_path.exists():
            # 加载现有模型
            model = TransformerLM(self.config)
            
            try:
                # 加载权重并修复键名
                state_dict = torch.load(model_path, map_location='cpu')
                
                # 检查键是否匹配
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                
                if len(pretrained_dict) == 0:
                    logger.warning("没有匹配的键，尝试重命名键...")
                    state_dict = rename_model_keys(state_dict)
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                
                # 加载匹配的权重
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                
                # 报告缺失的键
                missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
                if missing_keys:
                    logger.warning(f"缺少权重: {', '.join(missing_keys)}")
                    if self.app:
                        self.app.log_message(f"警告: 缺少权重: {', '.join(missing_keys)}", level="warning")
                
                logger.info(f"从 {model_path} 加载现有模型")
                if self.app:
                    self.app.log_message(f"从 {model_path} 加载现有模型")
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                logger.error(traceback.format_exc())
                if self.app:
                    self.app.log_message(f"加载模型失败: {str(e)}", level="error")
                # 创建新模型作为后备
                model = TransformerLM(self.config)
                logger.info("创建新模型作为后备")
                if self.app:
                    self.app.log_message("创建新模型作为后备")
        else:
            # 创建新模型
            model = TransformerLM(self.config)
            logger.info("创建新模型")
            if self.app:
                self.app.log_message("创建新模型")
        
        return model
    
    def _save_config(self):
        """保存配置到文件"""
        config_path = self.model_dir / "config.json"
        config_data = self.config.copy()
        config_data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        config_data["model_version"] = "1.1"  # 更新模型版本
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"配置已保存到: {config_path}")
        if self.app:
            self.app.log_message(f"配置已保存到: {config_path}")
    
    def create_data_loaders(self):
        """创建训练和验证数据加载器"""
        if self.app:
            self.app.log_message("创建数据加载器...")
        logger.info("创建数据加载器...")
        
        try:
            # 创建数据集
            dataset = TextDataset(
                self.train_file,
                self.tokenizer,
                self.config["seq_length"]
            )
            
            # 划分训练集和验证集
            total_size = len(dataset)
            if total_size == 0:
                raise ValueError("数据集为空，请检查输入文件")
                
            train_size = int(0.9 * total_size)
            val_size = total_size - train_size
            
            # 如果验证集太小，调整比例
            if val_size < 10:
                train_size = max(1, total_size - 10)
                val_size = total_size - train_size
            
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=0,  # 避免多进程问题
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.config["batch_size"], 16),  # 验证集使用较小的批大小
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            logger.info(f"数据集大小: 总计 {total_size}, 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
            if self.app:
                self.app.log_message(f"数据集大小: 总计 {total_size}, 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
            
            return train_loader, val_loader
        except Exception as e:
            logger.error(f"创建数据加载器失败: {str(e)}")
            if self.app:
                self.app.log_message(f"创建数据加载器失败: {str(e)}", level="error")
            raise
    
    def train_epoch(self, epoch, train_loader):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        steps = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if self.stop_training:
                logger.info("训练已停止")
                if self.app:
                    self.app.log_message("训练已停止")
                return None
                
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            target_ids = batch['target_ids'].to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            # 计算损失 - 只计算目标序列的损失
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 每10个batch记录一次
            if batch_idx % 10 == 0 or batch_idx == steps - 1:
                avg_loss = total_loss / total_samples
                progress = (batch_idx + 1) / steps * 100
                msg = f"Epoch {epoch} | 批次 {batch_idx+1}/{steps} | 损失: {avg_loss:.4f} | 进度: {progress:.1f}%"
                logger.info(msg)
                if self.app:
                    self.app.log_message(msg)
                    self.app.update_progress(progress, f"训练中: {progress:.1f}%")
        
        return total_loss / total_samples
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        steps = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 将数据移动到设备
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(input_ids)
                
                # 计算损失
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                
                # 记录损失
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 更新进度
                if batch_idx % 5 == 0 or batch_idx == steps - 1:
                    progress = (batch_idx + 1) / steps * 100
                    if self.app:
                        self.app.update_progress(progress, f"验证中: {progress:.1f}%")
        
        return total_loss / total_samples
    
    def save_model(self, epoch, val_loss):
        """保存模型 - 确保键名正确"""
        model_path = self.model_dir / "model.pth"
        
        # 确保使用正确的键名保存
        state_dict = self.model.state_dict()
        
        # 保存权重
        torch.save(state_dict, model_path)
        msg = f"模型已保存到: {model_path} (Epoch {epoch}, 损失 {val_loss:.4f})"
        logger.info(msg)
        if self.app:
            self.app.log_message(msg)
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        if self.app:
            self.app.log_message("开始训练...")
            self.app.update_progress(0, "准备训练...")
        
        start_time = time.time()
        
        try:
            # 创建数据加载器
            train_loader, val_loader = self.create_data_loaders()
            
            # 训练循环
            for epoch in range(1, self.config["epochs"] + 1):
                self.current_epoch = epoch
                
                if self.stop_training:
                    logger.info("训练已停止")
                    if self.app:
                        self.app.log_message("训练已停止")
                    break
                    
                epoch_start = time.time()
                
                # 训练一个epoch
                train_loss = self.train_epoch(epoch, train_loader)
                if train_loss is None:  # 训练被停止
                    break
                    
                # 验证
                val_loss = self.validate(val_loader)
                
                # 更新学习率
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 记录epoch信息
                epoch_time = time.time() - epoch_start
                msg = f"Epoch {epoch}/{self.config['epochs']} - " \
                      f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, " \
                      f"学习率: {current_lr:.2e}, 时间: {epoch_time:.2f}s"
                logger.info(msg)
                if self.app:
                    self.app.log_message(msg)
                
                # 更新进度
                epoch_progress = epoch / self.config["epochs"] * 100
                if self.app:
                    self.app.update_progress(epoch_progress, f"Epoch {epoch}/{self.config['epochs']}")
                
                # 检查最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_model(epoch, val_loss)
                else:
                    self.epochs_without_improvement += 1
                
                # 提前停止检查
                if self.epochs_without_improvement >= self.config["early_stop_patience"]:
                    msg = f"验证损失在 {self.config['early_stop_patience']} 个epoch内没有改善，提前停止"
                    logger.info(msg)
                    if self.app:
                        self.app.log_message(msg)
                    break
                
                # 达到目标损失
                if val_loss <= self.config["target_loss"]:
                    msg = f"达到目标损失 {self.config['target_loss']}，停止训练"
                    logger.info(msg)
                    if self.app:
                        self.app.log_message(msg)
                    break
                
                # 清理内存
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # 训练完成
            total_time = time.time() - start_time
            msg = f"训练完成! 总时间: {total_time:.2f}秒, 最佳验证损失: {self.best_loss:.4f}"
            logger.info(msg)
            if self.app:
                self.app.log_message(msg)
                self.app.training_completed()
                self.app.update_progress(100, "训练完成!")
        
        except Exception as e:
            logger.error(f"训练出错: {str(e)}")
            logger.error(traceback.format_exc())
            if self.app:
                self.app.log_message(f"训练出错: {str(e)}", level="error")
                self.app.log_message(traceback.format_exc(), level="error")
                self.app.training_failed(str(e))

# 5. GUI应用程序
class TrainingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformer语言模型训练器")
        self.geometry("1000x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 训练状态
        self.is_training = False
        self.trainer = None
        self.training_thread = None
        
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.main_frame, text="训练控制")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # 训练文件选择
        ttk.Label(control_frame, text="训练文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_file_var = tk.StringVar()
        train_file_entry = ttk.Entry(control_frame, textvariable=self.train_file_var, width=30)
        train_file_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.browse_train_file).grid(row=0, column=2, padx=5, pady=2)
        
        # 模型目录选择
        ttk.Label(control_frame, text="模型目录:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.model_dir_var = tk.StringVar(value=str(Path.home() / "transformer_model"))
        model_dir_entry = ttk.Entry(control_frame, textvariable=self.model_dir_var, width=30)
        model_dir_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.browse_model_dir).grid(row=1, column=2, padx=5, pady=2)
        
        # 设备选择
        ttk.Label(control_frame, text="训练设备:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar(value="auto")
        device_combobox = ttk.Combobox(control_frame, textvariable=self.device_var, width=28)
        device_combobox['values'] = ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps')
        device_combobox.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # 参数设置
        params_frame = ttk.LabelFrame(control_frame, text="训练参数")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        ttk.Label(params_frame, text="批大小:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="序列长度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.seq_length_var = tk.IntVar(value=256)
        ttk.Entry(params_frame, textvariable=self.seq_length_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="学习率:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value="3e-4")
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="最大轮数:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="目标损失:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.target_loss_var = tk.DoubleVar(value=1.5)
        ttk.Entry(params_frame, textvariable=self.target_loss_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="早停耐心值:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.early_stop_var = tk.IntVar(value=5)
        ttk.Entry(params_frame, textvariable=self.early_stop_var, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 按钮区域
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="开始训练", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止训练", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧日志面板
        log_frame = ttk.LabelFrame(self.main_frame, text="训练日志")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # 进度条
        progress_frame = ttk.Frame(self)
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        ttk.Label(progress_frame, text="进度:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100,
            length=300
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_label = ttk.Label(progress_frame, text="准备就绪")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 设置网格列权重
        control_frame.columnconfigure(1, weight=1)
        
        # 添加清空日志按钮
        clear_btn = ttk.Button(control_frame, text="清空日志", command=self.clear_log)
        clear_btn.grid(row=5, column=0, columnspan=3, pady=5)
    
    def clear_log(self):
        """清空日志"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def log_message(self, message, level="info"):
        """记录日志消息"""
        # 特殊处理分词器错误
        if "tokenizer" in message.lower() or "vocab" in message.lower():
            solution = (
                "\n解决方案:\n"
                "1. 删除模型目录下的 'vocab' 文件夹\n"
                "2. 重新开始训练\n"
                "3. 如果问题仍然存在，尝试使用纯英文路径"
            )
            message += solution
        
        def update_log():
            self.log_text.config(state=tk.NORMAL)
            # 添加带颜色的标签
            if level == "error":
                self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [ERROR] ", "error")
            elif level == "warning":
                self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [WARNING] ", "warning")
            else:
                self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [INFO] ", "info")
            
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            
            # 更新状态栏
            self.status_var.set(message[:100])  # 显示前100个字符
        
        self.after(0, update_log)
    
    def update_progress(self, value, text=None):
        """更新进度条"""
        def update():
            self.progress_var.set(value)
            if text:
                self.progress_label.config(text=text)
            if value >= 100:
                self.progress_label.config(text="完成!")
        
        self.after(0, update)
    
    def browse_train_file(self):
        """浏览训练文件"""
        file_path = filedialog.askopenfilename(
            title="选择训练文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.train_file_var.set(file_path)
    
    def browse_model_dir(self):
        """浏览模型目录"""
        dir_path = filedialog.askdirectory(
            title="选择模型目录",
            initialdir=str(Path.home())
        )
        if dir_path:
            self.model_dir_var.set(dir_path)
    
    def start_training(self):
        """开始训练"""
        if self.is_training:
            return
            
        train_file = self.train_file_var.get()
        model_dir = self.model_dir_var.get()
        
        if not train_file:
            messagebox.showerror("错误", "请选择训练文件")
            return
            
        if not os.path.exists(train_file):
            messagebox.showerror("错误", "训练文件不存在")
            return
            
        if not model_dir:
            messagebox.showerror("错误", "请选择模型目录")
            return
            
        # 准备配置
        config = {
            "batch_size": self.batch_size_var.get(),
            "seq_length": self.seq_length_var.get(),
            "learning_rate": float(self.lr_var.get()),
            "epochs": self.epochs_var.get(),
            "target_loss": self.target_loss_var.get(),
            "embed_dim": 384,
            "nhead": 6,
            "num_layers": 8,
            "dim_feedforward": 1536,
            "early_stop_patience": self.early_stop_var.get(),
            "min_lr": 1e-6,
            "weight_decay": 1e-5,
            "dropout": 0.1
        }
        
        # 更新UI状态
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("训练中...")
        self.log_message("开始训练...")
        self.update_progress(0, "初始化...")
        
        # 在后台线程中启动训练
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(config, train_file, model_dir),
            daemon=True
        )
        self.training_thread.start()
    
    def run_training(self, config, train_file, model_dir):
        """执行训练任务"""
        try:
            # 初始化训练器
            self.trainer = ModelTrainer(
                config, 
                train_file, 
                model_dir, 
                device=self.device_var.get(),
                app=self
            )
            
            # 开始训练
            self.trainer.train()
            
        except Exception as e:
            self.log_message(f"训练出错: {str(e)}", level="error")
            self.log_message(traceback.format_exc(), level="error")
            self.status_var.set(f"训练出错: {str(e)}")
            self.training_failed(str(e))
        
        finally:
            # 重置UI状态
            self.is_training = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def stop_training(self):
        """停止训练"""
        if self.is_training and self.trainer:
            self.trainer.stop_training = True
            self.status_var.set("正在停止训练...")
            self.log_message("正在停止训练...")
    
    def training_completed(self):
        """训练完成时的回调"""
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("训练完成")
        messagebox.showinfo("训练完成", "模型训练成功完成!")
    
    def training_failed(self, error_msg):
        """训练失败时的回调"""
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(f"训练失败: {error_msg}")
        messagebox.showerror("训练失败", f"训练过程中出错:\n{error_msg}")
    
    def on_close(self):
        """关闭应用程序"""
        if self.is_training:
            if messagebox.askyesno("确认", "训练仍在进行中，确定要退出吗？"):
                self.stop_training()
                self.destroy()
        else:
            self.destroy()

# 6. 主函数
def main():
    # 设置工作目录为脚本所在目录
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(application_path)
    
    # 创建应用并处理未捕获的异常
    def handle_exception(exc_type, exc_value, exc_traceback):
        """全局异常处理"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
        
        # 创建错误报告
        error_report = (
            f"发生未处理的异常:\n\n"
            f"类型: {exc_type.__name__}\n"
            f"信息: {str(exc_value)}\n\n"
            f"详细信息已记录到日志文件。"
        )
        
        # 尝试在UI中显示错误
        try:
            messagebox.showerror("致命错误", error_report)
        except:
            print(error_report)
    
    sys.excepthook = handle_exception
    
    # 设置DPI感知 (Windows)
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    
    # 创建并运行应用
    app = TrainingApp()
    
    # 配置标签样式
    app.log_text.tag_config("error", foreground="red")
    app.log_text.tag_config("warning", foreground="orange")
    app.log_text.tag_config("info", foreground="black")
    
    app.mainloop()

if __name__ == "__main__":
    main()
