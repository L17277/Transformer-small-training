import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import re
import threading
import time
import logging
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

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

class Config:
    # 默认配置参数 - 针对因果语言模型优化
    DEFAULT_PARAMS = {
        "learning_rate": 3e-4,
        "batch_size": 32,
        "epochs": 100,
        "seq_length": 256,
        "target_loss": 1.5,
        "vocab_size": 30000,
        "embed_dim": 384,
        "nhead": 6,
        "num_layers": 8,
        "dim_feedforward": 1536,
        "early_stop_patience": 10,
        "min_lr": 1e-6,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "model_type": "decoder",  # 明确模型类型
        "model_version": "1.0"   # 新增模型版本标识
    }

def normalize_path(path):
    """规范化路径，处理特殊字符"""
    try:
        path = os.path.abspath(path)
        # Windows系统处理长路径和特殊字符
        if sys.platform == "win32":
            # 转换路径为短格式（8.3格式）避免Unicode问题
            import ctypes
            from ctypes import wintypes
            
            kernel32 = ctypes.WinDLL('kernel32')
            kernel32.GetShortPathNameW.argtypes = [
                wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD
            ]
            kernel32.GetShortPathNameW.restype = wintypes.DWORD
            
            buf = ctypes.create_unicode_buffer(512)
            if kernel32.GetShortPathNameW(path, buf, ctypes.sizeof(buf)):
                return buf.value
        return path
    except Exception as e:
        logger.error(f"路径规范化失败: {str(e)}")
        return path

class InferenceEngine:
    """推理引擎 - 与训练模型完全兼容"""
    def __init__(self, model_dir, device='auto'):
        self.model_dir = Path(normalize_path(model_dir))
        self.device = self._select_device(device)
        
        # 加载配置并确保模型类型正确
        self.config = self._load_config()
        self._ensure_model_type()
        
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self._warmup_model()
        
        logger.info(f"推理引擎初始化完成 (设备: {self.device}, 模型类型: {self.config['model_type']})")

    def _select_device(self, device):
        """智能选择设备，支持多GPU"""
        if device == "auto":
            if torch.cuda.is_available():
                # 如果有多个GPU，选择第一个
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _load_config(self):
        """加载模型配置并进行健壮性处理"""
        config_path = self.model_dir / "config.json"
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            with open(config_path) as f:
                config = json.load(f)
                
            # 确保关键参数存在
            required_keys = ["vocab_size", "embed_dim", "num_layers", 
                            "nhead", "dim_feedforward", "seq_length"]
            for key in required_keys:
                if key not in config:
                    logger.warning(f"配置缺失关键参数: {key}, 使用默认值")
                    # 使用默认值填充缺失的关键参数
                    config[key] = Config.DEFAULT_PARAMS.get(key, 0)
                    
            # 设置默认值
            config.setdefault("model_type", "decoder")
            config.setdefault("dropout", 0.1)
            config.setdefault("max_seq_len", config.get("seq_length", 256))
            
            # 兼容性处理
            if "embedding_dim" not in config:
                config["embedding_dim"] = config["embed_dim"]
            
            # 添加模型版本信息
            config.setdefault("model_version", "1.0")
            
            logger.info(f"加载模型配置: {config}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
            # 尝试创建默认配置作为后备
            logger.info("尝试创建默认配置作为后备")
            config = Config.DEFAULT_PARAMS.copy()
            config["model_version"] = "fallback"
            return config

    def _ensure_model_type(self):
        """确保模型类型正确，自动修复无效类型"""
        model_type = self.config.get("model_type", "").lower()
        
        # 允许的模型类型变体
        valid_types = ["decoder", "gpt", "causal", "generative", "transformer-decoder"]
        
        if model_type not in valid_types:
            logger.warning(f"检测到无效模型类型: '{model_type}'，自动修复为'decoder'")
            self.config["model_type"] = "decoder"
            
            # 尝试更新配置文件
            try:
                config_path = self.model_dir / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    config_data["model_type"] = "decoder"
                    with open(config_path, "w") as f:
                        json.dump(config_data, f, indent=2)
                    logger.info("配置文件已更新，模型类型设置为'decoder'")
            except Exception as e:
                logger.error(f"更新配置文件失败: {str(e)}")

    def _load_tokenizer(self):
        """安全加载分词器，兼容训练特殊标记"""
        try:
            tokenizer_dir = self.model_dir / "vocab"
            
            # 规范化路径处理
            tokenizer_dir = Path(normalize_path(str(tokenizer_dir)))
            
            # 检查分词器目录是否存在
            if not tokenizer_dir.exists() or not tokenizer_dir.is_dir():
                raise FileNotFoundError(f"分词器目录不存在: {tokenizer_dir}")
            
            # 调试：列出目录内容
            try:
                logger.info(f"分词器目录内容: {[f.name for f in tokenizer_dir.iterdir()]}")
            except Exception as list_error:
                logger.warning(f"无法列出分词器目录内容: {str(list_error)}")
            
            # 尝试多种加载方式
            try:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    str(tokenizer_dir),
                    use_fast=True,
                    local_files_only=True,
                    legacy=False,
                    padding_side="left"
                )
            except Exception as first_error:
                logger.warning(f"标准加载失败，尝试替代方法: {str(first_error)}")
                
                # 尝试直接加载tokenizer.json
                tokenizer_file = tokenizer_dir / "tokenizer.json"
                if tokenizer_file.exists():
                    tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=str(tokenizer_file),
                        unk_token="[UNK]",
                        pad_token="[PAD]",
                        cls_token="[CLS]",
                        sep_token="[SEP]",
                        mask_token="[MASK]",
                    )
                else:
                    raise FileNotFoundError(f"未找到tokenizer.json文件: {tokenizer_file}")
            
            # 确保特殊标记与训练时一致
            special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
                "mask_token": "[MASK]"
            }
            
            # 添加缺失的特殊token
            for token_type, token_value in special_tokens.items():
                if getattr(tokenizer, token_type) is None:
                    tokenizer.add_special_tokens({token_type: token_value})
            
            # 词表大小验证
            if len(tokenizer) != self.config["vocab_size"]:
                logger.warning(f"词表不匹配: 配置{self.config['vocab_size']} vs 实际{len(tokenizer)}")
                # 自动调整配置中的词汇表大小
                self.config["vocab_size"] = len(tokenizer)
                
            return tokenizer
        except Exception as e:
            logger.error(f"分词器加载失败: {str(e)}")
            
            # 创建简易后备分词器
            logger.warning("创建简易后备分词器")
            from transformers import BertTokenizerFast
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            
            # 添加特殊token
            tokenizer.add_special_tokens({
                "pad_token": "[PAD]",
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
                "mask_token": "[MASK]"
            })
            
            return tokenizer

    def _load_model(self):
        """增强型模型加载，确保与训练兼容"""
        try:
            # 初始化模型
            model = self._create_model()
            
            # 加载权重
            model_path = self.model_dir / "model.pth"
            if not model_path.exists():
                # 尝试其他可能的模型文件名
                possible_names = ["model.pt", "pytorch_model.bin", "weights.pth"]
                for name in possible_names:
                    alt_path = self.model_dir / name
                    if alt_path.exists():
                        model_path = alt_path
                        logger.info(f"使用备用模型文件: {alt_path}")
                        break
                else:
                    raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
                
            # 根据设备选择加载方式
            map_location = torch.device('cpu') if self.device.type == 'mps' else self.device
            state_dict = torch.load(model_path, map_location=map_location)
            
            # 处理可能的权重前缀
            if all(k.startswith('module.') for k in state_dict.keys()):
                # 去除DataParallel包装的前缀
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 严格参数匹配
            model_state_dict = model.state_dict()
            matched_keys = []
            missing_keys = []
            unexpected_keys = []
            
            for key, param in state_dict.items():
                if key in model_state_dict:
                    # 检查形状是否匹配
                    if param.shape != model_state_dict[key].shape:
                        logger.warning(f"权重形状不匹配: {key} {param.shape} vs {model_state_dict[key].shape}")
                        # 尝试智能调整权重
                        min_dim0 = min(param.size(0), model_state_dict[key].size(0))
                        min_dim1 = min(param.size(1), model_state_dict[key].size(1))
                        model_state_dict[key][:min_dim0, :min_dim1] = param[:min_dim0, :min_dim1]
                        logger.info(f"部分加载权重: {key}")
                    else:
                        model_state_dict[key] = param
                        matched_keys.append(key)
                else:
                    unexpected_keys.append(key)
            
            for key in model_state_dict.keys():
                if key not in state_dict:
                    missing_keys.append(key)
            
            model.load_state_dict(model_state_dict)
            
            if missing_keys:
                logger.warning(f"模型缺失参数: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"模型多余参数: {unexpected_keys}")
                
            # 设备转移
            model = model.to(self.device)
            
            # 混合精度优化
            if self.device.type == "cuda":
                model = model.half()
                
            model.eval()
            logger.info(f"模型成功加载: {len(matched_keys)}个参数匹配")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            # 创建简易后备模型
            logger.warning("创建简易后备模型")
            return self._create_fallback_model()

    def _create_fallback_model(self):
        """创建简易后备模型"""
        class FallbackModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 128)
                self.fc = nn.Linear(128, vocab_size)
            
            def forward(self, x):
                x = self.embedding(x)
                return self.fc(x.mean(dim=1))
        
        model = FallbackModel(self.config["vocab_size"])
        return model.to(self.device)

    def _create_model(self):
        """创建与训练一致的模型架构"""
        class TransformerLM(nn.Module):
            def __init__(self, config):
                super().__init__()
                # 使用配置中的参数名，确保兼容性
                vocab_size = config["vocab_size"]
                embed_dim = config.get("embedding_dim", config.get("embed_dim", 384))
                nhead = config["nhead"]
                num_layers = config["num_layers"]
                dim_feedforward = config["dim_feedforward"]
                dropout = config.get("dropout", 0.1)
                seq_length = config.get("max_seq_len", config.get("seq_length", 256))
                
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.pos_embed = nn.Embedding(seq_length, embed_dim)
                
                # 与训练一致的Transformer Decoder
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True
                )
                self.decoder = nn.TransformerDecoder(
                    decoder_layer,
                    num_layers=num_layers
                )
                self.fc = nn.Linear(embed_dim, vocab_size)
                
                # 初始化权重
                self._init_weights()
                
            def _init_weights(self):
                for name, p in self.named_parameters():
                    if p.dim() > 1:
                        if 'weight' in name:
                            if 'embedding' in name:
                                nn.init.normal_(p, mean=0, std=0.02)
                            elif 'linear' in name:
                                nn.init.xavier_uniform_(p)
                            elif 'norm' in name:
                                nn.init.constant_(p, 1.0)
                        elif 'bias' in name:
                            nn.init.constant_(p, 0.0)
            
            def forward(self, x):
                positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
                x = self.embedding(x) + self.pos_embed(positions)
                
                # 生成因果掩码
                tgt_mask = self.generate_causal_mask(x.size(1)).to(x.device)
                
                # Transformer Decoder
                memory = torch.zeros_like(x)  # 不使用encoder输出
                x = self.decoder(x, memory, tgt_mask=tgt_mask)
                return self.fc(x)
            
            def generate_causal_mask(self, sz):
                mask = torch.triu(torch.ones(sz, sz) * float('-inf'))
                return mask.transpose(0, 1)
        
        return TransformerLM(self.config)

    def _warmup_model(self):
        """模型预热，支持不同输入长度"""
        logger.info("执行模型预热...")
        try:
            input_lengths = [1, 16, 64, 256]  # 不同输入长度
            for length in input_lengths:
                dummy_input = torch.tensor(
                    [[self.tokenizer.pad_token_id] * length], 
                    device=self.device
                )
                with torch.inference_mode():
                    _ = self.model(dummy_input)
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU内存使用情况: 已分配 {torch.cuda.memory_allocated()/1024**2:.2f} MB, "
                            f"已缓存 {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        except Exception as e:
            logger.warning(f"模型预热失败: {str(e)}")

    def generate(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
        """生成文本，增强错误处理"""
        try:
            start_time = time.time()
            
            # 编码输入
            input_ids = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=self.config["max_seq_len"]
            ).to(self.device)
            
            # 生成配置
            generation_config = {
                "max_length": max_length + input_ids.size(1),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # 生成文本
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids, 
                    **generation_config
                )
            
            # 解码结果
            decoded = self.tokenizer.decode(
                output[0], 
                skip_special_tokens=True
            )
            
            # 计算性能指标
            duration = time.time() - start_time
            tokens_generated = output.size(1) - input_ids.size(1)
            tokens_per_sec = tokens_generated / duration if duration > 0 else 0
            
            logger.info(f"生成完成: {tokens_generated} tokens, {tokens_per_sec:.2f} tokens/秒")
            return decoded
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            return f"生成错误: {str(e)}"

class InferenceFrame(ttk.Frame):
    """模型推理框架 - 嵌入在标签页中的推理界面"""
    def __init__(self, parent, model_dir=None):
        super().__init__(parent)
        self.parent = parent
        self.model_dir = model_dir
        self.inference_engine = None
        self.create_widgets()
        
        # 如果提供了模型目录，尝试加载
        if model_dir and Path(model_dir).exists():
            self.load_model(model_dir)

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="模型控制")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # 模型目录选择
        ttk.Label(control_frame, text="模型目录:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_dir_var = tk.StringVar()
        model_dir_entry = ttk.Entry(control_frame, textvariable=self.model_dir_var, width=30)
        model_dir_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.browse_model_dir).grid(row=0, column=2, padx=5, pady=2)
        
        # 加载模型按钮
        ttk.Button(control_frame, text="加载模型", command=self.on_load_model).grid(row=1, column=0, columnspan=3, pady=10)
        
        # 模型状态
        self.model_status = ttk.Label(control_frame, text="模型未加载", foreground="red")
        self.model_status.grid(row=2, column=0, columnspan=3, pady=5)
        
        # 生成参数设置
        params_frame = ttk.LabelFrame(control_frame, text="生成参数")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        ttk.Label(params_frame, text="生成长度:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_length_var = tk.IntVar(value=100)
        ttk.Scale(params_frame, from_=10, to=500, variable=self.max_length_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(params_frame, textvariable=self.max_length_var).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="温度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.temp_var = tk.DoubleVar(value=0.7)
        ttk.Scale(params_frame, from_=0.1, to=1.5, variable=self.temp_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(params_frame, textvariable=self.temp_var).grid(row=1, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Top K:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.top_k_var = tk.IntVar(value=50)
        ttk.Scale(params_frame, from_=1, to=100, variable=self.top_k_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(params_frame, textvariable=self.top_k_var).grid(row=2, column=2, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Top P:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.top_p_var = tk.DoubleVar(value=0.9)
        ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.top_p_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(params_frame, textvariable=self.top_p_var).grid(row=3, column=2, padx=5, pady=2)
        
        # 生成按钮
        ttk.Button(control_frame, text="生成文本", command=self.on_generate).grid(row=4, column=0, columnspan=3, pady=10)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))
        
        # 右侧输入/输出面板
        io_frame = ttk.Frame(main_frame)
        io_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 输入区域
        input_frame = ttk.LabelFrame(io_frame, text="输入提示")
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_text.insert(tk.END, "输入您的提示文本...")
        
        # 输出区域
        output_frame = ttk.LabelFrame(io_frame, text="生成结果")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
        
        # 添加清除按钮
        ttk.Button(io_frame, text="清除结果", command=self.clear_output).pack(side=tk.BOTTOM, pady=5)
        
        # 设置网格列权重
        control_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(1, weight=1)
        
        # 绑定事件
        self.input_text.bind("<FocusIn>", self.on_input_focus)
        self.bind("<Control-r>", lambda e: self.on_generate())
        
    def on_input_focus(self, event):
        """清除输入框的默认文本"""
        if self.input_text.get("1.0", "end-1c") == "输入您的提示文本...":
            self.input_text.delete("1.0", tk.END)
            self.input_text.config(foreground="black")
    
    def clear_output(self):
        """清除输出结果"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
    
    def browse_model_dir(self):
        """浏览模型目录"""
        path = filedialog.askdirectory(
            title="选择模型目录",
            initialdir=str(Path.home())
        )
        if path:
            normalized_path = normalize_path(path)
            self.model_dir_var.set(normalized_path)
            self.load_model(normalized_path)

    def on_load_model(self):
        """加载模型按钮事件"""
        model_dir = self.model_dir_var.get()
        if not model_dir:
            messagebox.showerror("错误", "请选择模型目录")
            return
        self.load_model(model_dir)

    def load_model(self, model_dir):
        """增强型模型加载，提供详细错误信息"""
        try:
            self.status_var.set("加载模型中...")
            self.update()
            
            # 在后台线程加载模型
            threading.Thread(
                target=self._load_model_thread, 
                args=(model_dir,),
                daemon=True
            ).start()
        except Exception as e:
            self._handle_load_error(f"加载失败: {str(e)}")

    def _load_model_thread(self, model_dir):
        """在后台线程加载模型"""
        try:
            self.inference_engine = InferenceEngine(model_dir)
            
            # 获取模型信息
            model_info = f"模型加载成功: {model_dir}\n"
            model_info += f"模型类型: {self.inference_engine.config['model_type']}\n"
            model_info += f"参数数量: {sum(p.numel() for p in self.inference_engine.model.parameters()):,}\n"
            model_info += f"词表大小: {self.inference_engine.config['vocab_size']}"
            
            self.model_status.config(text=model_info, foreground="green")
            self.status_var.set("模型加载成功")
            logger.info(f"模型加载成功: {model_dir}")
            
            # 自动生成欢迎文本
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, "你好，模型已成功加载！")
            self.on_generate()
        except Exception as e:
            self._handle_load_error(f"加载失败: {str(e)}")

    def _handle_load_error(self, error_msg):
        """统一处理加载错误"""
        self.model_status.config(text=error_msg, foreground="red")
        self.status_var.set(error_msg)
        logger.error(error_msg)
        
        # 提供详细错误信息
        if "配置文件" in error_msg or "config" in error_msg:
            messagebox.showerror("配置错误", 
                "模型配置文件(config.json)存在问题:\n\n"
                f"{error_msg}\n\n"
                "请检查配置文件格式和内容是否正确。")
        elif "分词器" in error_msg or "tokenizer" in error_msg:
            messagebox.showerror("分词器错误", 
                "分词器加载失败:\n\n"
                f"{error_msg}\n\n"
                "可能原因:\n"
                "1. 模型目录缺少vocab子目录\n"
                "2. 分词器文件损坏\n"
                "3. 分词器版本不兼容\n\n"
                "解决方案:\n"
                "1. 删除模型目录下的 'vocab' 文件夹\n"
                "2. 重新开始训练\n"
                "3. 如果问题仍然存在，尝试使用纯英文路径")
        elif "模型权重" in error_msg or "model.pth" in error_msg:
            messagebox.showerror("模型错误", 
                "模型权重加载失败:\n\n"
                f"{error_msg}\n\n"
                "可能原因:\n"
                "1. 模型文件损坏\n"
                "2. 模型架构与权重不匹配\n"
                "3. 文件权限问题\n"
                "4. PyTorch版本不兼容")
        else:
            messagebox.showerror("加载错误", 
                f"加载模型时发生未知错误:\n\n{error_msg}\n\n"
                "建议检查日志文件获取更多信息。")

    def on_generate(self):
        """生成文本事件处理"""
        if not self.inference_engine:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt or prompt == "输入您的提示文本...":
            messagebox.showinfo("提示", "请输入提示文本")
            return
        
        self.status_var.set("生成中...")
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "生成中，请稍候...")
        self.output_text.config(state=tk.DISABLED)
        self.update()
        
        # 在后台线程生成文本
        threading.Thread(
            target=self._generate_thread,
            args=(prompt,),
            daemon=True
        ).start()
    
    def _generate_thread(self, prompt):
        """在后台线程生成文本"""
        try:
            start_time = time.time()
            
            # 获取生成参数
            max_length = self.max_length_var.get()
            temperature = self.temp_var.get()
            top_k = self.top_k_var.get()
            top_p = self.top_p_var.get()
            
            # 生成文本
            generated = self.inference_engine.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # 更新UI
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, generated)
            self.output_text.config(state=tk.DISABLED)
            
            # 更新状态
            duration = time.time() - start_time
            self.status_var.set(f"生成完成 ({duration:.2f}秒)")
            
        except Exception as e:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"生成错误: {str(e)}")
            self.output_text.config(state=tk.DISABLED)
            self.status_var.set(f"生成错误: {str(e)}")
            logger.error(f"生成失败: {str(e)}")

class TrainingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformer语言模型训练器")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建标签页
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 训练标签页
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="模型训练")
        self.create_train_tab()
        
        # 推理标签页
        self.infer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.infer_tab, text="模型推理")
        self.create_infer_tab()
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 训练状态
        self.is_training = False
        self.trainer = None
        self.training_thread = None
        
        # 设置应用程序图标
        try:
            self.iconbitmap("app_icon.ico")
        except:
            pass
    
    def create_train_tab(self):
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.train_tab, text="训练控制")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # 训练文件选择
        ttk.Label(control_frame, text="训练文件:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_file_var = tk.StringVar()
        train_file_entry = ttk.Entry(control_frame, textvariable=self.train_file_var, width=30)
        train_file_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.browse_train_file).grid(row=0, column=2, padx=5, pady=2)
        
        # 模型目录选择
        ttk.Label(control_frame, text="模型目录:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.model_dir_var = tk.StringVar()
        model_dir_entry = ttk.Entry(control_frame, textvariable=self.model_dir_var, width=30)  # 修复：ttk 而不是 tttk
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
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="序列长度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.seq_length_var = tk.IntVar(value=256)
        ttk.Entry(params_frame, textvariable=self.seq_length_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="学习率:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value="3e-4")
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="最大轮数:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="目标损失:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.target_loss_var = tk.DoubleVar(value=1.5)
        ttk.Entry(params_frame, textvariable=self.target_loss_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 按钮区域
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="开始训练", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止训练", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧日志面板
        log_frame = ttk.LabelFrame(self.train_tab, text="训练日志")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # 设置网格列权重
        control_frame.columnconfigure(1, weight=1)
    
    def create_infer_tab(self):
        # 在推理标签页中创建推理框架
        self.inference_frame = InferenceFrame(self.infer_tab)
        self.inference_frame.pack(fill=tk.BOTH, expand=True)
    
    def browse_train_file(self):
        """浏览训练文件"""
        file_path = filedialog.askopenfilename(
            title="选择训练文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            normalized_path = normalize_path(file_path)
            self.train_file_var.set(normalized_path)
    
    def browse_model_dir(self):
        """浏览模型目录"""
        dir_path = filedialog.askdirectory(
            title="选择模型目录",
            initialdir=str(Path.home())
        )
        if dir_path:
            normalized_path = normalize_path(dir_path)
            self.model_dir_var.set(normalized_path)
    
    def start_training(self):
        """开始训练"""
        if self.is_training:
            return
            
        train_file = self.train_file_var.get()
        model_dir = self.model_dir_var.get()
        
        if not train_file or not Path(train_file).exists():
            messagebox.showerror("错误", "请选择有效的训练文件")
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
            "vocab_size": 30000,
            "embed_dim": 384,
            "nhead": 6,
            "num_layers": 8,
            "dim_feedforward": 1536,
            "early_stop_patience": 10,
            "min_lr": 1e-6,
            "weight_decay": 1e-5,
            "dropout": 0.1
        }
        
        # 更新UI状态
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("训练中...")
        
        # 在后台线程中启动训练
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(config, train_file, model_dir, self.device_var.get()),
            daemon=True
        )
        self.training_thread.start()
    
    def run_training(self, config, train_file, model_dir, device):
        """执行训练任务"""
        try:
            # 初始化训练器
            self.trainer = ModelTrainer(config, train_file, model_dir, device=device)
            
            # 开始训练
            self.trainer.train()
            
            # 训练完成
            self.status_var.set("训练完成")
            self.log_message("训练完成！")
            
        except Exception as e:
            self.log_message(f"训练出错: {str(e)}", level="error")
            self.status_var.set(f"训练出错: {str(e)}")
        
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
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [{level.upper()}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 同时输出到控制台
        print(f"{time.strftime('%H:%M:%S')} [{level.upper()}] {message}")
    
    def on_close(self):
        """关闭应用程序"""
        if self.is_training:
            if messagebox.askyesno("确认", "训练仍在进行中，确定要退出吗？"):
                self.stop_training()
                self.destroy()
        else:
            self.destroy()

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, file_path, tokenizer, seq_length=256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text = self.load_and_preprocess_text()
        
        # 修复编码处理 - 直接使用编码结果
        self.encoded_text = self.tokenizer.encode(self.text)  # 移除 .ids
        
    def load_and_preprocess_text(self):
        """加载并预处理文本"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 简单的文本清理
        text = re.sub(r'\s+', ' ', text)  # 替换多个空白字符为单个空格
        text = text.strip()
        return text
    
    def __len__(self):
        """返回数据集长度"""
        return max(0, len(self.encoded_text) // self.seq_length)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 用于创建目标序列
        
        # 提取输入和目标
        segment = self.encoded_text[start_idx:end_idx]
        if len(segment) < self.seq_length + 1:
            # 填充不足的序列
            padding = [self.tokenizer.pad_token_id] * (self.seq_length + 1 - len(segment))
            segment.extend(padding)
        
        input_ids = segment[:self.seq_length]
        target_ids = segment[1:self.seq_length+1]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

class TransformerModel(nn.Module):
    """Transformer语言模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.embedding = nn.Embedding(
            config['vocab_size'], 
            config['embed_dim'],
            padding_idx=0
        )
        
        # 位置编码
        self.position_embedding = nn.Embedding(
            config['seq_length'], 
            config['embed_dim']
        )
        
        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['embed_dim'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        
        # Transformer 解码器
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['num_layers']
        )
        
        # 输出层
        self.fc_out = nn.Linear(config['embed_dim'], config['vocab_size'])
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        # 词嵌入层初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])
        
        # 位置编码初始化
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # 输出层初始化
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
        
        # Transformer层初始化
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids):
        """前向传播"""
        # 获取序列长度
        seq_len = input_ids.size(1)
        
        # 创建位置索引
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        
        # 嵌入层
        input_emb = self.embedding(input_ids) + self.position_embedding(positions)
        
        # 创建因果掩码
        causal_mask = self.generate_causal_mask(seq_len).to(input_ids.device)
        
        # Transformer
        # 注意：对于纯解码器模型，我们使用一个虚拟的memory输入
        memory = torch.zeros_like(input_emb)  # 创建一个与输入形状相同的零张量
        output = self.transformer(
            tgt=input_emb,
            memory=memory,
            tgt_mask=causal_mask
        )
        
        # 输出层
        logits = self.fc_out(output)
        return logits
    
    def generate_causal_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

class ModelTrainer:
    def __init__(self, config, train_file, model_dir, device='auto', new_model=True):
        self.config = config
        self.train_file = train_file
        self.model_dir = Path(model_dir)
        self.new_model = new_model
        self.stop_training = False
        
        # 确保模型目录存在
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置并添加模型类型
        self.config["model_type"] = "decoder"  # 强制设置为decoder类型
        self._save_config()
        
        # 设备选择
        self.device = self._select_device(device)
        
        # 初始化模型和分词器
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model().to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            min_lr=self.config["min_lr"]
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # 训练状态
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"模型训练器初始化完成 (设备: {self.device}, 模型类型: {self.config['model_type']})")
    
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
        """初始化分词器"""
        tokenizer_dir = self.model_dir / "vocab"
        tokenizer_dir.mkdir(exist_ok=True)
        
        # 如果分词器不存在，则训练一个新的
        if self.new_model or not (tokenizer_dir / "tokenizer.json").exists():
            # 创建ByteLevelBPETokenizer实例
            tokenizer = ByteLevelBPETokenizer()
            
            # 训练分词器
            tokenizer.train(
                files=[self.train_file],
                vocab_size=self.config["vocab_size"],
                min_frequency=2,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            )
            
            # 保存分词器文件 - 确保生成所有必要文件
            tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
            
            # 创建PreTrainedTokenizerFast实例并保存为Hugging Face格式
            fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )
            fast_tokenizer.save_pretrained(str(tokenizer_dir))
            logger.info(f"分词器已保存到: {tokenizer_dir}")
        
        # 加载分词器 - 使用PreTrainedTokenizerFast
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                str(tokenizer_dir),
                use_fast=True,
                local_files_only=True,
                legacy=False,
                padding_side="left"
            )
        except OSError as e:
            # 如果标准方式失败，尝试替代加载方式
            logger.warning(f"标准分词器加载失败，尝试替代方法: {str(e)}")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )
        
        # 添加特殊token（确保它们存在）
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[CLS]",
            "eos_token": "[SEP]",
            "mask_token": "[MASK]"
        }
        
        for token_type, token_value in special_tokens.items():
            if getattr(tokenizer, token_type) is None:
                tokenizer.add_special_tokens({token_type: token_value})
        
        # 验证词表大小
        if len(tokenizer) != self.config["vocab_size"]:
            logger.warning(f"词表大小不匹配: 配置{self.config['vocab_size']} vs 实际{len(tokenizer)}")
            self.config["vocab_size"] = len(tokenizer)
        
        return tokenizer
    
    def _init_model(self):
        """初始化模型"""
        # 创建新模型或加载现有模型
        model_path = self.model_dir / "model.pth"
        if not self.new_model and model_path.exists():
            model = TransformerModel(self.config)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info(f"从 {model_path} 加载现有模型")
        else:
            model = TransformerModel(self.config)
            logger.info("创建新模型")
        
        return model
    
    def _save_config(self):
        """保存配置到文件，添加版本信息和时间戳"""
        config_path = self.model_dir / "config.json"
        config_data = self.config.copy()
        config_data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        config_data["model_version"] = "1.1"  # 更新模型版本
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"配置已保存到: {config_path}")
    
    def create_data_loaders(self):
        """创建训练和验证数据加载器"""
        # 创建数据集
        dataset = TextDataset(
            self.train_file,
            self.tokenizer,
            self.config["seq_length"]
        )
        
        # 划分训练集和验证集
        total_size = len(dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"数据集大小: 总计 {total_size}, 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if self.stop_training:
                logger.info("训练已停止")
                return None
                
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            
            # 计算损失
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1)
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 每10个batch记录一次
            if batch_idx % 10 == 0:
                avg_loss = total_loss / total_samples
                logger.info(f"训练批次 {batch_idx}/{len(train_loader)} - 损失: {avg_loss:.4f}")
        
        return total_loss / total_samples
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移动到设备
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
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
        
        return total_loss / total_samples
    
    def save_model(self, epoch, val_loss):
        """保存模型"""
        model_path = self.model_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"模型已保存到: {model_path} (Epoch {epoch}, 损失 {val_loss:.4f})")
        
        # 保存训练历史
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        start_time = time.time()
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders()
        
        # 训练循环
        for epoch in range(1, self.config["epochs"] + 1):
            if self.stop_training:
                logger.info("训练已停止")
                break
                
            epoch_start = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            if train_loss is None:  # 训练被停止
                break
                
            # 验证
            val_loss = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)
            
            # 记录epoch信息
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch}/{self.config['epochs']} - "
                        f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                        f"学习率: {current_lr:.2e}, 时间: {epoch_time:.2f}s")
            
            # 检查最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_model(epoch, val_loss)
            else:
                self.epochs_without_improvement += 1
            
            # 提前停止检查
            if self.epochs_without_improvement >= self.config["early_stop_patience"]:
                logger.info(f"验证损失在 {self.config['early_stop_patience']} 个epoch内没有改善，提前停止")
                break
            
            # 达到目标损失
            if val_loss <= self.config["target_loss"]:
                logger.info(f"达到目标损失 {self.config['target_loss']}，停止训练")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成! 总时间: {total_time:.2f}秒, 最佳验证损失: {self.best_loss:.4f}")

if __name__ == "__main__":
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
    
    try:
        # 设置DPI感知
        if sys.platform == "win32":
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        
        app = TrainingApp()
        app.mainloop()
    except Exception as e:
        logger.critical(f"程序崩溃: {str(e)}", exc_info=True)
        messagebox.showerror("严重错误", f"程序发生未处理异常:\n{str(e)}")