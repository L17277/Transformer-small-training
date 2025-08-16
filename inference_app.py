import sys
import os
import re
import json
import time
import logging
import threading
import traceback
import torch
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
        logging.FileHandler(f'inference_{time.strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型定义 - 与训练程序中的模型结构保持一致
class TransformerLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词汇表大小和嵌入维度
        self.vocab_size = config['vocab_size']
        self.embed_dim = config.get('embed_dim', config.get('embedding_dim', 384))
        
        # 词嵌入层
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim)
        
        # 位置编码
        max_seq_len = config.get('max_seq_len', config.get('seq_length', 256))
        self.pos_embed = torch.nn.Embedding(max_seq_len, self.embed_dim)
        
        # Transformer 解码器层
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        
        # Transformer 解码器
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['num_layers']
        )
        
        # 输出层
        self.fc = torch.nn.Linear(self.embed_dim, self.vocab_size)
        
        # 层归一化
        self.ln = torch.nn.LayerNorm(self.embed_dim)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 创建位置索引
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 词嵌入 + 位置嵌入
        x_emb = self.embedding(x) + self.pos_embed(positions)
        x_emb = self.ln(x_emb)
        
        # 创建目标掩码
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer 解码器
        output = self.decoder(
            tgt=x_emb, 
            memory=x_emb, 
            tgt_mask=tgt_mask
        )
        
        # 输出层
        logits = self.fc(output)
        
        return logits

# 模型推理封装类
class ModelInference:
    def __init__(self, model_dir, device='auto', app=None):
        self.model_dir = Path(model_dir)
        self.app = app  # 对GUI应用的引用
        self.device = self._select_device(device)
        self.tokenizer = None
        self.model = None
        self.config = None
        self.loaded = False
        
        # 检查模型目录是否存在
        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        
        # 加载配置
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载分词器
        vocab_dir = self.model_dir / "vocab"
        if not vocab_dir.exists():
            raise FileNotFoundError(f"分词器目录不存在: {vocab_dir}")
        
        vocab_file = vocab_dir / "vocab.json"
        merges_file = vocab_dir / "merges.txt"
        
        if not vocab_file.exists() or not merges_file.exists():
            raise FileNotFoundError("分词器文件不存在")
        
        # 加载分词器
        self.tokenizer = ByteLevelBPETokenizer(
            str(vocab_file),
            str(merges_file)
        )
        
        # 添加特殊token
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([token])
        
        # 加载模型
        model_path = self.model_dir / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化模型
        self.model = TransformerLM(self.config).to(self.device)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 修复键名（如果需要）
        fixed_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if "transformer.layers" in new_key:
                new_key = new_key.replace("transformer.layers", "decoder.layers")
            if "fc_out" in new_key:
                new_key = new_key.replace("fc_out", "fc")
            if "position_embedding" in new_key:
                new_key = new_key.replace("position_embedding", "pos_embed")
            fixed_state_dict[new_key] = value
        
        # 加载修复后的权重
        self.model.load_state_dict(fixed_state_dict)
        self.model.eval()  # 设置为评估模式
        
        self.loaded = True
        logger.info(f"模型加载完成 (设备: {self.device})")
        if self.app:
            self.app.log_message(f"模型加载完成 (设备: {self.device})")
    
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
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
        """生成文本"""
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        # 编码提示文本
        encoded = self.tokenizer.encode(prompt)
        input_ids = encoded.ids
        
        # 将输入转换为张量
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 生成文本
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # 获取模型输出
                outputs = self.model(input_tensor)
                
                # 获取最后一个token的logits
                next_token_logits = outputs[0, -1, :]
                
                # 应用温度
                next_token_logits = next_token_logits / temperature
                
                # 应用top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # 应用top-p过滤
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 将第一个token设为False以确保至少有一个token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # 从logits中采样下一个token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
                
                # 添加到生成的序列中
                generated_ids.append(next_token_id.item())
                
                # 如果生成了[SEP]或[PAD]则停止
                if next_token_id in [self.tokenizer.token_to_id("[SEP]"), self.tokenizer.token_to_id("[PAD]")]:
                    break
                
                # 更新输入
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)
                
                # 如果输入序列太长，截断
                if input_tensor.size(1) > self.config.get("max_seq_len", 256):
                    input_tensor = input_tensor[:, -self.config.get("max_seq_len", 256):]
        
        # 解码生成的文本
        generated_tokens = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 移除特殊标记
        generated_text = re.sub(r'\[[A-Z]+\]', '', generated_tokens)
        
        return prompt + generated_text

# GUI推理应用程序
class InferenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformer语言模型推理器")
        self.geometry("800x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 模型状态
        self.model_loaded = False
        self.inference = None
        
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.main_frame, text="模型控制")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # 模型目录选择
        ttk.Label(control_frame, text="模型目录:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_dir_var = tk.StringVar(value=str(Path.home() / "transformer_model"))
        model_dir_entry = ttk.Entry(control_frame, textvariable=self.model_dir_var, width=30)
        model_dir_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Button(control_frame, text="浏览...", command=self.browse_model_dir).grid(row=0, column=2, padx=5, pady=2)
        
        # 设备选择
        ttk.Label(control_frame, text="推理设备:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar(value="auto")
        device_combobox = ttk.Combobox(control_frame, textvariable=self.device_var, width=28)
        device_combobox['values'] = ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps')
        device_combobox.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # 加载模型按钮
        ttk.Button(control_frame, text="加载模型", command=self.load_model).grid(row=2, column=0, columnspan=3, pady=10)
        
        # 参数设置
        params_frame = ttk.LabelFrame(control_frame, text="生成参数")
        params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        ttk.Label(params_frame, text="生成长度:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.max_length_var = tk.IntVar(value=100)
        ttk.Entry(params_frame, textvariable=self.max_length_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="温度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.temperature_var = tk.DoubleVar(value=0.7)
        ttk.Entry(params_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Top K:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.top_k_var = tk.IntVar(value=50)
        ttk.Entry(params_frame, textvariable=self.top_k_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Top P:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.top_p_var = tk.DoubleVar(value=0.9)
        ttk.Entry(params_frame, textvariable=self.top_p_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 提示输入
        prompt_frame = ttk.LabelFrame(self.main_frame, text="输入提示")
        prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=5, wrap=tk.WORD, font=("Consolas", 10))
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.prompt_text.insert(tk.END, "从前，在一个遥远的王国里")
        
        # 生成按钮
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.generate_btn = ttk.Button(btn_frame, text="生成文本", command=self.generate_text, state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # 输出区域
        output_frame = ttk.LabelFrame(self.main_frame, text="生成结果")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
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
        
        # 设置网格列权重
        control_frame.columnconfigure(1, weight=1)
    
    def log_message(self, message, level="info"):
        """记录日志消息"""
        def update_log():
            # 在输出文本框中显示消息
            self.output_text.config(state=tk.NORMAL)
            # 添加带颜色的标签
            if level == "error":
                self.output_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [ERROR] ", "error")
            elif level == "warning":
                self.output_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [WARNING] ", "warning")
            else:
                self.output_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} [INFO] ", "info")
            
            self.output_text.insert(tk.END, f"{message}\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
            
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
    
    def browse_model_dir(self):
        """浏览模型目录"""
        dir_path = filedialog.askdirectory(
            title="选择模型目录",
            initialdir=str(Path.home())
        )
        if dir_path:
            self.model_dir_var.set(dir_path)
    
    def load_model(self):
        """加载模型"""
        model_dir = self.model_dir_var.get()
        
        if not model_dir:
            messagebox.showerror("错误", "请选择模型目录")
            return
            
        if not os.path.exists(model_dir):
            messagebox.showerror("错误", "模型目录不存在")
            return
        
        # 更新UI状态
        self.status_var.set("加载模型中...")
        self.update_progress(0, "加载模型...")
        
        # 在后台线程中加载模型
        threading.Thread(
            target=self._load_model_thread,
            args=(model_dir,),
            daemon=True
        ).start()
    
    def _load_model_thread(self, model_dir):
        """后台线程加载模型"""
        try:
            self.inference = ModelInference(
                model_dir, 
                device=self.device_var.get(),
                app=self
            )
            self.model_loaded = True
            self.generate_btn.config(state=tk.NORMAL)
            self.status_var.set("模型加载成功")
            self.log_message("模型加载成功")
            self.update_progress(100, "模型加载完成!")
        except Exception as e:
            self.log_message(f"加载模型失败: {str(e)}", level="error")
            self.log_message(traceback.format_exc(), level="error")
            self.status_var.set(f"加载模型失败: {str(e)}")
            self.update_progress(0, "加载失败")
    
    def generate_text(self):
        """生成文本"""
        if not self.model_loaded or self.inference is None:
            messagebox.showerror("错误", "模型未加载")
            return
        
        # 获取提示文本
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("错误", "请输入提示文本")
            return
        
        # 获取生成参数
        max_length = self.max_length_var.get()
        temperature = self.temperature_var.get()
        top_k = self.top_k_var.get()
        top_p = self.top_p_var.get()
        
        # 更新UI状态
        self.status_var.set("生成文本中...")
        self.update_progress(0, "生成文本...")
        
        # 在后台线程中生成文本
        threading.Thread(
            target=self._generate_text_thread,
            args=(prompt, max_length, temperature, top_k, top_p),
            daemon=True
        ).start()
    
    def _generate_text_thread(self, prompt, max_length, temperature, top_k, top_p):
        """后台线程生成文本"""
        try:
            # 生成文本
            start_time = time.time()
            
            # 在输出区域显示提示
            self.log_message(f"提示: {prompt}")
            self.log_message(f"开始生成文本 (长度={max_length}, 温度={temperature}, top_k={top_k}, top_p={top_p})")
            
            # 分步生成文本，更新进度
            generated_text = ""
            step = max(1, max_length // 20)  # 每5%更新一次进度
            
            for i in range(0, max_length, step):
                # 生成下一部分文本
                partial_length = min(step, max_length - i)
                generated = self.inference.generate_text(
                    prompt + generated_text,
                    max_length=partial_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # 获取新生成的部分
                new_text = generated[len(prompt + generated_text):]
                generated_text += new_text
                
                # 更新输出
                self.log_message(f"生成进度: {len(generated_text)}/{max_length} 字符")
                self.log_message(f"新内容: {new_text}")
                
                # 更新进度
                progress = min(100, (i + step) / max_length * 100)
                self.update_progress(progress, f"生成中: {progress:.1f}%")
            
            # 显示最终结果
            self.log_message("\n最终生成结果:")
            self.log_message(generated_text)
            
            # 计算时间
            gen_time = time.time() - start_time
            tokens_per_sec = max_length / gen_time if gen_time > 0 else 0
            
            # 完成消息
            self.log_message(f"\n生成完成! 总时间: {gen_time:.2f}秒, 速度: {tokens_per_sec:.1f} 字符/秒")
            self.update_progress(100, "生成完成!")
            self.status_var.set("文本生成完成")
        except Exception as e:
            self.log_message(f"生成文本失败: {str(e)}", level="error")
            self.log_message(traceback.format_exc(), level="error")
            self.status_var.set(f"生成失败: {str(e)}")
            self.update_progress(0, "生成失败")
    
    def on_close(self):
        """关闭应用程序"""
        self.destroy()

# 主函数
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
    app = InferenceApp()
    
    # 配置标签样式
    app.output_text.tag_config("error", foreground="red")
    app.output_text.tag_config("warning", foreground="orange")
    app.output_text.tag_config("info", foreground="black")
    
    app.mainloop()

if __name__ == "__main__":
    main()