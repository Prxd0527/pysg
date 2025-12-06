"""
GUI应用模块 - 水果识别系统图形用户界面
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from src.predictor import FruitPredictor
from utils.config import WINDOW_TITLE, WINDOW_SIZE, PREVIEW_SIZE, SUPPORTED_FORMATS, MODEL_PATH
from utils.image_utils import resize_image_for_display


class FruitRecognitionGUI:
    """水果识别GUI应用"""
    
    def __init__(self, root):
        """
        初始化GUI应用
        
        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.resizable(True, True)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 变量
        self.current_image_path = None
        self.predictor = None
        self.photo_image = None
        
        # 初始化预测器
        self.init_predictor()
        
        # 创建GUI组件
        self.create_widgets()
        
        # 居中窗口
        self.center_window()
    
    def init_predictor(self):
        """初始化预测器"""
        try:
            if os.path.exists(MODEL_PATH):
                self.predictor = FruitPredictor()
            else:
                messagebox.showwarning(
                    "模型未找到",
                    f"模型文件不存在: {MODEL_PATH}\n\n请先训练模型后再使用。"
                )
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
    
    def create_widgets(self):
        """创建GUI组件"""
        # 设置主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 标题
        title_label = tk.Label(
            main_frame,
            text="🍎 水果识别系统",
            font=("Arial", 24, "bold"),
            fg="#2E7D32"
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # 内容框架
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # 左侧面板 - 图片显示
        left_frame = ttk.LabelFrame(content_frame, text="图片预览", padding="10")
        left_frame.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图片显示区域
        self.image_label = tk.Label(
            left_frame,
            text="点击下方按钮选择图片",
            bg="#F5F5F5",
            width=50,
            height=20,
            relief=tk.SUNKEN
        )
        self.image_label.pack(expand=True, fill=tk.BOTH, pady=(0, 10))
        
        # 按钮框架
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X)
        
        # 选择图片按钮
        self.select_btn = ttk.Button(
            button_frame,
            text="📁 选择图片",
            command=self.select_image
        )
        self.select_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 识别按钮
        self.predict_btn = ttk.Button(
            button_frame,
            text="🔍 开始识别",
            command=self.predict_image,
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT)
        
        # 右侧面板 - 识别结果
        right_frame = ttk.LabelFrame(content_frame, text="识别结果", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # 结果显示区域
        self.result_text = tk.Text(
            right_frame,
            font=("Consolas", 11),
            wrap=tk.WORD,
            bg="#FAFAFA",
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text['yscrollcommand'] = scrollbar.set
        
        # 进度条
        self.progress = ttk.Progressbar(right_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 状态栏
        self.status_label = tk.Label(
            main_frame,
            text="就绪",
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#E8F5E9"
        )
        self.status_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 初始化结果显示
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        welcome_text = """
═══════════════════════════════════════════
    欢迎使用水果识别系统！
═══════════════════════════════════════════

📋 使用说明:
1. 点击「选择图片」按钮上传水果图片
2. 点击「开始识别」进行识别
3. 查看识别结果和置信度

🍎 支持的水果类别:
- 苹果 (Apple)
- 香蕉 (Banana)
- 葡萄 (Grape)
- 芒果 (Mango)
- 橙子 (Orange)
- 西瓜 (Watermelon)

💡 提示:
- 支持JPG、PNG等常见图片格式
- 图片越清晰,识别准确率越高
- 建议使用正面拍摄的水果图片

═══════════════════════════════════════════
        """
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, welcome_text)
    
    def center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def select_image(self):
        """选择图片"""
        file_path = filedialog.askopenfilename(
            title="选择水果图片",
            filetypes=SUPPORTED_FORMATS
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn['state'] = tk.NORMAL
            self.status_label['text'] = f"已选择: {os.path.basename(file_path)}"
    
    def display_image(self, image_path):
        """
        显示图片
        
        Args:
            image_path: 图片路径
        """
        try:
            # 调整图片大小
            img = resize_image_for_display(image_path, PREVIEW_SIZE)
            if img:
                self.photo_image = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.photo_image, text="")
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def predict_image(self):
        """识别图片"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择图片！")
            return
        
        if not self.predictor:
            messagebox.showerror("错误", "预测器未初始化！")
            return
        
        # 在新线程中进行预测，避免界面冻结
        thread = threading.Thread(target=self._predict_thread)
        thread.daemon = True
        thread.start()
    
    def _predict_thread(self):
        """预测线程"""
        # 更新UI
        self.root.after(0, self._start_prediction)
        
        try:
            # 进行预测
            result = self.predictor.predict(self.current_image_path)
            
            # 更新结果
            self.root.after(0, lambda: self._show_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _start_prediction(self):
        """开始预测时的UI更新"""
        self.predict_btn['state'] = tk.DISABLED
        self.select_btn['state'] = tk.DISABLED
        self.progress.start(10)
        self.status_label['text'] = "正在识别..."
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "🔄 正在识别中，请稍候...\n")
    
    def _show_result(self, result):
        """
        显示识别结果
        
        Args:
            result: 预测结果字典
        """
        self.progress.stop()
        self.predict_btn['state'] = tk.NORMAL
        self.select_btn['state'] = tk.NORMAL
        
        if not result['success']:
            self.status_label['text'] = "识别失败"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, f"❌ 识别失败: {result.get('error', '未知错误')}")
            return
        
        # 格式化结果
        result_text = f"""
═══════════════════════════════════════════
       🎯 识别结果
═══════════════════════════════════════════

🍎 预测类别: {result['predicted_class_cn']} ({result['predicted_class']})
📊 置信度: {result['confidence_percent']:.2f}%

───────────────────────────────────────────
       📋 详细概率分布
───────────────────────────────────────────

Top 3 预测结果:
"""
        
        for i, pred in enumerate(result['top_3_predictions'], 1):
            bar_length = int(pred['probability_percent'] / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            result_text += f"\n{i}. {pred['class_cn']:8s} ({pred['class']:12s})\n"
            result_text += f"   [{bar}] {pred['probability_percent']:5.2f}%\n"
        
        result_text += "\n═══════════════════════════════════════════\n"
        
        # 显示结果
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        
        # 更新状态栏
        self.status_label['text'] = f"识别完成: {result['predicted_class_cn']} ({result['confidence_percent']:.1f}%)"
    
    def _show_error(self, error_message):
        """显示错误信息"""
        self.progress.stop()
        self.predict_btn['state'] = tk.NORMAL
        self.select_btn['state'] = tk.NORMAL
        self.status_label['text'] = "识别出错"
        messagebox.showerror("错误", f"识别过程出错: {error_message}")


def run_gui():
    """运行GUI应用"""
    root = tk.Tk()
    app = FruitRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
