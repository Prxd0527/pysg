"""
GUI应用模块 - 水果识别系统图形用户界面（美化版）
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font
from PIL import Image, ImageTk, ImageDraw
import threading
import cv2
import numpy as np
from datetime import datetime
from src.predictor import FruitPredictor
from utils.config import WINDOW_TITLE, WINDOW_SIZE, PREVIEW_SIZE, SUPPORTED_FORMATS, MODEL_PATH
from utils.image_utils import resize_image_for_display


# ==================== 颜色主题配置 ====================
class Theme:
    """现代化颜色主题"""
    # 主色调
    PRIMARY = "#4CAF50"       # 绿色 - 主按钮
    PRIMARY_DARK = "#388E3C"  # 深绿色 - 悬停
    PRIMARY_LIGHT = "#81C784" # 浅绿色
    
    # 辅助色
    SECONDARY = "#FF9800"     # 橙色 - 强调
    SECONDARY_DARK = "#F57C00"
    
    # 背景色
    BG_MAIN = "#F0F4F8"       # 主背景 - 浅蓝灰
    BG_CARD = "#FFFFFF"       # 卡片背景 - 白色
    BG_DARK = "#E8ECF0"       # 深背景
    
    # 文字色
    TEXT_PRIMARY = "#2D3748"   # 主要文字 - 深灰
    TEXT_SECONDARY = "#718096" # 次要文字 - 中灰
    TEXT_WHITE = "#FFFFFF"     # 白色文字
    
    # 边框色
    BORDER = "#E2E8F0"        # 边框 - 浅灰
    BORDER_FOCUS = "#4CAF50"  # 聚焦边框
    
    # 状态色
    SUCCESS = "#48BB78"       # 成功 - 绿色
    WARNING = "#ECC94B"       # 警告 - 黄色
    ERROR = "#F56565"         # 错误 - 红色
    INFO = "#4299E1"          # 信息 - 蓝色


# ==================== 自定义组件 ====================
class ModernButton(tk.Canvas):
    """现代化按钮组件"""
    
    def __init__(self, parent, text, command=None, width=180, height=45, 
                 bg_color=Theme.PRIMARY, hover_color=Theme.PRIMARY_DARK,
                 fg_color=Theme.TEXT_WHITE, icon="", **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=parent.cget('bg'), highlightthickness=0, **kwargs)
        
        self.command = command
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.text = icon + " " + text if icon else text
        self.enabled = True
        
        self._draw_button(self.bg_color)
        
        # 绑定事件
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
    
    def _draw_button(self, color):
        """绘制圆角按钮"""
        self.delete("all")
        radius = 10
        
        # 绘制圆角矩形
        self._create_rounded_rect(2, 2, self.width-2, self.height-2, radius, fill=color, outline="")
        
        # 绘制文字
        self.create_text(self.width//2, self.height//2, text=self.text, 
                        fill=self.fg_color, font=("微软雅黑", 11, "bold"))
    
    def _create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """创建圆角矩形"""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_enter(self, event):
        if self.enabled:
            self._draw_button(self.hover_color)
            self.config(cursor="hand2")
    
    def _on_leave(self, event):
        if self.enabled:
            self._draw_button(self.bg_color)
            self.config(cursor="")
    
    def _on_click(self, event):
        if self.enabled and self.command:
            self.command()
    
    def set_enabled(self, enabled):
        """设置按钮启用状态"""
        self.enabled = enabled
        if enabled:
            self._draw_button(self.bg_color)
        else:
            self._draw_button(Theme.BG_DARK)


class Card(tk.Frame):
    """卡片式容器组件"""
    
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, **kwargs)
        
        # 标题
        if title:
            title_frame = tk.Frame(self, bg=Theme.BG_CARD)
            title_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
            
            tk.Label(title_frame, text=title, font=("微软雅黑", 12, "bold"),
                    bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(anchor=tk.W)
            
            # 分隔线
            separator = tk.Frame(self, height=1, bg=Theme.BORDER)
            separator.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # 内容区
        self.content = tk.Frame(self, bg=Theme.BG_CARD)
        self.content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))


# ==================== 主应用 ====================
class FruitRecognitionGUI:
    """水果识别GUI应用（美化版）"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🍎 " + WINDOW_TITLE)
        self.root.geometry("950x700")
        self.root.resizable(True, True)
        self.root.configure(bg=Theme.BG_MAIN)
        
        # 变量
        self.current_image_path = None
        self.predictor = None
        self.photo_image = None
        
        # 设置字体
        self.setup_fonts()
        
        # 初始化预测器
        self.init_predictor()
        
        # 创建GUI组件
        self.create_widgets()
        
        # 居中窗口
        self.center_window()
    
    def setup_fonts(self):
        """设置字体"""
        self.font_title = ("微软雅黑", 24, "bold")
        self.font_subtitle = ("微软雅黑", 14, "bold")
        self.font_normal = ("微软雅黑", 11)
        self.font_result = ("Consolas", 11)
    
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
        # 主框架
        main_frame = tk.Frame(self.root, bg=Theme.BG_MAIN)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ===== 标题区域 =====
        self.create_header(main_frame)
        
        # ===== 内容区域 =====
        content_frame = tk.Frame(main_frame, bg=Theme.BG_MAIN)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # 左侧面板
        self.create_left_panel(content_frame)
        
        # 右侧面板
        self.create_right_panel(content_frame)
        
        # ===== 状态栏 =====
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """创建标题区域"""
        header_frame = tk.Frame(parent, bg=Theme.BG_MAIN)
        header_frame.pack(fill=tk.X)
        
        # 标题
        title_label = tk.Label(
            header_frame,
            text="🍎 水果智能识别系统",
            font=self.font_title,
            bg=Theme.BG_MAIN,
            fg=Theme.PRIMARY
        )
        title_label.pack(side=tk.LEFT)
        
        # 副标题
        subtitle_label = tk.Label(
            header_frame,
            text="基于深度学习的水果图像分类",
            font=("微软雅黑", 10),
            bg=Theme.BG_MAIN,
            fg=Theme.TEXT_SECONDARY
        )
        subtitle_label.pack(side=tk.LEFT, padx=(15, 0), pady=(10, 0))
    
    def create_left_panel(self, parent):
        """创建左侧面板 - 图片预览"""
        left_frame = tk.Frame(parent, bg=Theme.BG_MAIN)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 图片卡片
        image_card = Card(left_frame, title="📷 图片预览")
        image_card.pack(fill=tk.BOTH, expand=True)
        
        # 图片显示区域
        self.image_frame = tk.Frame(image_card.content, bg=Theme.BG_DARK, 
                                    width=380, height=380)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="📁\n\n点击下方按钮\n选择水果图片",
            font=("微软雅黑", 12),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # ===== 按钮区域 =====
        # 第一行：图片来源选择
        source_frame = tk.Frame(image_card.content, bg=Theme.BG_CARD)
        source_frame.pack(fill=tk.X, pady=(15, 8))
        
        tk.Label(source_frame, text="📥 选择图片来源：", 
                font=("微软雅黑", 10), bg=Theme.BG_CARD, 
                fg=Theme.TEXT_SECONDARY).pack(side=tk.LEFT, padx=(0, 10))
        
        # 选择文件按钮
        self.select_btn = ModernButton(
            source_frame,
            text="从文件选择",
            icon="📁",
            command=self.select_image,
            width=140,
            height=40,
            bg_color=Theme.INFO,
            hover_color="#3182CE"
        )
        self.select_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # 摄像头按钮
        self.camera_btn = ModernButton(
            source_frame,
            text="摄像头拍照",
            icon="📷",
            command=self.open_camera,
            width=140,
            height=40,
            bg_color=Theme.SECONDARY,
            hover_color=Theme.SECONDARY_DARK
        )
        self.camera_btn.pack(side=tk.LEFT)
        
        # 第二行：识别按钮（居中显示）
        action_frame = tk.Frame(image_card.content, bg=Theme.BG_CARD)
        action_frame.pack(fill=tk.X, pady=(8, 0))
        
        # 识别按钮 - 更大更醒目
        self.predict_btn = ModernButton(
            action_frame,
            text="🔍 开始识别",
            icon="",
            command=self.predict_image,
            width=300,
            height=50,
            bg_color=Theme.PRIMARY,
            hover_color=Theme.PRIMARY_DARK
        )
        self.predict_btn.pack(anchor=tk.CENTER)
        self.predict_btn.set_enabled(False)
    
    def create_right_panel(self, parent):
        """创建右侧面板 - 识别结果"""
        right_frame = tk.Frame(parent, bg=Theme.BG_MAIN, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)
        
        # 结果卡片
        result_card = Card(right_frame, title="📊 识别结果")
        result_card.pack(fill=tk.BOTH, expand=True)
        
        # 结果显示区域
        self.result_frame = tk.Frame(result_card.content, bg=Theme.BG_CARD)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 初始化欢迎信息
        self.show_welcome_message()
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = tk.Frame(parent, bg=Theme.BG_DARK, height=35)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        status_frame.pack_propagate(False)
        
        # 状态文字
        self.status_label = tk.Label(
            status_frame,
            text="✅ 就绪",
            font=("微软雅黑", 9),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=8)
        
        # 版权信息
        copyright_label = tk.Label(
            status_frame,
            text="© 2025 水果识别系统 | 基于 MobileNetV2",
            font=("微软雅黑", 9),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY
        )
        copyright_label.pack(side=tk.RIGHT, padx=15, pady=8)
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        # 清空结果区域
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # 欢迎标题
        welcome_title = tk.Label(
            self.result_frame,
            text="👋 欢迎使用！",
            font=("微软雅黑", 16, "bold"),
            bg=Theme.BG_CARD,
            fg=Theme.PRIMARY
        )
        welcome_title.pack(pady=(20, 15))
        
        # 使用说明
        instructions = [
            ("📁", "点击「选择图片」上传水果图片"),
            ("🔍", "点击「开始识别」进行识别"),
            ("📊", "查看识别结果和置信度"),
        ]
        
        for icon, text in instructions:
            item_frame = tk.Frame(self.result_frame, bg=Theme.BG_CARD)
            item_frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(item_frame, text=icon, font=("微软雅黑", 14),
                    bg=Theme.BG_CARD).pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(item_frame, text=text, font=self.font_normal,
                    bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(side=tk.LEFT)
        
        # 支持的水果列表
        tk.Label(self.result_frame, text="🍎 支持识别的水果", 
                font=("微软雅黑", 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(pady=(25, 10))
        
        fruits = ["🍎 苹果", "🍌 香蕉", "🍇 葡萄", "🥭 芒果", "🍊 橙子", "🍉 西瓜"]
        
        fruits_frame = tk.Frame(self.result_frame, bg=Theme.BG_CARD)
        fruits_frame.pack(pady=5)
        
        for i, fruit in enumerate(fruits):
            row = i // 3
            col = i % 3
            tk.Label(fruits_frame, text=fruit, font=self.font_normal,
                    bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                    width=10).grid(row=row, column=col, padx=5, pady=3)
    
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
            self.predict_btn.set_enabled(True)
            self.status_label.config(text=f"📷 已选择: {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        """显示图片"""
        try:
            img = resize_image_for_display(image_path, (360, 360))
            if img:
                self.photo_image = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.photo_image, text="",
                                          bg=Theme.BG_DARK)
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
        
        # 在新线程中进行预测
        thread = threading.Thread(target=self._predict_thread)
        thread.daemon = True
        thread.start()
    
    def _predict_thread(self):
        """预测线程"""
        self.root.after(0, self._start_prediction)
        
        try:
            result = self.predictor.predict(self.current_image_path)
            self.root.after(0, lambda: self._show_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _start_prediction(self):
        """开始预测时的UI更新"""
        self.predict_btn.set_enabled(False)
        self.select_btn.set_enabled(False)
        self.status_label.config(text="🔄 正在识别中...")
        
        # 清空结果区域并显示加载信息
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.result_frame, text="🔄", font=("微软雅黑", 48),
                bg=Theme.BG_CARD, fg=Theme.PRIMARY).pack(pady=(50, 20))
        tk.Label(self.result_frame, text="正在识别中，请稍候...", 
                font=self.font_normal, bg=Theme.BG_CARD, 
                fg=Theme.TEXT_SECONDARY).pack()
    
    def _show_result(self, result):
        """显示识别结果"""
        self.predict_btn.set_enabled(True)
        self.select_btn.set_enabled(True)
        
        if not result['success']:
            self.status_label.config(text="❌ 识别失败")
            self._show_error(result.get('error', '未知错误'))
            return
        
        # 清空结果区域
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # 主结果显示
        main_result_frame = tk.Frame(self.result_frame, bg=Theme.BG_CARD)
        main_result_frame.pack(fill=tk.X, pady=(10, 20))
        
        # 预测类别
        tk.Label(main_result_frame, text="🎯 识别结果", 
                font=("微软雅黑", 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor=tk.W)
        
        tk.Label(main_result_frame, 
                text=f"{result['predicted_class_cn']}",
                font=("微软雅黑", 28, "bold"),
                bg=Theme.BG_CARD, fg=Theme.PRIMARY).pack(anchor=tk.W, pady=(5, 0))
        
        tk.Label(main_result_frame, text=f"({result['predicted_class']})",
                font=("微软雅黑", 11),
                bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor=tk.W)
        
        # 置信度
        confidence_frame = tk.Frame(self.result_frame, bg=Theme.BG_CARD)
        confidence_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(confidence_frame, text="📊 置信度", 
                font=("微软雅黑", 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor=tk.W)
        
        # 进度条
        progress_frame = tk.Frame(confidence_frame, bg=Theme.BG_DARK, height=25)
        progress_frame.pack(fill=tk.X, pady=(8, 5))
        progress_frame.pack_propagate(False)
        
        confidence_pct = result['confidence_percent']
        progress_width = int(confidence_pct * 3.5)  # 调整宽度比例
        
        progress_bar = tk.Frame(progress_frame, bg=Theme.SUCCESS, 
                               width=progress_width, height=25)
        progress_bar.pack(side=tk.LEFT)
        
        tk.Label(confidence_frame, 
                text=f"{confidence_pct:.1f}%",
                font=("微软雅黑", 16, "bold"),
                bg=Theme.BG_CARD, fg=Theme.SUCCESS).pack(anchor=tk.W)
        
        # Top 3 预测结果
        top3_frame = tk.Frame(self.result_frame, bg=Theme.BG_CARD)
        top3_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(top3_frame, text="📋 Top 3 预测", 
                font=("微软雅黑", 12, "bold"),
                bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor=tk.W, pady=(0, 10))
        
        for i, pred in enumerate(result['top_3_predictions'], 1):
            item_frame = tk.Frame(top3_frame, bg=Theme.BG_CARD)
            item_frame.pack(fill=tk.X, pady=3)
            
            # 排名
            rank_colors = [Theme.PRIMARY, Theme.SECONDARY, Theme.INFO]
            rank_label = tk.Label(item_frame, text=str(i), 
                                 font=("微软雅黑", 10, "bold"),
                                 bg=rank_colors[i-1], fg=Theme.TEXT_WHITE,
                                 width=2, height=1)
            rank_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # 类别名
            tk.Label(item_frame, text=pred['class_cn'],
                    font=self.font_normal, bg=Theme.BG_CARD,
                    fg=Theme.TEXT_PRIMARY, width=8, anchor=tk.W).pack(side=tk.LEFT)
            
            # 概率
            tk.Label(item_frame, text=f"{pred['probability_percent']:.2f}%",
                    font=("Consolas", 11), bg=Theme.BG_CARD,
                    fg=Theme.TEXT_SECONDARY).pack(side=tk.RIGHT)
        
        # 更新状态栏
        self.status_label.config(
            text=f"✅ 识别完成: {result['predicted_class_cn']} ({confidence_pct:.1f}%)"
        )
    
    def _show_error(self, error_message):
        """显示错误信息"""
        self.predict_btn.set_enabled(True)
        self.select_btn.set_enabled(True)
        self.status_label.config(text="❌ 识别出错")
        messagebox.showerror("错误", f"识别过程出错: {error_message}")
    
    def open_camera(self):
        """打开摄像头拍照窗口"""
        camera_window = CameraWindow(self.root, self)
        camera_window.grab_set()  # 模态窗口
    
    def set_camera_image(self, image_path):
        """设置从摄像头拍摄的图片"""
        self.current_image_path = image_path
        self.display_image(image_path)
        self.predict_btn.set_enabled(True)
        self.status_label.config(text=f"📷 已拍摄: {os.path.basename(image_path)}")


# ==================== 摄像头窗口 ====================
class CameraWindow(tk.Toplevel):
    """摄像头拍照窗口"""
    
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.title("📷 摄像头拍照")
        self.geometry("700x650")
        self.configure(bg=Theme.BG_MAIN)
        self.resizable(False, False)
        
        # 摄像头变量
        self.cap = None
        self.is_running = False
        self.current_frame = None
        
        # 创建界面
        self.create_widgets()
        
        # 启动摄像头
        self.start_camera()
        
        # 窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 居中显示
        self.center_window()
    
    def create_widgets(self):
        """创建界面组件"""
        # 标题
        title_label = tk.Label(
            self,
            text="📷 摄像头拍照识别",
            font=("微软雅黑", 18, "bold"),
            bg=Theme.BG_MAIN,
            fg=Theme.PRIMARY
        )
        title_label.pack(pady=(15, 10))
        
        # 提示文字
        tip_label = tk.Label(
            self,
            text="将水果放在摄像头前，点击「拍照」按钮进行识别",
            font=("微软雅黑", 10),
            bg=Theme.BG_MAIN,
            fg=Theme.TEXT_SECONDARY
        )
        tip_label.pack(pady=(0, 15))
        
        # 视频显示区域
        video_frame = tk.Frame(self, bg=Theme.BG_DARK, width=640, height=480)
        video_frame.pack(padx=20)
        video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(
            video_frame,
            text="正在启动摄像头...",
            font=("微软雅黑", 14),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY
        )
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # 按钮区域
        button_frame = tk.Frame(self, bg=Theme.BG_MAIN)
        button_frame.pack(pady=15)
        
        # 拍照按钮
        self.capture_btn = ModernButton(
            button_frame,
            text="拍照",
            icon="📸",
            command=self.capture_photo,
            width=150,
            bg_color=Theme.PRIMARY,
            hover_color=Theme.PRIMARY_DARK
        )
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # 取消按钮
        cancel_btn = ModernButton(
            button_frame,
            text="取消",
            icon="❌",
            command=self.on_close,
            width=150,
            bg_color=Theme.ERROR,
            hover_color="#E53E3E"
        )
        cancel_btn.pack(side=tk.LEFT)
    
    def center_window(self):
        """居中显示窗口"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头！请检查摄像头是否已连接。")
                self.destroy()
                return
            
            self.is_running = True
            self.update_frame()
        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {str(e)}")
            self.destroy()
    
    def update_frame(self):
        """更新视频帧"""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                
                # 转换为PIL图像
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # 调整大小
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                
                # 转换为Tkinter图像
                photo = ImageTk.PhotoImage(img)
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
            
            # 每30ms更新一次
            self.after(30, self.update_frame)
    
    def capture_photo(self):
        """拍照"""
        if self.current_frame is None:
            messagebox.showwarning("警告", "请等待摄像头启动完成！")
            return
        
        try:
            # 创建captures目录
            captures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
            os.makedirs(captures_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(captures_dir, filename)
            
            # 保存图片
            cv2.imwrite(filepath, self.current_frame)
            
            # 关闭摄像头窗口
            self.stop_camera()
            self.destroy()
            
            # 设置图片到主界面
            self.main_app.set_camera_image(filepath)
            
            # 自动开始识别
            self.main_app.predict_image()
            
        except Exception as e:
            messagebox.showerror("错误", f"保存图片失败: {str(e)}")
    
    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def on_close(self):
        """关闭窗口"""
        self.stop_camera()
        self.destroy()


def run_gui():
    """运行GUI应用"""
    root = tk.Tk()
    app = FruitRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

