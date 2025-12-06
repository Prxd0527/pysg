"""
图像处理工具模块 - 提供图像加载、预处理、增强等功能
"""
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from utils.config import IMG_SIZE, DATA_AUGMENTATION


def load_and_preprocess_image(image_path, target_size=IMG_SIZE):
    """
    加载并预处理单张图片
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸 (height, width)
    
    Returns:
        预处理后的图片数组
    """
    try:
        # 使用PIL加载图片
        img = load_img(image_path, target_size=target_size)
        # 转换为数组
        img_array = img_to_array(img)
        # 归一化到[0, 1]
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"加载图片失败: {image_path}, 错误: {e}")
        return None


def preprocess_for_prediction(image_path, target_size=IMG_SIZE):
    """
    为预测准备图片（添加batch维度）
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸
    
    Returns:
        预处理后的图片数组 (1, height, width, channels)
    """
    img_array = load_and_preprocess_image(image_path, target_size)
    if img_array is not None:
        # 添加batch维度
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    return None


def create_data_generator(augmentation=True):
    """
    创建数据生成器
    
    Args:
        augmentation: 是否使用数据增强
    
    Returns:
        ImageDataGenerator对象
    """
    if augmentation:
        # 带数据增强的生成器
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=DATA_AUGMENTATION['rotation_range'],
            width_shift_range=DATA_AUGMENTATION['width_shift_range'],
            height_shift_range=DATA_AUGMENTATION['height_shift_range'],
            shear_range=DATA_AUGMENTATION['shear_range'],
            zoom_range=DATA_AUGMENTATION['zoom_range'],
            horizontal_flip=DATA_AUGMENTATION['horizontal_flip'],
            fill_mode=DATA_AUGMENTATION['fill_mode']
        )
    else:
        # 仅归一化的生成器
        datagen = ImageDataGenerator(rescale=1./255)
    
    return datagen


def resize_image_for_display(image_path, target_size):
    """
    调整图片大小用于GUI显示
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸 (width, height)
    
    Returns:
        调整后的PIL Image对象
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"调整图片大小失败: {e}")
        return None


def load_image_cv2(image_path):
    """
    使用OpenCV加载图片
    
    Args:
        image_path: 图片路径
    
    Returns:
        BGR格式的图片数组
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"OpenCV加载图片失败: {e}")
        return None


def convert_bgr_to_rgb(image):
    """
    将BGR格式转换为RGB格式
    
    Args:
        image: BGR格式的图片数组
    
    Returns:
        RGB格式的图片数组
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image_array, save_path):
    """
    保存图片到文件
    
    Args:
        image_array: 图片数组
        save_path: 保存路径
    """
    try:
        # 如果是归一化的数组，反归一化
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        # 转换为PIL Image并保存
        img = Image.fromarray(image_array)
        img.save(save_path)
        print(f"图片已保存到: {save_path}")
    except Exception as e:
        print(f"保存图片失败: {e}")


def validate_image_format(image_path):
    """
    验证图片格式是否有效
    
    Args:
        image_path: 图片路径
    
    Returns:
        布尔值，表示是否有效
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    import os
    _, ext = os.path.splitext(image_path.lower())
    return ext in valid_extensions
