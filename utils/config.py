"""
配置文件 - 存储项目的全局配置参数
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录配置
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')

# 模型目录配置
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_model.h5')

# 水果类别配置
FRUIT_CLASSES = [
    'Apple',      # 苹果
    'Banana',     # 香蕉
    'Grape',      # 葡萄
    'Mango',      # 芒果
    'Orange',     # 橙子
    'Watermelon'  # 西瓜
]

# 中文类别映射
CLASS_NAMES_CN = {
    'Apple': '苹果',
    'Banana': '香蕉',
    'Grape': '葡萄',
    'Mango': '芒果',
    'Orange': '橙子',
    'Watermelon': '西瓜'
}

# 图像配置
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# 训练配置
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

# 数据增强配置
DATA_AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# MobileNetV2配置
BASE_MODEL_NAME = 'MobileNetV2'
WEIGHTS = 'imagenet'
INCLUDE_TOP = False
POOLING = 'avg'

# 自定义层配置
DENSE_UNITS = 128
DROPOUT_RATE = 0.5

# 优化器配置
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

# 早停配置
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# GUI配置
WINDOW_TITLE = '水果识别系统 - Fruit Recognition System'
WINDOW_SIZE = '900x700'
PREVIEW_SIZE = (400, 400)

# 支持的图片格式
SUPPORTED_FORMATS = [
    ('Image Files', '*.jpg *.jpeg *.png *.bmp *.gif'),
    ('JPEG Files', '*.jpg *.jpeg'),
    ('PNG Files', '*.png'),
    ('All Files', '*.*')
]
