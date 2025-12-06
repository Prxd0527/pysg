"""
数据准备模块 - 负责数据集的下载、组织和预处理
"""
import os
import shutil
from pathlib import Path
from utils.config import (
    TRAIN_DIR, TEST_DIR, VALIDATION_DIR, 
    FRUIT_CLASSES, DATA_DIR
)


def create_data_directories():
    """
    创建数据目录结构
    """
    directories = [TRAIN_DIR, TEST_DIR, VALIDATION_DIR]
    
    for directory in directories:
        for fruit_class in FRUIT_CLASSES:
            class_dir = os.path.join(directory, fruit_class)
            os.makedirs(class_dir, exist_ok=True)
            print(f"创建目录: {class_dir}")
    
    print("数据目录结构创建完成！")


def print_dataset_guide():
    """
    打印数据集获取指南
    """
    guide = """
    ═══════════════════════════════════════════════════════════
    📦 数据集获取指南
    ═══════════════════════════════════════════════════════════
    
    推荐数据集: Kaggle Fruits 360
    
    方法1: 使用Kaggle API (推荐)
    ─────────────────────────────────────────────────────
    1. 注册Kaggle账号: https://www.kaggle.com
    2. 获取API Token:
       - 进入Kaggle账户设置
       - 点击 "Create New API Token"
       - 下载 kaggle.json 文件
    3. 配置API:
       - Windows: 将kaggle.json放到 C:\\Users\\<用户名>\\.kaggle\\
       - Linux/Mac: 将kaggle.json放到 ~/.kaggle/
    4. 安装Kaggle CLI:
       pip install kaggle
    5. 下载数据集:
       kaggle datasets download -d moltean/fruits
    6. 解压数据集到data目录
    
    方法2: 手动下载
    ─────────────────────────────────────────────────────
    1. 访问: https://www.kaggle.com/moltean/fruits
    2. 点击 "Download" 按钮
    3. 解压下载的zip文件
    4. 将数据整理到以下结构:
    
    data/
    ├── train/
    │   ├── Apple/
    │   ├── Banana/
    │   ├── Grape/
    │   ├── Mango/
    │   ├── Orange/
    │   └── Watermelon/
    ├── test/
    │   └── (同上结构)
    └── validation/
        └── (同上结构)
    
    方法3: 使用示例数据集
    ─────────────────────────────────────────────────────
    如果只是测试，可以准备少量图片:
    - 每个类别准备10-20张训练图片
    - 每个类别准备5-10张测试图片
    - 手动放入对应的目录中
    
    注意事项:
    ─────────────────────────────────────────────────────
    ✓ 图片格式: JPG, PNG, BMP等常见格式
    ✓ 建议尺寸: 至少100x100像素
    ✓ 数量建议: 每类至少50张训练图片以获得较好效果
    ✓ 数据分布: 训练集:验证集:测试集 = 70:15:15
    
    ═══════════════════════════════════════════════════════════
    """
    print(guide)


def count_images_in_directory(directory):
    """
    统计目录中的图片数量
    
    Args:
        directory: 目录路径
    
    Returns:
        字典，每个类别的图片数量
    """
    class_counts = {}
    
    if not os.path.exists(directory):
        return class_counts
    
    for fruit_class in FRUIT_CLASSES:
        class_dir = os.path.join(directory, fruit_class)
        if os.path.exists(class_dir):
            # 统计图片文件
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            class_counts[fruit_class] = len(image_files)
        else:
            class_counts[fruit_class] = 0
    
    return class_counts


def print_dataset_summary():
    """
    打印数据集摘要信息
    """
    print("\n" + "="*60)
    print("📊 数据集摘要")
    print("="*60)
    
    for dataset_name, dataset_dir in [
        ("训练集", TRAIN_DIR),
        ("验证集", VALIDATION_DIR),
        ("测试集", TEST_DIR)
    ]:
        print(f"\n{dataset_name}: {dataset_dir}")
        counts = count_images_in_directory(dataset_dir)
        total = sum(counts.values())
        
        if total > 0:
            for fruit_class, count in counts.items():
                print(f"  - {fruit_class}: {count} 张图片")
            print(f"  总计: {total} 张图片")
        else:
            print(f"  ⚠️  目录为空或不存在")
    
    print("\n" + "="*60)


def validate_dataset():
    """
    验证数据集是否准备完毕
    
    Returns:
        布尔值，表示数据集是否有效
    """
    train_counts = count_images_in_directory(TRAIN_DIR)
    test_counts = count_images_in_directory(TEST_DIR)
    
    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    
    if train_total == 0:
        print("❌ 训练集为空！请先准备数据集。")
        return False
    
    if test_total == 0:
        print("⚠️  测试集为空！建议准备测试数据。")
    
    # 检查每个类别是否都有数据
    missing_classes = [cls for cls, count in train_counts.items() if count == 0]
    if missing_classes:
        print(f"⚠️  以下类别在训练集中没有数据: {', '.join(missing_classes)}")
    
    print("✅ 数据集验证通过！")
    return True


def organize_downloaded_data(source_dir, selected_fruits=None):
    """
    整理下载的Fruits 360数据集
    
    Args:
        source_dir: 下载并解压后的源目录
        selected_fruits: 要选择的水果类别列表（如果为None，使用默认配置）
    """
    if selected_fruits is None:
        selected_fruits = FRUIT_CLASSES
    
    print(f"开始整理数据集...")
    print(f"源目录: {source_dir}")
    print(f"目标类别: {', '.join(selected_fruits)}")
    
    # Fruits 360数据集通常有Training和Test两个文件夹
    source_train = os.path.join(source_dir, 'Training')
    source_test = os.path.join(source_dir, 'Test')
    
    if not os.path.exists(source_train) and not os.path.exists(source_test):
        print("❌ 未找到Training或Test目录，请检查源目录路径！")
        return
    
    # 复制训练数据
    if os.path.exists(source_train):
        for fruit in selected_fruits:
            source_class_dir = os.path.join(source_train, fruit)
            target_class_dir = os.path.join(TRAIN_DIR, fruit)
            
            if os.path.exists(source_class_dir):
                os.makedirs(target_class_dir, exist_ok=True)
                # 复制图片
                for img_file in os.listdir(source_class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        shutil.copy2(
                            os.path.join(source_class_dir, img_file),
                            os.path.join(target_class_dir, img_file)
                        )
                print(f"✓ 已复制 {fruit} 训练数据")
            else:
                print(f"⚠️  未找到 {fruit} 的训练数据")
    
    # 复制测试数据
    if os.path.exists(source_test):
        for fruit in selected_fruits:
            source_class_dir = os.path.join(source_test, fruit)
            target_class_dir = os.path.join(TEST_DIR, fruit)
            
            if os.path.exists(source_class_dir):
                os.makedirs(target_class_dir, exist_ok=True)
                for img_file in os.listdir(source_class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        shutil.copy2(
                            os.path.join(source_class_dir, img_file),
                            os.path.join(target_class_dir, img_file)
                        )
                print(f"✓ 已复制 {fruit} 测试数据")
            else:
                print(f"⚠️  未找到 {fruit} 的测试数据")
    
    print("\n数据整理完成！")
    print_dataset_summary()


if __name__ == "__main__":
    # 创建数据目录
    create_data_directories()
    
    # 打印数据集获取指南
    print_dataset_guide()
    
    # 打印数据集摘要
    print_dataset_summary()
    
    # 如果你已经下载了Fruits 360数据集，可以取消下面的注释来整理数据
    # source_directory = "path/to/your/downloaded/fruits-360"
    # organize_downloaded_data(source_directory)
