"""
继续训练模块 - 在现有模型基础上继续训练
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils.config import *
from utils.image_utils import create_data_generator


def load_existing_model(model_path=MODEL_PATH):
    """
    加载现有模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"正在加载现有模型: {model_path}")
    # 使用compile=False加载，然后手动重新编译
    model = load_model(model_path, compile=False)
    
    # 重新编译模型
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # 使用较小的学习率
        loss=LOSS,
        metrics=METRICS
    )
    
    print("✅ 模型加载成功")
    return model


def prepare_data_generators():
    """
    准备训练和验证数据生成器
    """
    print("\n准备数据生成器...")
    
    # 训练数据生成器（带数据增强）
    train_datagen = create_data_generator(augmentation=True)
    
    # 验证数据生成器（仅归一化）
    val_datagen = create_data_generator(augmentation=False)
    
    # 从目录加载训练数据
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=FRUIT_CLASSES
    )
    
    # 从目录加载验证数据
    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR if os.path.exists(VALIDATION_DIR) and os.listdir(VALIDATION_DIR) else TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=FRUIT_CLASSES
    )
    
    print(f"✓ 训练样本数: {train_generator.samples}")
    print(f"✓ 验证样本数: {validation_generator.samples}")
    
    return train_generator, validation_generator


def setup_callbacks():
    """设置训练回调函数"""
    callbacks = []
    
    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 学习率衰减回调
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # 模型检查点回调
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    return callbacks


def continue_training(model, train_generator, validation_generator, epochs=10):
    """
    继续训练模型
    
    Args:
        model: 现有模型
        train_generator: 训练数据生成器
        validation_generator: 验证数据生成器
        epochs: 额外训练轮数
    
    Returns:
        训练历史对象
    """
    print("\n" + "="*60)
    print("🔄 开始继续训练...")
    print(f"   额外训练轮数: {epochs}")
    print("="*60)
    
    # 设置回调
    callbacks = setup_callbacks()
    
    # 继续训练
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ 继续训练完成！")
    print(f"模型已保存到: {MODEL_PATH}")
    
    return history


def plot_training_history(history, save_path='models/continue_training_history.png'):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (Continue Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Continue Training)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    # 设置额外训练轮数（可以根据需要调整）
    EXTRA_EPOCHS = 10
    
    print("="*60)
    print("🔄 增量训练模式")
    print("="*60)
    print("此脚本将在现有模型基础上继续训练。")
    print("请确保已将新的训练数据添加到 data/train/ 目录。")
    print("="*60)
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请先运行 model_training.py 训练初始模型。")
        exit(1)
    
    # 检查训练数据
    if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
        print("❌ 训练数据不存在！")
        exit(1)
    
    # 加载现有模型
    model = load_existing_model()
    
    # 准备数据生成器
    train_gen, val_gen = prepare_data_generators()
    
    # 继续训练
    history = continue_training(model, train_gen, val_gen, epochs=EXTRA_EPOCHS)
    
    # 绘制训练历史
    plot_training_history(history)
    
    print("\n🎉 增量训练完成！")
    print(f"📁 更新后的模型: {MODEL_PATH}")
    print(f"📊 训练历史图: models/continue_training_history.png")
