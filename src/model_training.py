"""
模型训练模块 - 使用迁移学习训练水果分类模型
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import *
from utils.image_utils import create_data_generator


def build_model(num_classes=len(FRUIT_CLASSES)):
    """
    构建基于MobileNetV2的迁移学习模型
    
    Args:
        num_classes: 分类类别数量
    
    Returns:
        编译好的Keras模型
    """
    print("正在构建模型...")
    
    # 加载预训练的MobileNetV2模型（不包含顶层）
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        include_top=INCLUDE_TOP,
        weights=WEIGHTS
    )
    
    # 冻结基础模型的所有层
    base_model.trainable = False
    
    print(f"基础模型: {BASE_MODEL_NAME}")
    print(f"总层数: {len(base_model.layers)}")
    print(f"可训练层数: {sum([layer.trainable for layer in base_model.layers])}")
    
    # 构建完整模型
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(DENSE_UNITS, activation='relu', name='dense_1'),
        Dropout(DROPOUT_RATE, name='dropout'),
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=METRICS
    )
    
    print("\n模型架构:")
    model.summary()
    
    return model


def prepare_data_generators():
    """
    准备训练和验证数据生成器
    
    Returns:
        train_generator, validation_generator
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
    print(f"✓ 类别映射: {train_generator.class_indices}")
    
    return train_generator, validation_generator


def setup_callbacks():
    """
    设置训练回调函数
    
    Returns:
        回调函数列表
    """
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


def train_model(model, train_generator, validation_generator, epochs=EPOCHS):
    """
    训练模型
    
    Args:
        model: 待训练的模型
        train_generator: 训练数据生成器
        validation_generator: 验证数据生成器
        epochs: 训练轮数
    
    Returns:
        训练历史对象
    """
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)
    
    # 设置回调
    callbacks = setup_callbacks()
    
    # 训练模型
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n训练完成！")
    print(f"模型已保存到: {MODEL_PATH}")
    
    return history


def plot_training_history(history, save_path='models/training_history.png'):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史对象
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存到: {save_path}")
    plt.close()


def fine_tune_model(model, train_generator, validation_generator, unfreeze_layers=20):
    """
    微调模型（可选的高级训练步骤）
    
    Args:
        model: 已训练的模型
        train_generator: 训练数据生成器
        validation_generator: 验证数据生成器
        unfreeze_layers: 解冻的层数
    
    Returns:
        微调后的训练历史
    """
    print("\n" + "="*60)
    print("开始微调模型...")
    print("="*60)
    
    # 解冻基础模型的部分层
    base_model = model.layers[0]
    base_model.trainable = True
    
    # 冻结前面的层，只训练后面的层
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    print(f"解冻层数: {unfreeze_layers}")
    print(f"可训练层数: {sum([layer.trainable for layer in base_model.layers])}")
    
    # 重新编译模型（使用更小的学习率）
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss=LOSS,
        metrics=METRICS
    )
    
    # 继续训练
    callbacks = setup_callbacks()
    history_fine = model.fit(
        train_generator,
        epochs=10,  # 微调较少的轮数
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n微调完成！")
    return history_fine


if __name__ == "__main__":
    # 检查数据目录
    if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
        print("❌ 训练数据不存在！请先运行 data_preparation.py 准备数据。")
        exit(1)
    
    # 准备数据生成器
    train_gen, val_gen = prepare_data_generators()
    
    # 构建模型
    model = build_model()
    
    # 训练模型
    history = train_model(model, train_gen, val_gen)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 可选：微调模型以获得更好的性能
    # history_fine = fine_tune_model(model, train_gen, val_gen)
    # plot_training_history(history_fine, 'models/fine_tuning_history.png')
    
    print("\n🎉 模型训练全部完成！")
    print(f"📁 模型文件: {MODEL_PATH}")
    print(f"📊 训练历史图: models/training_history.png")
