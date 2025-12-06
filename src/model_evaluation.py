"""
模型评估模块 - 评估已训练模型的性能
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import *
from utils.image_utils import create_data_generator


def load_trained_model(model_path=MODEL_PATH):
    """
    加载已训练的模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        加载的Keras模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    # 使用compile=False解决Keras版本兼容性问题
    model = load_model(model_path, compile=False)
    print("✓ 模型加载成功")
    return model


def prepare_test_generator():
    """
    准备测试数据生成器
    
    Returns:
        测试数据生成器
    """
    test_datagen = create_data_generator(augmentation=False)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=FRUIT_CLASSES
    )
    
    print(f"测试样本数: {test_generator.samples}")
    return test_generator


def evaluate_model(model, test_generator):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        test_generator: 测试数据生成器
    
    Returns:
        评估结果（loss, accuracy）
    """
    print("\n" + "="*60)
    print("评估模型...")
    print("="*60)
    
    results = model.evaluate(test_generator, verbose=1)
    
    print(f"\n测试损失: {results[0]:.4f}")
    print(f"测试准确率: {results[1]:.4f} ({results[1]*100:.2f}%)")
    
    return results


def generate_predictions(model, test_generator):
    """
    生成预测结果
    
    Args:
        model: 训练好的模型
        test_generator: 测试数据生成器
    
    Returns:
        y_true, y_pred (真实标签和预测标签)
    """
    print("\n生成预测结果...")
    
    # 重置生成器
    test_generator.reset()
    
    # 进行预测
    predictions = model.predict(test_generator, verbose=1)
    
    # 获取预测类别
    y_pred = np.argmax(predictions, axis=1)
    
    # 获取真实类别
    y_true = test_generator.classes
    
    return y_true, y_pred, predictions


def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=FRUIT_CLASSES,
        yticklabels=FRUIT_CLASSES,
        cbar_kws={'label': '样本数量'}
    )
    plt.title('混淆矩阵', fontsize=16, pad=20)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred):
    """
    打印分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    """
    print("\n" + "="*60)
    print("分类报告")
    print("="*60)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=FRUIT_CLASSES,
        digits=4
    )
    print(report)
    
    # 保存报告到文件
    report_path = 'models/classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("分类报告\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"✓ 分类报告已保存: {report_path}")


def plot_class_accuracy(y_true, y_pred, save_path='models/class_accuracy.png'):
    """
    绘制每个类别的准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    accuracies = []
    
    for i, fruit_class in enumerate(FRUIT_CLASSES):
        # 找到该类别的所有样本
        class_mask = (y_true == i)
        class_true = y_true[class_mask]
        class_pred = y_pred[class_mask]
        
        # 计算准确率
        if len(class_true) > 0:
            accuracy = accuracy_score(class_true, class_pred)
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(FRUIT_CLASSES)), accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 在柱子上添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('水果类别', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('各类别识别准确率', fontsize=14, pad=20)
    plt.xticks(range(len(FRUIT_CLASSES)), FRUIT_CLASSES, rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 类别准确率图已保存: {save_path}")
    plt.close()


def plot_top_predictions(predictions, test_generator, num_samples=9, save_path='models/sample_predictions.png'):
    """
    可视化部分预测结果
    
    Args:
        predictions: 预测概率数组
        test_generator: 测试数据生成器
        num_samples: 要显示的样本数量
        save_path: 保存路径
    """
    # 重置生成器并获取一批样本
    test_generator.reset()
    images, labels = next(test_generator)
    
    # 只取前num_samples个样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    pred_probs = predictions[:num_samples]
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # 显示图片
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        # 获取真实标签和预测标签
        true_label = FRUIT_CLASSES[np.argmax(labels[i])]
        pred_label = FRUIT_CLASSES[np.argmax(pred_probs[i])]
        confidence = np.max(pred_probs[i]) * 100
        
        # 设置标题颜色（正确为绿色，错误为红色）
        color = 'green' if true_label == pred_label else 'red'
        
        # 设置标题
        axes[i].set_title(
            f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.1f}%',
            fontsize=9,
            color=color
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 样本预测图已保存: {save_path}")
    plt.close()


def run_full_evaluation():
    """
    运行完整的模型评估流程
    """
    print("\n" + "🔍 "*20)
    print("水果识别模型评估")
    print("🔍 "*20)
    
    # 加载模型
    model = load_trained_model()
    
    # 准备测试数据
    test_gen = prepare_test_generator()
    
    # 评估模型
    evaluate_model(model, test_gen)
    
    # 生成预测
    y_true, y_pred, predictions = generate_predictions(model, test_gen)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred)
    
    # 打印分类报告
    print_classification_report(y_true, y_pred)
    
    # 绘制类别准确率
    plot_class_accuracy(y_true, y_pred)
    
    # 绘制样本预测
    plot_top_predictions(predictions, test_gen)
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)
    print("生成的文件:")
    print("  - models/confusion_matrix.png (混淆矩阵)")
    print("  - models/classification_report.txt (分类报告)")
    print("  - models/class_accuracy.png (类别准确率)")
    print("  - models/sample_predictions.png (样本预测)")
    print("="*60)


if __name__ == "__main__":
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请先运行 model_training.py 训练模型。")
        exit(1)
    
    # 检查测试数据是否存在
    if not os.path.exists(TEST_DIR) or not os.listdir(TEST_DIR):
        print(f"❌ 测试数据不存在: {TEST_DIR}")
        print("请先准备测试数据。")
        exit(1)
    
    # 运行评估
    run_full_evaluation()
