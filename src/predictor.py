"""
预测模块 - 使用训练好的模型进行水果识别
"""
import os
import numpy as np
from tensorflow.keras.models import load_model
from utils.config import MODEL_PATH, FRUIT_CLASSES, CLASS_NAMES_CN, IMG_SIZE
from utils.image_utils import preprocess_for_prediction


class FruitPredictor:
    """水果识别预测器"""
    
    def __init__(self, model_path=MODEL_PATH):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.class_names = FRUIT_CLASSES
        self.class_names_cn = CLASS_NAMES_CN
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"正在加载模型: {self.model_path}")
        self.model = load_model(self.model_path)
        print("✅ 模型加载成功")
    
    def predict(self, image_path):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
        
        Returns:
            字典，包含预测结果信息
        """
        # 预处理图片
        img_array = preprocess_for_prediction(image_path, IMG_SIZE)
        
        if img_array is None:
            return {
                'success': False,
                'error': '图片加载失败'
            }
        
        # 进行预测
        predictions = self.model.predict(img_array, verbose=0)
        
        # 获取预测结果
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        predicted_class_cn = self.class_names_cn[predicted_class]
        
        # 获取所有类别的概率
        all_probabilities = {}
        for i, class_name in enumerate(self.class_names):
            all_probabilities[class_name] = float(predictions[0][i])
        
        # 返回结果
        result = {
            'success': True,
            'predicted_class': predicted_class,
            'predicted_class_cn': predicted_class_cn,
            'confidence': float(confidence),
            'confidence_percent': float(confidence * 100),
            'all_probabilities': all_probabilities,
            'top_3_predictions': self._get_top_k_predictions(predictions[0], k=3)
        }
        
        return result
    
    def _get_top_k_predictions(self, predictions, k=3):
        """
        获取Top-K预测结果
        
        Args:
            predictions: 预测概率数组
            k: 返回前k个结果
        
        Returns:
            Top-K预测结果列表
        """
        # 获取概率最高的k个索引
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        top_k_results = []
        for idx in top_k_indices:
            top_k_results.append({
                'class': self.class_names[idx],
                'class_cn': self.class_names_cn[self.class_names[idx]],
                'probability': float(predictions[idx]),
                'probability_percent': float(predictions[idx] * 100)
            })
        
        return top_k_results
    
    def predict_batch(self, image_paths):
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片路径列表
        
        Returns:
            预测结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def print_result(self, result):
        """
        打印预测结果
        
        Args:
            result: 预测结果字典
        """
        if not result['success']:
            print(f"❌ 预测失败: {result.get('error', '未知错误')}")
            return
        
        print("\n" + "="*60)
        print("🍎 水果识别结果")
        print("="*60)
        print(f"预测类别: {result['predicted_class_cn']} ({result['predicted_class']})")
        print(f"置信度: {result['confidence_percent']:.2f}%")
        print("\n前3名预测结果:")
        print("-"*60)
        for i, pred in enumerate(result['top_3_predictions'], 1):
            print(f"{i}. {pred['class_cn']:10s} ({pred['class']:12s}) - {pred['probability_percent']:5.2f}%")
        print("="*60)


def predict_single_image(image_path, model_path=MODEL_PATH):
    """
    便捷函数：预测单张图片
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
    
    Returns:
        预测结果字典
    """
    predictor = FruitPredictor(model_path)
    result = predictor.predict(image_path)
    predictor.print_result(result)
    return result


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python predictor.py <图片路径>")
        print("示例: python predictor.py test_image.jpg")
        exit(1)
    
    image_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在: {image_path}")
        exit(1)
    
    # 进行预测
    predict_single_image(image_path)
