# 水果识别系统 - 快速开始指南

## 📋 系统要求
- Python 3.8 或更高版本
- 至少 2GB 可用内存
- 支持的操作系统: Windows, Linux, macOS

## 🚀 快速开始

### 步骤 1: 安装依赖
```powershell
pip install -r requirements.txt
```

### 步骤 2: 准备数据集

#### 方法一：使用公开数据集（推荐）
1. 从 Kaggle 下载 Fruits 360 数据集: https://www.kaggle.com/moltean/fruits
2. 解压数据集
3. 运行数据准备脚本:
```powershell
python src/data_preparation.py
```

#### 方法二：使用自己的数据
将图片按以下结构组织:
```
data/
├── train/
│   ├── Apple/
│   ├── Banana/
│   ├── Grape/
│   ├── Mango/
│   ├── Orange/
│   └── Watermelon/
└── test/
    └── (同上结构)
```

### 步骤 3: 训练模型
```powershell
python src/model_training.py
```
训练时间约 10-30 分钟（取决于数据量和硬件配置）

### 步骤 4: 评估模型（可选）
```powershell
python src/model_evaluation.py
```

### 步骤 5: 运行GUI应用
```powershell
python main.py
```

## 💡 使用说明

### GUI界面使用
1. 点击「选择图片」按钮上传水果图片
2. 点击「开始识别」按钮进行识别
3. 查看识别结果和置信度

### 命令行使用
```powershell
python src/predictor.py <图片路径>
```

示例:
```powershell
python src/predictor.py test_images/apple.jpg
```

## 🍎 支持的水果类别
- 苹果 (Apple)
- 香蕉 (Banana)
- 葡萄 (Grape)
- 芒果 (Mango)
- 橙子 (Orange)
- 西瓜 (Watermelon)

## 📁 项目结构
```
pysg/
├── data/              # 数据集目录
├── models/            # 训练好的模型
├── src/               # 源代码
│   ├── data_preparation.py    # 数据准备
│   ├── model_training.py      # 模型训练
│   ├── model_evaluation.py    # 模型评估
│   ├── predictor.py           # 预测模块
│   └── gui_app.py             # GUI应用
├── utils/             # 工具模块
│   ├── config.py              # 配置文件
│   └── image_utils.py         # 图像处理工具
├── requirements.txt   # 依赖包
├── README.md          # 项目文档
└── main.py           # 程序入口
```

## ⚠️ 常见问题

### 问题1: 缺少依赖包
**解决方案**: 确保已安装所有依赖
```powershell
pip install -r requirements.txt
```

### 问题2: 模型文件不存在
**解决方案**: 先训练模型
```powershell
python src/model_training.py
```

### 问题3: 训练数据为空
**解决方案**: 确保数据目录结构正确，每个类别至少有一些图片

### 问题4: 识别准确率低
**可能原因**:
- 训练数据不足
- 图片质量差
- 类别不在支持范围内

**解决方案**:
- 增加训练数据
- 使用清晰的正面拍摄图片
- 检查水果类别是否在支持列表中

## 🎯 性能优化建议
1. 使用GPU加速训练（如果有NVIDIA显卡）
2. 增加训练数据量以提高准确率
3. 调整 `utils/config.py` 中的超参数
4. 使用数据增强扩充训练集

## 📞 技术支持
如遇到问题，请检查:
1. Python 版本是否正确
2. 依赖包是否完整安装
3. 数据集是否准备完毕
4. 模型文件是否存在

## 📄 许可证
本项目仅供学习和研究使用。
