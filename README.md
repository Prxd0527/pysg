# 水果识别系统 (Fruit Recognition System)

## 项目概括
本项目旨在开发一个基于深度学习的水果图像识别系统，使用Python语言实现。系统采用迁移学习技术，基于预训练的MobileNetV2模型进行水果分类，并提供友好的图形用户界面（GUI），让用户能够方便地上传图片并获得识别结果。

## 技术选型
- **主要编程语言**: Python 3.8+
- **深度学习框架**: TensorFlow 2.x + Keras
- **预训练模型**: MobileNetV2（轻量级，适合实时识别）
- **GUI框架**: Tkinter（Python内置）
- **图像处理**: OpenCV, Pillow
- **数据处理**: NumPy, Pandas
- **数据集**: Kaggle Fruits 360 数据集
- **可视化**: Matplotlib
- **版本控制**: Git

## 项目结构 / 模块划分
```
pysg/
├── data/                      # 数据目录
│   ├── train/                 # 训练数据
│   ├── test/                  # 测试数据
│   └── validation/            # 验证数据
├── models/                    # 模型保存目录
│   └── fruit_model.h5         # 训练好的模型
├── src/                       # 源代码目录
│   ├── data_preparation.py    # 数据准备模块
│   ├── model_training.py      # 模型训练模块
│   ├── model_evaluation.py    # 模型评估模块
│   ├── predictor.py           # 预测模块
│   └── gui_app.py             # GUI应用主程序
├── utils/                     # 工具模块
│   ├── config.py              # 配置文件
│   └── image_utils.py         # 图像处理工具
├── tests/                     # 测试代码
├── copy/                      # 备份目录
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目文档
└── main.py                    # 程序入口
```

## 核心功能 / 模块详解

### 1. 数据准备模块 (data_preparation.py)
- 从Kaggle下载并组织数据集
- 数据预处理（图像缩放、归一化）
- 数据增强（旋转、翻转、缩放等）
- 数据集划分（训练集、验证集、测试集）

### 2. 模型训练模块 (model_training.py)
- 加载预训练的MobileNetV2模型
- 冻结基础层，添加自定义分类层
- 配置训练参数（优化器、损失函数等）
- 执行迁移学习训练
- 保存训练好的模型

### 3. 模型评估模块 (model_evaluation.py)
- 在测试集上评估模型性能
- 生成混淆矩阵
- 计算准确率、精确率、召回率等指标
- 可视化训练历史

### 4. 预测模块 (predictor.py)
- 加载训练好的模型
- 对单张图片进行预测
- 返回识别结果和置信度

### 5. GUI应用模块 (gui_app.py)
- 创建图形用户界面
- 图片上传功能
- 显示识别结果
- 显示置信度信息

## 数据模型
### 支持的水果类别（共6类）
1. 苹果 (Apple)
2. 香蕉 (Banana)
3. 橙子 (Orange)
4. 葡萄 (Grape)
5. 芒果 (Mango)
6. 西瓜 (Watermelon)

### 模型架构
- **基础模型**: MobileNetV2 (ImageNet预训练权重)
- **输入尺寸**: 224x224x3
- **自定义层**: 
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.5)
  - Dense(6, activation='softmax')

## 技术实现细节

### 1. 配置管理 (`utils/config.py`)
- 集中管理所有项目配置参数
- 包含路径配置、模型参数、训练超参数等
- 支持6个水果类别的中英文映射
- 定义图像尺寸为224x224（MobileNetV2标准输入）

### 2. 图像处理工具 (`utils/image_utils.py`)
- **图像加载**: 使用PIL和Keras加载并归一化图像
- **数据增强**: 支持旋转、平移、缩放、翻转等增强操作
- **预处理管道**: 自动调整大小、归一化到[0,1]区间
- **数据生成器**: 创建带/不带数据增强的ImageDataGenerator

### 3. 数据准备模块 (`src/data_preparation.py`)
- **目录管理**: 自动创建训练、测试、验证数据目录结构
- **数据统计**: 实时统计各类别图片数量
- **数据验证**: 检查数据集完整性
- **数据整理**: 支持从Fruits 360数据集批量导入
- **使用指南**: 提供详细的数据获取和组织指南

### 4. 模型训练模块 (`src/model_training.py`)
- **迁移学习**: 基于ImageNet预训练的MobileNetV2
- **模型架构**:
  - 基础模型: MobileNetV2（冻结所有层）
  - GlobalAveragePooling2D
  - Dense(128) + ReLU
  - Dropout(0.5)
  - Dense(6) + Softmax
- **优化器**: Adam (学习率 0.0001)
- **回调机制**:
  - EarlyStopping: 验证损失5轮不改善则停止
  - ReduceLROnPlateau: 动态降低学习率
  - ModelCheckpoint: 保存最佳模型
- **训练可视化**: 自动生成准确率和损失曲线图
- **微调功能**: 可选的解冻基础模型部分层进行微调

### 5. 模型评估模块 (`src/model_evaluation.py`)
- **性能指标**: 计算准确率、精确率、召回率、F1分数
- **混淆矩阵**: 使用热力图可视化分类结果
- **分类报告**: 生成详细的每类别性能报告
- **类别准确率**: 柱状图展示各水果类别的识别准确率
- **样本预测**: 可视化展示实际预测结果

### 6. 预测模块 (`src/predictor.py`)
- **FruitPredictor类**: 封装预测逻辑
- **单图预测**: 支持单张图片快速预测
- **批量预测**: 支持多张图片批量处理
- **Top-K预测**: 返回前3名预测结果及概率
- **结果格式化**: 输出中英文类别名、置信度、概率分布
- **命令行接口**: 支持通过命令行直接预测

### 7. GUI应用 (`src/gui_app.py`)
- **框架**: Tkinter（Python内置，无需额外安装）
- **界面布局**:
  - 左侧: 图片预览区域（400x400）
  - 右侧: 识别结果显示区域
  - 底部: 状态栏
- **功能特性**:
  - 图片选择和预览
  - 一键识别功能
  - 实时显示置信度和Top-3预测
  - 进度条显示识别状态
  - 多线程处理避免界面冻结
- **用户体验**: 清晰的视觉反馈和错误提示

### 8. 主程序 (`main.py`)
- 统一程序入口
- 自动检查模型文件是否存在
- 提供友好的启动信息和错误提示
- 处理异常和键盘中断

## 代码检查与问题记录
[本部分用于记录代码检查结果和开发过程中遇到的问题及其解决方案。]

## 依赖包列表
```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
pillow>=9.0.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

## 使用说明
### 安装依赖
```powershell
pip install -r requirements.txt
```

### 训练模型
```powershell
python src/model_training.py
```

### 运行GUI应用
```powershell
python main.py
```

## 数据集说明
**推荐数据集**: Kaggle Fruits 360
- **数据集链接**: https://www.kaggle.com/moltean/fruits
- **数据规模**: 包含131种水果，每种水果有多个角度的图片
- **图片规格**: 100x100像素，RGB格式
- **使用方式**: 从中选择6种常见水果进行训练

**数据获取方式**:
1. 注册Kaggle账号并获取API Token
2. 使用Kaggle API下载数据集
3. 或手动下载后解压到data目录

## 项目特点
1. **迁移学习**: 利用预训练模型，减少训练时间和数据需求
2. **轻量级模型**: MobileNetV2适合在普通计算机上运行
3. **友好界面**: Tkinter GUI，操作简单直观
4. **高准确率**: 基于成熟的深度学习架构，识别准确率高
5. **易于扩展**: 可轻松添加更多水果类别

## 后续优化方向
- 支持更多水果类别
- 优化模型以提高识别速度
- 添加批量图片识别功能
- 部署为Web应用
