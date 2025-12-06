# 水果识别系统 - 使用说明书

## 🎯 项目介绍

这是一个基于深度学习的水果图像识别系统，采用迁移学习技术，能够准确识别6种常见水果。系统提供友好的图形用户界面，让用户能够轻松上传图片并获得识别结果。

### ✨ 核心特性
- 🧠 基于MobileNetV2的迁移学习
- 🎯 支持6种常见水果识别
- 🖥️ 友好的GUI图形界面
- 📊 详细的置信度显示
- ⚡ 快速的识别速度
- 📈 完整的模型评估工具

## 📦 完整使用流程

### 第一步：环境准备

#### 1.1 检查Python版本
```powershell
python --version
```
确保Python版本 >= 3.8

#### 1.2 安装依赖包
```powershell
# 进入项目目录
cd d:\akaifa\pysg

# 安装所有依赖
pip install -r requirements.txt
```

**预计安装时间**: 5-10分钟（取决于网络速度）

### 第二步：准备训练数据

#### 2.1 运行数据准备脚本
```powershell
python src/data_preparation.py
```

此脚本会：
- ✅ 创建数据目录结构
- 📋 显示数据获取指南
- 📊 统计当前数据情况

#### 2.2 下载数据集（三种方式）

**方式A: Kaggle API（推荐）**
```powershell
# 安装Kaggle CLI
pip install kaggle

# 下载数据集（需要先配置Kaggle API Token）
kaggle datasets download -d moltean/fruits

# 解压数据集
# 然后在 data_preparation.py 中取消注释相关代码来整理数据
```

**方式B: 手动下载**
1. 访问 https://www.kaggle.com/moltean/fruits
2. 下载数据集ZIP文件
3. 解压并按照目录结构组织到 `data/` 文件夹

**方式C: 小规模测试**
准备少量图片（每类10-20张）放入对应目录：
```
data/
├── train/
│   ├── Apple/      (放入苹果图片)
│   ├── Banana/     (放入香蕉图片)
│   ├── Grape/      (放入葡萄图片)
│   ├── Mango/      (放入芒果图片)
│   ├── Orange/     (放入橙子图片)
│   └── Watermelon/ (放入西瓜图片)
└── test/
    └── (同上结构)
```

### 第三步：训练模型

#### 3.1 开始训练
```powershell
python src/model_training.py
```

**训练过程**:
- 📥 加载预训练的MobileNetV2模型
- 🏗️ 构建自定义分类层
- 🔄 开始迁移学习训练
- 💾 自动保存最佳模型
- 📊 生成训练历史图表

**预计训练时间**:
- CPU: 20-40分钟
- GPU: 5-15分钟

**训练输出**:
- `models/fruit_model.h5` - 训练好的模型
- `models/training_history.png` - 训练曲线图

#### 3.2 查看训练结果
训练完成后会生成准确率和损失曲线图，保存在 `models/training_history.png`

### 第四步：评估模型（可选但推荐）

```powershell
python src/model_evaluation.py
```

**评估输出**:
- 📊 测试集准确率
- 🔲 混淆矩阵 (`models/confusion_matrix.png`)
- 📋 分类报告 (`models/classification_report.txt`)
- 📈 类别准确率 (`models/class_accuracy.png`)
- 🖼️ 样本预测可视化 (`models/sample_predictions.png`)

### 第五步：运行GUI应用

#### 5.1 启动程序
```powershell
python main.py
```

#### 5.2 使用GUI界面

**界面说明**:
```
┌─────────────────────────────────────────────────────┐
│          🍎 水果识别系统                             │
├──────────────────┬──────────────────────────────────┤
│                  │                                  │
│   图片预览区     │      识别结果区                  │
│                  │                                  │
│   [图片显示]     │   🎯 预测类别: 苹果               │
│                  │   📊 置信度: 98.5%               │
│   📁 选择图片    │                                  │
│   🔍 开始识别    │   Top 3 预测结果:                │
│                  │   1. 苹果   98.5%                │
│                  │   2. 橙子    0.8%                │
│                  │   3. 芒果    0.4%                │
└──────────────────┴──────────────────────────────────┘
│ 状态栏: 识别完成                                    │
└─────────────────────────────────────────────────────┘
```

**操作步骤**:
1. 点击「📁 选择图片」按钮
2. 浏览并选择水果图片（支持JPG、PNG等格式）
3. 图片会在左侧预览区显示
4. 点击「🔍 开始识别」按钮
5. 等待识别完成（通常1-2秒）
6. 右侧会显示识别结果和置信度

### 第六步：命令行预测（可选）

如果不想使用GUI，可以直接命令行预测：

```powershell
python src/predictor.py your_image.jpg
```

## 🎨 支持的水果类别

| 中文名 | 英文名     | 说明           |
|--------|-----------|----------------|
| 苹果   | Apple     | 各种颜色的苹果 |
| 香蕉   | Banana    | 新鲜香蕉       |
| 葡萄   | Grape     | 成串葡萄       |
| 芒果   | Mango     | 黄色芒果       |
| 橙子   | Orange    | 圆形橙子       |
| 西瓜   | Watermelon| 西瓜切面或整个 |

## 📊 性能说明

### 预期准确率
- **训练集准确率**: 95-99%
- **测试集准确率**: 85-95%（取决于数据质量）

### 影响准确率的因素
1. **图片质量**: 清晰度越高越好
2. **拍摄角度**: 正面拍摄效果最佳
3. **光照条件**: 光线充足更好
4. **背景干扰**: 简洁背景效果更好
5. **水果完整性**: 完整水果比切片更容易识别

### 性能指标
- **单图预测时间**: 0.5-2秒
- **批量处理**: 支持
- **内存占用**: 约200-500MB

## ⚙️ 高级配置

### 修改配置参数
编辑 `utils/config.py` 可以调整：
- 图像尺寸
- 批次大小
- 训练轮数
- 学习率
- 数据增强参数
- GUI窗口大小

### 添加新的水果类别
1. 在 `FRUIT_CLASSES` 中添加新类别
2. 在 `CLASS_NAMES_CN` 中添加中文名
3. 准备该类别的训练和测试图片
4. 重新训练模型

### 微调模型
如果初次训练效果不理想，可以微调：
```powershell
# 在 model_training.py 末尾取消注释微调代码
python src/model_training.py
```

## ❓ 常见问题解答

### Q1: 为什么识别准确率不高？
**A**: 可能的原因：
- 训练数据太少（建议每类至少50张）
- 图片质量差（模糊、昏暗）
- 背景过于复杂
- 水果外观差异大

**解决方案**:
- 增加训练数据
- 使用高质量图片
- 进行模型微调

### Q2: 训练时内存不足怎么办？
**A**: 调整 `config.py` 中的 `BATCH_SIZE`，改为16或8

### Q3: 可以识别未训练的水果吗？
**A**: 不可以。系统只能识别训练过的6种水果。要识别新水果需要重新训练。

### Q4: 支持实时摄像头识别吗？
**A**: 当前版本不支持，但架构上可以轻松扩展此功能。

### Q5: 可以部署为Web应用吗？
**A**: 可以！可以使用Flask/Django等框架将其包装为Web服务。

## 🛠️ 故障排除

### 问题：导入错误
```
ModuleNotFoundError: No module named 'tensorflow'
```
**解决**: `pip install tensorflow`

### 问题：找不到模型文件
```
FileNotFoundError: 模型文件不存在
```
**解决**: 先运行 `python src/model_training.py` 训练模型

### 问题：Tkinter错误
```
ImportError: No module named '_tkinter'
```
**解决**: 
- Windows: 重新安装Python并勾选"tcl/tk"
- Linux: `sudo apt-get install python3-tk`

## 📞 获取支持

遇到问题时的检查清单：
- [ ] Python版本是否 >= 3.8
- [ ] 是否安装了所有依赖包
- [ ] 数据目录结构是否正确
- [ ] 是否已训练模型
- [ ] 图片格式是否支持

## 📚 附加资源

### 相关文档
- `README.md` - 项目整体说明
- `QUICKSTART.md` - 快速开始指南
- 本文件 - 详细使用说明

### 数据集资源
- Kaggle Fruits 360: https://www.kaggle.com/moltean/fruits
- ImageNet: https://www.image-net.org/

### 技术文档
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- MobileNetV2论文: https://arxiv.org/abs/1801.04381

---

**祝您使用愉快！** 🎉

如有任何问题或建议，欢迎反馈。
