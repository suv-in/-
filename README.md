# 植物分类深度学习项目

这是一个基于深度学习的植物图像分类项目，使用PyTorch框架实现。

## 项目概述

本项目实现了对100种不同植物的图像分类，使用ResNet18作为基础模型，并进行了GPU优化训练。

## 项目结构

```
submission/
├── code/                 # 源代码目录
│   ├── model.py         # 模型定义
│   ├── train.py         # 训练脚本
│   ├── predict.py       # 预测脚本
│   ├── utils.py         # 工具函数
│   └── requirements.txt  # 依赖包列表
├── model/               # 模型文件
│   ├── best_model.pth   # 训练好的模型权重
│   └── config.json      # 模型配置
└── results/             # 训练结果
```

## 环境要求

- Python 3.8+
- PyTorch 2.2.1
- torchvision 0.17.1
- 其他依赖见 `code/requirements.txt`

## 安装依赖

```bash
pip install -r code/requirements.txt
```

## 使用方法

### 训练模型

```bash
cd code
python train.py
```

### 使用模型进行预测

```bash
cd code
python predict.py --image_path <图片路径>
```

## 模型性能

- 输入图像尺寸：224x224
- 类别数量：100
- 批量大小：32
- 训练轮数：20

## 许可证

MIT License