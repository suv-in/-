import os
import csv
import json
import logging
from PIL import Image
import torch
from torchvision import transforms

class Config:
    """配置类"""
    
    def __init__(self):
        self.img_size = 224
        self.batch_size = 32
        self.num_epochs = 20
        self.learning_rate = 0.001
        self.num_classes = 100
        self.model_name = 'resnet18'
    
    def to_dict(self):
        return {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }

def setup_logging(log_file='training.log'):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_transform(img_size=224, is_train=True):
    """获取数据预处理变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_label_mapping(label_file):
    """创建标签映射"""
    labels_set = set()
    
    with open(label_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过头部
        for row in reader:
            if len(row) >= 2:
                label = int(row[1])
                labels_set.add(label)
    
    # 创建映射
    sorted_labels = sorted(labels_set)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    reverse_mapping = {new_label: old_label for old_label, new_label in label_mapping.items()}
    
    return label_mapping, reverse_mapping

def save_config(config, filepath):
    """保存配置到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config(filepath):
    """从JSON文件加载配置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)