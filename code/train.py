#!/usr/bin/env python3
"""
训练脚本 - 植物分类模型训练（GPU优化版）
"""
import os
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageFile

# 处理截断图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from utils import Config, setup_logging, get_transform, create_label_mapping, save_config

class PlantDataset:
    """植物图像数据集"""
    
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.img_labels = []
        self.transform = transform
        self.label_mapping = {}
        self.reverse_mapping = {}
        
        # 创建标签映射
        self.label_mapping, self.reverse_mapping = create_label_mapping(label_file)
        
        # 读取标签文件
        with open(label_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过头部
            for row in reader:
                if len(row) >= 2:
                    img_name, label = row[0], int(row[1])
                    img_path = os.path.join(img_dir, img_name)
                    if os.path.exists(img_path):
                        # 应用标签映射
                        mapped_label = self.label_mapping[label]
                        self.img_labels.append((img_name, mapped_label))
        
        self.logger = setup_logging()
        self.logger.info(f"✅ 加载了 {len(self.img_labels)} 个训练样本")
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            self.logger.warning(f"⚠️ 加载图像失败 {img_path}: {e}")
            # 返回一个空白图像
            return torch.zeros(3, 224, 224), label

def train():
    """训练函数"""
    logger = setup_logging()
    logger.info("🚀 开始训练植物分类模型...")
    
    # 配置参数
    config = Config()
    
    # 创建数据集
    dataset = PlantDataset(
        img_dir='/workspace/train',
        label_file='/workspace/train_labels00000.csv',
        transform=get_transform(config.img_size, is_train=True)
    )
    
    if len(dataset) == 0:
        logger.error("❌ 没有找到有效数据，训练终止")
        return
    
    # 创建数据加载器（优化GPU使用率）
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size * 4,  # 增加批处理大小
        shuffle=True,
        num_workers=8,  # 增加数据加载线程
        pin_memory=True,  # 启用内存锁定
        persistent_workers=True,  # 保持工作进程
        prefetch_factor=2  # 预取因子
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config.to_dict())
    model = model.to(device)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # 混合精度训练
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    logger.info(f"📊 训练配置（GPU优化版）:")
    logger.info(f"  设备: {device}")
    logger.info(f"  批次大小: {config.batch_size * 4}")
    logger.info(f"  学习率: {config.learning_rate}")
    logger.info(f"  总轮数: {config.num_epochs}")
    logger.info(f"  类别数: {config.num_classes}")
    logger.info(f"  混合精度训练: {scaler is not None}")
    logger.info(f"  数据加载线程: 8")
    
    # 训练循环（GPU优化）
    model.train()
    best_acc = 0.0
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            batch_start_time = time.time()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            if scaler:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            batch_time = time.time() - batch_start_time
            
            if batch_idx % 10 == 0:
                accuracy = 100. * correct / total if total > 0 else 0
                
                # GPU使用率监控（安全版本）
                if device.type == 'cuda':
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                        # 安全的GPU利用率获取
                        gpu_util = 0
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            pynvml.nvmlShutdown()
                        except:
                            gpu_util = 0
                        
                        logger.info(f'Epoch: {epoch+1}/{config.num_epochs}, '
                                  f'Batch: {batch_idx}/{len(dataloader)}, '
                                  f'Loss: {loss.item():.4f}, '
                                  f'Acc: {accuracy:.2f}%, '
                                  f'Time: {batch_time:.3f}s, '
                                  f'GPU: {gpu_memory:.1f}GB/{gpu_util}%')
                    except Exception as e:
                        logger.info(f'Epoch: {epoch+1}/{config.num_epochs}, '
                                  f'Batch: {batch_idx}/{len(dataloader)}, '
                                  f'Loss: {loss.item():.4f}, '
                                  f'Acc: {accuracy:.2f}%, '
                                  f'Time: {batch_time:.3f}s')
                else:
                    logger.info(f'Epoch: {epoch+1}/{config.num_epochs}, '
                              f'Batch: {batch_idx}/{len(dataloader)}, '
                              f'Loss: {loss.item():.4f}, '
                              f'Acc: {accuracy:.2f}%, '
                              f'Time: {batch_time:.3f}s')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            os.makedirs('../model', exist_ok=True)
            torch.save(model.state_dict(), '../model/best_model.pth')
            
            # 保存配置和标签映射
            config_dict = config.to_dict()
            config_dict['label_mapping'] = {str(k): v for k, v in dataset.label_mapping.items()}
            config_dict['reverse_mapping'] = {str(k): v for k, v in dataset.reverse_mapping.items()}
            save_config(config_dict, '../model/config.json')
            logger.info(f"💾 保存最佳模型，准确率: {accuracy:.2f}%")
            logger.info(f"💾 保存标签映射: {len(dataset.label_mapping)} 类映射到 {len(set(dataset.label_mapping.values()))} 类")
        
        # GPU性能统计
        if device.type == 'cuda':
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            torch.cuda.reset_peak_memory_stats()
            logger.info(f'✅ Epoch {epoch+1}完成 - 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%, '
                      f'时间: {epoch_time:.1f}s, 学习率: {current_lr:.6f}, GPU峰值: {gpu_memory:.1f}GB')
        else:
            logger.info(f'✅ Epoch {epoch+1}完成 - 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%, '
                      f'时间: {epoch_time:.1f}s, 学习率: {current_lr:.6f}')
    
    logger.info(f"🎉 训练完成! 最佳准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    train()