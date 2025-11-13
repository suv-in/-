import torch
import torch.nn as nn
from torchvision import models

class PlantClassifier(nn.Module):
    """植物分类模型"""
    
    def __init__(self, num_classes=100, model_name='resnet18'):
        super(PlantClassifier, self).__init__()
        
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    def forward(self, x):
        return self.model(x)

def create_model(config):
    """根据配置创建模型"""
    model = PlantClassifier(
        num_classes=config['num_classes'],
        model_name=config['model_name']
    )
    return model