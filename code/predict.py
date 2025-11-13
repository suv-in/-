import os
import sys
import argparse
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from utils import setup_logging, load_config

def predict(test_dir, output_file):
    """预测函数"""
    logger = setup_logging()
    logger.info(f"🚀 开始预测: {test_dir}")
    
    # 加载模型配置
    config_path = '../model/config.json'
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return
    
    config = load_config(config_path)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    
    model_path = '../model/best_model.pth'
    if os.path.exists(model_path):
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        
        # 直接加载状态字典到模型
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ 模型加载成功: {model_path}")
    else:
        logger.error(f"❌ 模型文件不存在: {model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 获取测试图像文件
    image_files = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(file)
    
    logger.info(f"📊 找到 {len(image_files)} 个测试图像")
    
    # 预测结果
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                # 将模型输出的类别ID映射回原始类别ID
                predicted_idx_value = predicted_idx.item()
                confidence_value = confidence.item()
                
                # 反向映射：从模型内部ID到原始类别ID
                reverse_mapping = {v: int(k) for k, v in config['label_mapping'].items()}
                predicted_original_label = reverse_mapping.get(predicted_idx_value, -1)
                
                results.append({
                    'filename': img_file,
                    'category_id': predicted_original_label,
                    'confidence': confidence_value
                })
                
                logger.info(f"📸 {img_file} -> 类别: {predicted_original_label}, 置信度: {confidence_value:.4f}")
                
        except Exception as e:
            logger.warning(f"⚠️ 处理图像失败 {img_file}: {e}")
            results.append({
                'filename': img_file,
                'category_id': -1,
                'confidence': 0.0
            })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存结果到CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'category_id', 'confidence'])
        
        for result in results:
            writer.writerow([
                result['filename'],
                result['category_id'],
                f"{result['confidence']:.4f}"
            ])
    
    logger.info(f"✅ 预测完成！结果保存到: {output_file}")
    
    # 统计信息
    successful_predictions = len([r for r in results if r['category_id'] != -1])
    avg_confidence = sum([r['confidence'] for r in results if r['category_id'] != -1]) / successful_predictions
    
    logger.info(f"📊 预测统计:")
    logger.info(f"   总图像数: {len(image_files)}")
    logger.info(f"   成功预测: {successful_predictions}")
    logger.info(f"   平均置信度: {avg_confidence:.4f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='植物分类预测脚本')
    parser.add_argument('test_dir', help='测试集文件夹路径')
    parser.add_argument('output_file', help='输出结果文件路径')
    parser.add_argument('--label_file', type=str, default=None, help='训练标签文件路径（可选）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_dir):
        print(f"❌ 测试文件夹不存在: {args.test_dir}")
        sys.exit(1)
    
    predict(args.test_dir, args.output_file)

if __name__ == "__main__":
    main()