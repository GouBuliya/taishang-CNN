#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的图像预测脚本 - 使用作者的预训练模型
"""

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import sys
import os

def load_pretrained_model():
    """加载预训练模型"""
    model_path = 'model.h5'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件 {model_path} 不存在!")
        return None
    
    print(f"📦 加载预训练模型: {model_path}")
    model = load_model(model_path)
    print("✅ 模型加载成功!")
    return model

def preprocess_image(image_path):
    """预处理图像"""
    if not os.path.exists(image_path):
        print(f"❌ 图像文件 {image_path} 不存在!")
        return None
    
    # 加载并调整图像尺寸 (与训练时一致)
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    
    # 归一化像素值 (与训练时一致)
    img_array = img_array / 255.0
    
    # 添加批次维度
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_trend(model, image_path):
    """预测K线图趋势"""
    print(f"🔍 分析图像: {image_path}")
    
    # 预处理图像
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    # 进行预测
    prediction = model.predict(processed_image, verbose=0)
    probability = prediction[0][0]
    
    # 解释预测结果
    # 阈值0.5: >0.5为上升趋势, <0.5为下降趋势
    if probability > 0.5:
        trend = "📈 上升趋势 (uptrend)"
        confidence = probability
    else:
        trend = "📉 下降趋势 (downtrend)" 
        confidence = 1 - probability
    
    print(f"预测结果: {trend}")
    print(f"置信度: {confidence:.2%}")
    print(f"原始概率值: {probability:.4f}")
    
    return {
        'trend': 'uptrend' if probability > 0.5 else 'downtrend',
        'probability': probability,
        'confidence': confidence
    }

def batch_predict(model, image_folder):
    """批量预测文件夹中的图像"""
    if not os.path.exists(image_folder):
        print(f"❌ 文件夹 {image_folder} 不存在!")
        return
    
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"❌ 文件夹 {image_folder} 中没有找到图像文件!")
        return
    
    print(f"🔄 开始批量预测，共 {len(image_files)} 张图像...")
    
    results = []
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        print(f"\n--- {i}/{len(image_files)} ---")
        result = predict_trend(model, image_path)
        if result:
            results.append((filename, result))
    
    # 汇总结果
    print(f"\n📊 批量预测完成!")
    print("=" * 50)
    uptrend_count = sum(1 for _, r in results if r['trend'] == 'uptrend')
    downtrend_count = len(results) - uptrend_count
    
    print(f"上升趋势: {uptrend_count} 张")
    print(f"下降趋势: {downtrend_count} 张")
    
    return results

def main():
    """主函数"""
    print("🚀 使用作者预训练模型进行趋势预测")
    print("=" * 50)
    
    # 加载模型
    model = load_pretrained_model()
    if model is None:
        return
    
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print(f"单张图像预测: python {sys.argv[0]} <图像路径>")
        print(f"批量预测: python {sys.argv[0]} <图像文件夹>")
        print("\n示例:")
        print(f"python {sys.argv[0]} chart_images5_1/uptrend/uptrend_104.png")
        print(f"python {sys.argv[0]} chart_images5_1/uptrend/")
        return
    
    target_path = sys.argv[1]
    
    if os.path.isfile(target_path):
        # 单张图像预测
        predict_trend(model, target_path)
    elif os.path.isdir(target_path):
        # 批量预测
        batch_predict(model, target_path)
    else:
        print(f"❌ 路径 {target_path} 不存在!")

if __name__ == "__main__":
    main() 