#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器 - 用于验证和管理 config.json 配置文件

@author: AI Assistant
"""

import json
import os
import sys
from pathlib import Path

def load_config(config_path='config.json'):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 配置文件 {config_path} 加载成功!")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件 {config_path} 不存在!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None

def validate_config(config):
    """
    验证配置文件的完整性
    
    Args:
        config (dict): 配置字典
        
    Returns:
        bool: 验证是否通过
    """
    required_sections = [
        'gpu', 'data', 'time_window', 'model', 'image_processing',
        'technical_indicators', 'chart_plotting', 'signal_annotation',
        'trading', 'training', 'model_architecture', 'ui', 'pattern_recognition'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ 缺少配置节: {', '.join(missing_sections)}")
        return False
    
    # 验证文件路径
    file_checks = [
        ('data.test_data_file', config['data']['test_data_file']),
        ('data.training_data_file', config['data']['training_data_file']),
    ]
    
    missing_files = []
    for name, filepath in file_checks:
        if not os.path.exists(filepath):
            missing_files.append(f"{name}: {filepath}")
    
    if missing_files:
        print("⚠️  以下文件不存在:")
        for file_info in missing_files:
            print(f"   - {file_info}")
    
    print("✅ 配置文件验证通过!")
    return True

def print_config_summary(config):
    """
    打印配置摘要
    
    Args:
        config (dict): 配置字典
    """
    print("\n📋 配置摘要:")
    print("=" * 50)
    
    print(f"GPU设置:")
    print(f"  - 内存限制: {config['gpu']['memory_limit_mb']}MB")
    
    print(f"\n数据设置:")
    print(f"  - 测试数据: {config['data']['test_data_file']}")
    print(f"  - 训练数据: {config['data']['training_data_file']}")
    print(f"  - 分隔符: '{config['data']['delimiter']}'")
    
    print(f"\n时间窗口:")
    print(f"  - 起始索引: {config['time_window']['initial_time_index']}")
    print(f"  - 结束索引: {config['time_window']['final_time_index']}")
    print(f"  - 窗口大小: {config['time_window']['window_size']}")
    print(f"  - 移动步长: {config['time_window']['shift_size']}")
    
    print(f"\n模型设置:")
    print(f"  - 模型文件: {config['model']['model_file']}")
    print(f"  - 预测阈值: {config['model']['prediction_threshold']}")
    
    print(f"\n图像处理:")
    print(f"  - 目标尺寸: {config['image_processing']['target_size']}")
    print(f"  - 颜色模式: {config['image_processing']['color_mode']}")
    print(f"  - 输出目录: {config['image_processing']['output_dir']}")
    
    print(f"\n技术指标:")
    print(f"  - SMA周期: {config['technical_indicators']['sma_period']}")
    print(f"  - SMA颜色: {config['technical_indicators']['sma_color']}")
    
    print(f"\n交易设置:")
    print(f"  - 初始资金: ${config['trading']['initial_amount_usd']}")
    print(f"  - 买入信号: {config['trading']['buy_signal']}")
    print(f"  - 卖出信号: {config['trading']['sell_signal']}")
    
    print(f"\n训练参数:")
    print(f"  - 批次大小: {config['training']['batch_size']}")
    print(f"  - 训练轮数: {config['training']['epochs']}")
    print(f"  - 学习率: {config['training']['learning_rate']}")
    print(f"  - Dropout率: {config['training']['dropout_rate']}")

def update_config_value(config, key_path, new_value):
    """
    更新配置值
    
    Args:
        config (dict): 配置字典
        key_path (str): 配置路径，如 'gpu.memory_limit_mb'
        new_value: 新值
        
    Returns:
        bool: 更新是否成功
    """
    keys = key_path.split('.')
    current = config
    
    # 导航到目标位置
    for key in keys[:-1]:
        if key not in current:
            print(f"❌ 配置路径 {key_path} 不存在!")
            return False
        current = current[key]
    
    # 更新值
    final_key = keys[-1]
    if final_key not in current:
        print(f"❌ 配置项 {key_path} 不存在!")
        return False
    
    old_value = current[final_key]
    current[final_key] = new_value
    print(f"✅ 已更新 {key_path}: {old_value} -> {new_value}")
    return True

def save_config(config, config_path='config.json'):
    """
    保存配置文件
    
    Args:
        config (dict): 配置字典
        config_path (str): 配置文件路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"✅ 配置已保存到 {config_path}")
        return True
    except Exception as e:
        print(f"❌ 保存配置失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 TradeCNN 配置管理器")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  {sys.argv[0]} validate                    # 验证配置文件")
        print(f"  {sys.argv[0]} summary                     # 显示配置摘要")
        print(f"  {sys.argv[0]} update <key_path> <value>   # 更新配置值")
        print("\n示例:")
        print(f"  {sys.argv[0]} validate")
        print(f"  {sys.argv[0]} summary")
        print(f"  {sys.argv[0]} update gpu.memory_limit_mb 8192")
        print(f"  {sys.argv[0]} update training.epochs 30")
        return
    
    command = sys.argv[1]
    config = load_config()
    
    if config is None:
        return
    
    if command == 'validate':
        validate_config(config)
    
    elif command == 'summary':
        print_config_summary(config)
    
    elif command == 'update':
        if len(sys.argv) != 4:
            print("❌ update 命令需要提供 key_path 和 value 参数")
            return
        
        key_path = sys.argv[2]
        value_str = sys.argv[3]
        
        # 尝试解析值的类型
        try:
            # 尝试解析为数字
            if '.' in value_str:
                new_value = float(value_str)
            else:
                new_value = int(value_str)
        except ValueError:
            # 尝试解析为布尔值
            if value_str.lower() in ['true', 'false']:
                new_value = value_str.lower() == 'true'
            else:
                # 作为字符串处理
                new_value = value_str
        
        if update_config_value(config, key_path, new_value):
            save_config(config)
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main() 