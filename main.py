#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_usage():
    """打印使用说明"""
    print("""
NOVA LLM 营养评估项目

使用方法:
  python main.py <command>

可用命令:
  generate    - 生成训练数据 (营养评估)
  train       - 训练模型并做可视化分析
  help        - 显示此帮助信息

示例:
  python main.py generate    # 生成营养评估数据
  python main.py train --sample-size 50    # 训练模型(50个样本，快速测试)
  python main.py train    # 最佳配置训练：多模型+参数搜索+完整可视化
""")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    try:
        if command == 'generate':
            from src.generate_training_data import main as generate_main
            generate_main()
        elif command == 'train':
            import subprocess
            result = subprocess.run([sys.executable, 'src/train_proxy_model.py'] + sys.argv[2:])
            sys.exit(result.returncode)
        elif command in ['help', '--help', '-h']:
            print_usage()
        else:
            print(f"未知命令: {command}")
            print_usage()

    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖模块都存在于src目录中")
    except Exception as e:
        print(f"执行错误: {e}")

if __name__ == "__main__":
    main()