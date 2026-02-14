#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_usage():
   "Print Instructions"
    print("""
NOVA LLM Nutrition Assessment Program
How to use:
  python main.py <command>

Available commands:
  generate    - Generate training data (nutritional assessment)
  train       - Train the model and perform visualization analysis.
  help        - Show this help information

示例:
  python main.py generate    # Generate nutritional assessment data
  python main.py train --sample-size 50    # Training the model (50 samples, quick test)
  python main.py train    # Optimal Configuration Training: Multiple Models + Parameter Search + Complete Visualization
""")

def main():
    """Main Function"""
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
            print(f"Unknown command: {command}")
            print_usage()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure that all dependent modules exist in the src directory.")
    except Exception as e:
        print(f"Execution error: {e}")

if __name__ == "__main__":

    main()
