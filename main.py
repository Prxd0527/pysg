"""
水果识别系统 - 主程序入口
Fruit Recognition System
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui_app import run_gui


def main():
    """主函数"""
    print("="*60)
    print("🍎 水果识别系统启动中...")
    print("Fruit Recognition System")
    print("="*60)
    
    # 检查模型是否存在
    from utils.config import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print("\n⚠️  警告: 模型文件不存在！")
        print(f"路径: {MODEL_PATH}")
        print("\n请先训练模型:")
        print("  python src/model_training.py")
        print("\n或者准备好数据后再训练模型。")
        print("\n现在将启动GUI，但识别功能不可用。")
        print("="*60)
    
    # 运行GUI
    try:
        run_gui()
    except KeyboardInterrupt:
        print("\n\n程序已手动终止。")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
