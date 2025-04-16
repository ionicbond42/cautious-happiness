# run_pipeline.py
import subprocess
import sys
from pathlib import Path


def main():
    print("=== 开始执行自动化流程 ===")

    # 获取项目根目录（根据当前脚本位置）
    project_root = Path(__file__).parent

    # 生成数据
    generate_script = project_root / "scripts" / "generate_data.py"
    subprocess.run([sys.executable, str(generate_script)], check=True)

    # 运行主程序
    model_script = project_root / "models" / "risk_model.py"
    subprocess.run([sys.executable, str(model_script)], check=True)

    print("=== 流程执行完毕 ===")
    input("按 Enter 键继续...")


if __name__ == "__main__":
    main()