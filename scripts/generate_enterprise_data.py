# scripts/generate_enterprise_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 配置参数
DATE_FORMAT = "%Y-%m-%d"
FINANCIAL_END_DATE = "2023-09-30"
GENERATE_PERSONAL = False  # 设为 True 可重新生成个人数据


def generate_enterprise_data(n_samples: int = 500) -> pd.DataFrame:
    """生成企业风险数据（增强版）"""
    np.random.seed(42)
    start_date = datetime(2000, 1, 1)

    # 基础字段生成
    data = {
        "company_id": [f"E{i:04d}" for i in range(1, n_samples + 1)],
        "establishment_date": [
            (start_date + timedelta(days=np.random.randint(0, 8000))).strftime(DATE_FORMAT)
            for _ in range(n_samples)
        ],
        "last_financial_report_date": FINANCIAL_END_DATE,
        "monthly_income": np.random.lognormal(14, 0.4, n_samples).round(2),  # 单位：万元
        "monthly_expense": np.random.lognormal(13, 0.3, n_samples).round(2),
        "total_liabilities": np.random.lognormal(16, 0.5, n_samples).round(2),
        "total_assets": np.random.lognormal(17, 0.4, n_samples).round(2),
        "industry_type": np.random.choice(
            ["制造业", "零售业", "科技", "金融", "建筑业"],
            size=n_samples,
            p=[0.3, 0.25, 0.2, 0.15, 0.1]
        )
    }

    # 财务数据校验
    data["total_assets"] = np.maximum(
        data["total_assets"],
        data["total_liabilities"] * 1.2 + 1e6  # 资产 >= 负债*1.2 + 100万
    )

    # 动态风险计算
    cash_flow = data["monthly_income"] - data["monthly_expense"]
    debt_ratio = data["total_liabilities"] / data["total_assets"]
    data["risk_level"] = np.where(
        (debt_ratio > 0.6) |
        (cash_flow < 0) |
        (data["total_assets"] < 1e7),
        1, 0
    )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # 路径配置
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按需生成个人数据
    if GENERATE_PERSONAL:
        from generate_data import generate_personal_data  # 复用原有生成函数

        personal_df = generate_personal_data()
        personal_df.to_csv(output_dir / "personal_data.csv", index=False)
        print("个人数据已重新生成")

    # 生成企业数据
    enterprise_df = generate_enterprise_data()
    enterprise_df.to_csv(output_dir / "enterprise_data.csv", index=False)
    print(f"企业数据已保存至：{output_dir / 'enterprise_data.csv'}")
    print(f"数据样本：\n{enterprise_df.head(2)}")