# scripts/generate_data.py
import pandas as pd
import numpy as np
from pathlib import Path

def generate_personal_data(n_samples: int = 1000) -> pd.DataFrame:
    data = {
        "user_id": np.arange(1, n_samples + 1),
        "first_credit_date": pd.date_range(start="2010-01-01", periods=n_samples).strftime("%Y-%m-%d"),
        "current_date": "2023-10-15",
        "income_records": [str([np.random.randint(2000, 8000) for _ in range(3)]) for _ in range(n_samples)],
        "login_records": [str([np.random.randint(1, 20) for _ in range(3)]) for _ in range(n_samples)],
        "unusual_operations": ["[]" if np.random.rand() > 0.1 else "['suspicious_login']" for _ in range(n_samples)],
        "total_debt": np.random.lognormal(mean=8, sigma=0.5, size=n_samples).round(2),
        "total_income": np.random.lognormal(mean=10, sigma=0.3, size=n_samples).round(2),
        "consumption_records": [str([np.random.randint(50, 1000) for _ in range(3)]) for _ in range(n_samples)],
        "loan_records": [str([np.random.randint(1000, 50000)]) if np.random.rand() > 0.7 else "[]" for _ in range(n_samples)],
        "credit_grade": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    personal_df = generate_personal_data()
    personal_df.to_csv(project_root / "data/generated/personal_data.csv", index=False)
