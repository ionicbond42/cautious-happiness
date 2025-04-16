# features/base_features.py
import numpy as np
import pandas as pd
from typing import List, Union
from datetime import datetime


def income_stability(income_records: Union[List[float], str]) -> float:
    """通用收入稳定性计算（兼容字符串存储的列表）"""
    if isinstance(income_records, str):
        records = eval(income_records)
    else:
        records = income_records

    if len(records) < 2:
        return 0.0

    avg = np.mean(records)
    if avg == 0:
        return 0.0
    return np.std(records) / avg


def generate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """基础特征生成（自动区分个人/企业）"""
    # 公共特征
    if 'income_records' in df.columns:
        df['income_stability'] = df['income_records'].apply(income_stability)

    # 个人特征
    if 'total_debt' in df.columns and 'total_income' in df.columns:
        df['debt_ratio'] = df['total_debt'] / (df['total_income'] + 1e-6)

    # 企业特征
    if 'monthly_income' in df.columns and 'monthly_expense' in df.columns:
        df['cash_flow'] = df['monthly_income'] - df['monthly_expense']
        df['cash_flow_ratio'] = df['cash_flow'] / (df['monthly_income'] + 1e-6)

    # 时间特征
    date_cols = {
        'personal': ('first_credit_date', 'current_date'),
        'enterprise': ('establishment_date', 'last_financial_report_date')
    }

    for col_pair in date_cols.values():
        if all(c in df.columns for c in col_pair):
            df['history_days'] = (df[col_pair[1]] - df[col_pair[0]]).dt.days
            break

    return df