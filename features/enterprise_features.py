# features/enterprise_features.py
import numpy as np
import pandas as pd


class FinancialAnalyzer:
    """企业财务分析器"""

    @staticmethod
    def financial_health(row: pd.Series) -> dict:
        """综合财务健康评估"""
        ratios = {
            'debt_ratio': row['total_liabilities'] / (row['total_assets'] + 1e-6),
            'current_ratio': row['total_assets'] / (row['total_liabilities'] + 1e-6),
            'profit_margin': (row['monthly_income'] - row['monthly_expense']) / (row['monthly_income'] + 1e-6)
        }

        health_status = 'good'
        if ratios['debt_ratio'] > 0.7:
            health_status = 'risky'
        elif ratios['profit_margin'] < 0.1:
            health_status = 'warning'

        return {**ratios, 'health_status': health_status}


def generate_enterprise_features(df: pd.DataFrame) -> pd.DataFrame:
    """企业特征生成"""
    if {'total_liabilities', 'total_assets'}.issubset(df.columns):
        df['debt_ratio'] = df['total_liabilities'] / (df['total_assets'] + 1e-6)

    if {'monthly_income', 'monthly_expense'}.issubset(df.columns):
        financial_features = df.apply(FinancialAnalyzer.financial_health, axis=1).apply(pd.Series)
        df = pd.concat([df, financial_features.add_prefix('fin_')], axis=1)

    if 'industry_type' in df.columns:
        df = pd.get_dummies(df, columns=['industry_type'], prefix='industry')

    return df