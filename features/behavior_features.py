# features/behavior_features.py
import pandas as pd
import numpy as np
from typing import List, Union


class BehaviorAnalyzer:
    """行为特征分析器（个人专用）"""

    @staticmethod
    def login_frequency(login_records: Union[List[int], str], period: int = 30) -> float:
        """标准化登录频率计算"""
        if isinstance(login_records, str):
            records = eval(login_records)
        else:
            records = login_records

        recent = records[-period:]
        return len(recent) / period if period > 0 else 0.0

    @staticmethod
    def consumption_analysis(consumption_records: Union[List[float], str]) -> dict:
        """消费行为多维分析"""
        if isinstance(consumption_records, str):
            records = eval(consumption_records)
        else:
            records = consumption_records

        return {
            'total': sum(records),
            'avg': np.mean(records) if records else 0,
            'max': max(records) if records else 0,
            'category': 'high' if sum(records) > 10000 else 'normal'
        }


def generate_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """行为特征生成（仅限个人数据）"""
    if 'login_records' in df.columns:
        df['login_freq_30d'] = df['login_records'].apply(
            BehaviorAnalyzer.login_frequency
        )

    if 'consumption_records' in df.columns:
        consumption_features = df['consumption_records'].apply(
            BehaviorAnalyzer.consumption_analysis
        ).apply(pd.Series)
        df = pd.concat([df, consumption_features.add_prefix('cons_')], axis=1)

    return df