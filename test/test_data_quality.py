# 检查字段是否存在
required_columns = [
    'consumption_records', 'loan_records', 'total_debt',
    'total_income', 'income_records'
]
missing = [col for col in required_columns if col not in personal_df.columns]
if missing:
    print(f"缺失关键字段: {missing}")
else:
    print("数据字段完整")