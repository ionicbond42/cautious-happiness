import pandas as pd
from features.base_features import generate_base_features
from features.behavior_features import generate_behavior_features
from features.enterprise_features import generate_enterprise_features

# 加载数据
personal_df = pd.read_csv('data/personal_data.csv')
enterprise_df = pd.read_csv('data/enterprise_data.csv')

# 确保目标变量存在
assert 'credit_grade' in personal_df.columns, "Target variable 'credit_grade' not found in personal data"
assert 'risk_level' in enterprise_df.columns, "Target variable 'risk_level' not found in enterprise data"

# 检查数据完整性与一致性
required_personal_fields = ['income_records', 'total_debt', 'total_income', 'first_credit_date', 'current_date', 'login_records', 'unusual_operations', 'monthly_income', 'monthly_expense']
required_enterprise_fields = ['monthly_income', 'monthly_expense', 'total_liabilities', 'total_assets', 'first_credit_date', 'current_date', 'login_records', 'unusual_operations']

# 检查缺失值
missing_personal_fields = personal_df.columns[personal_df.isnull().any()].tolist()
missing_enterprise_fields = enterprise_df.columns[enterprise_df.isnull().any()].tolist()

if missing_personal_fields:
    print(f"Missing fields in personal data: {missing_personal_fields}")
    # 处理缺失值，例如填补或删除
    personal_df.fillna(personal_df.median(), inplace=True)

if missing_enterprise_fields:
    print(f"Missing fields in enterprise data: {missing_enterprise_fields}")
    # 处理缺失值，例如填补或删除
    enterprise_df.fillna(enterprise_df.median(), inplace=True)

# 检查异常值
# 这里可以添加具体的异常值检测和处理逻辑
# 示例：检查收入记录是否为非负数
if (personal_df['income_records'].apply(lambda x: any(i < 0 for i in x))).any():
    print("Found negative income records in personal data. Handling...")
    # 处理异常值，例如删除或修正
    personal_df = personal_df[personal_df['income_records'].apply(lambda x: all(i >= 0 for i in x))]

if (enterprise_df['monthly_income'] < 0).any() or (enterprise_df['monthly_expense'] < 0).any():
    print("Found negative income or expense records in enterprise data. Handling...")
    # 处理异常值，例如删除或修正
    enterprise_df = enterprise_df[(enterprise_df['monthly_income'] >= 0) & (enterprise_df['monthly_expense'] >= 0)]

# 预处理数据
def preprocess_data(df):
    """预处理数据"""
    df['first_credit_date'] = pd.to_datetime(df['first_credit_date'])
    df['current_date'] = pd.to_datetime(df['current_date'])
    df['income_records'] = df['income_records'].apply(eval)
    df['login_records'] = df['login_records'].apply(eval)
    df['unusual_operations'] = df['unusual_operations'].apply(eval)
    if 'monthly_income' in df.columns:
        df['monthly_income'] = df['monthly_income'].apply(eval)
    if 'monthly_expense' in df.columns:
        df['monthly_expense'] = df['monthly_expense'].apply(eval)
    return df

personal_df = preprocess_data(personal_df)
enterprise_df = preprocess_data(enterprise_df)

# 生成特征
personal_df = generate_base_features(personal_df)
personal_df = generate_behavior_features(personal_df)

enterprise_df = generate_base_features(enterprise_df)
enterprise_df = generate_behavior_features(enterprise_df)
enterprise_df = generate_enterprise_features(enterprise_df)

# 数据增强：生成更多数据集
def augment_data(df, n_samples=1000):
    """通过随机扰动现有数据生成新的样本"""
    augmented_df = df.sample(n=n_samples, replace=True, random_state=42)
    augmented_df['income_records'] = augmented_df['income_records'].apply(lambda x: [i + (i * 0.1 * (2 * pd.np.random.rand() - 1)) for i in x])
    augmented_df['total_debt'] = augmented_df['total_debt'] * (1 + 0.1 * (2 * pd.np.random.rand() - 1))
    augmented_df['total_income'] = augmented_df['total_income'] * (1 + 0.1 * (2 * pd.np.random.rand() - 1))
    augmented_df['monthly_income'] = augmented_df['monthly_income'].apply(lambda x: [i + (i * 0.1 * (2 * pd.np.random.rand() - 1)) for i in x])
    augmented_df['monthly_expense'] = augmented_df['monthly_expense'].apply(lambda x: [i + (i * 0.1 * (2 * pd.np.random.rand() - 1)) for i in x])
    augmented_df['total_liabilities'] = augmented_df['total_liabilities'] * (1 + 0.1 * (2 * pd.np.random.rand() - 1))
    augmented_df['total_assets'] = augmented_df['total_assets'] * (1 + 0.1 * (2 * pd.np.random.rand() - 1))
    return augmented_df

personal_augmented_df = augment_data(personal_df)
enterprise_augmented_df = augment_data(enterprise_df)

# 保存综合数据表
personal_df.to_csv('data/combined_personal_data.csv', index=False)
enterprise_df.to_csv('data/combined_enterprise_data.csv', index=False)

# 保存增强后的数据表
personal_augmented_df.to_csv('data/augmented_personal_data.csv', index=False)
enterprise_augmented_df.to_csv('data/augmented_enterprise_data.csv', index=False)