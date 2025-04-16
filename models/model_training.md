# 模型训练说明

## 1. 数据加载
- 使用 `load_data` 函数加载 CSV 文件中的数据。

## 2. 数据预处理
- 将日期列转换为 `datetime` 类型。
- 将字符串形式的列表转换为实际的 Python 列表。

## 3. 特征生成
- 使用 `generate_base_features` 生成基础特征。
- 使用 `generate_behavior_features` 生成行为序列特征。

## 4. 模型训练
- 使用 `RandomForestClassifier` 训练模型。
- 评估模型的准确率并输出结果。

## 示例代码
```python
# 加载个人数据
personal_df = load_data('data/personal_data.csv')
personal_df = preprocess_data(personal_df)
personal_df = generate_features(personal_df)

# 假设目标列是 'credit_grade'（信用等级）
personal_model = train_model(personal_df, 'credit_grade')

# 加载企业数据
enterprise_df = load_data('data/enterprise_data.csv')
enterprise_df = preprocess_data(enterprise_df)
enterprise_df = generate_features(enterprise_df)

# 假设目标列是 'risk_level'（风险等级）
enterprise_model = train_model(enterprise_df, 'risk_level')