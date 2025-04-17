"""
风险模型训练全流程 - 增强版
功能：整合企业/个人数据，完成特征工程、XGBoost模型训练、评估及可视化
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from datetime import datetime
from matplotlib import font_manager

# 指定中文字体路径（根据实际路径修改）
import os
import platform
from matplotlib import font_manager

# 判断平台类型
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/msyh.ttc"
else:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # Linux下常用中文字体

# 加载字体
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    print(f"Loaded font from: {font_path}")
else:
    print(f"Font file not found: {font_path}")

# 注册字体
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

# ---- 路径配置 ----
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 添加Docker环境检测
import os
DOCKER_ENV = os.getenv('DOCKER_ENV', 'false').lower() == 'true'

# 修改路径配置
if DOCKER_ENV:
    # Docker容器内路径
    MODEL_SAVE_PATH = Path("/app/models/saved_models")
    FEATURE_IMPORTANCE_PATH = Path("/app/reports/feature_analysis")
    font_path = "/usr/share/fonts/truetype/msttcorefonts/msyh.ttc"  # Docker容器内字体路径
    data_dir = Path("/app/data/generated")
else:
    # 本地开发环境路径
    MODEL_SAVE_PATH = project_root / "models/saved_models"
    FEATURE_IMPORTANCE_PATH = project_root / "reports/feature_analysis"
    font_path = "C:/Windows/Fonts/Microsoft YaHei UI/msyh.ttc"  # Windows本地字体路径
    data_dir = project_root / "data/generated"

# 确保目录存在
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
FEATURE_IMPORTANCE_PATH.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# ---- 自定义模块导入 ----
from features.base_features import generate_base_features
from features.behavior_features import generate_behavior_features
from features.enterprise_features import generate_enterprise_features

# ---- 常量定义 ----
RANDOM_STATE = 42
DATE_FORMAT = "%Y-%m-%d"
MODEL_SAVE_PATH = project_root / "models/saved_models"
FEATURE_IMPORTANCE_PATH = project_root / "reports/feature_analysis"


#加载并验证原始数据
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 使用修改后的data_dir路径
    personal_path = data_dir / "personal_data.csv"
    enterprise_path = data_dir / "enterprise_data.csv"

    # 增强路径验证
    missing_files = []
    for path in [personal_path, enterprise_path]:
        if not path.exists():
            missing_files.append(path.name)
    if missing_files:
        raise FileNotFoundError(f"缺失数据文件: {', '.join(missing_files)}")

    return (
        pd.read_csv(personal_path, parse_dates=['first_credit_date', 'current_date']),
        pd.read_csv(
            enterprise_path,
            parse_dates=['establishment_date', 'last_financial_report_date'],
            converters={
                'industry_type': lambda x: x.strip()  # 清理行业类型字段
            }
        )
    )


# ---- 企业数据预处理增强 ----
def advanced_preprocessing(df: pd.DataFrame, is_enterprise: bool = False) -> pd.DataFrame:
    df = df.copy()
    """
    增强数据预处理流程
    修改点：适配行业类型字段处理
    """
    # 公共处理逻辑
    list_columns = ['income_records', 'login_records', 'unusual_operations']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # 企业专用处理
    if is_enterprise:
        # 新增行业类型处理
        if 'industry_type' in df.columns:
            df['industry_type'] = df['industry_type'].astype('category')

        # 财务字段处理
        financial_cols = [
            'monthly_income', 'monthly_expense',
            'total_liabilities', 'total_assets'
        ]
        for col in financial_cols:
            if col in df.columns:
                # 添加对数转换处理大规模数值
                df[col] = np.log1p(pd.to_numeric(df[col], errors='coerce'))
                df = df[df[col].between(df[col].quantile(0.01), df[col].quantile(0.99))]

    # 缺失值处理策略
    impute_strategy = {
        'numeric': 'median',
        'category': 'mode',
        'datetime': datetime.now().strftime(DATE_FORMAT)
    }

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median() if impute_strategy['numeric'] == 'median' else df[col].mean()
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                fill_value = pd.to_datetime(impute_strategy['datetime'])
            else:
                fill_value = df[col].mode()[0] if impute_strategy['category'] == 'mode' else 'unknown'
            df[col].fillna(fill_value, inplace=True)

    # 异常值过滤（保留95%分位数）
    if is_enterprise:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            lower = df[col].quantile(0.025)
            upper = df[col].quantile(0.975)
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df.reset_index(drop=True)


def generate_enhanced_features(df: pd.DataFrame, is_enterprise: bool) -> pd.DataFrame:
    """安全处理混合类型数据的特征工程"""
    # 1. 生成原始特征
    df = generate_base_features(df)
    df = generate_behavior_features(df)
    if is_enterprise:
        df = generate_enterprise_features(df)

    # 2. 分离数据类型
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    # 3. 处理分类变量（示例使用标签编码）
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 4. 过滤低方差数值特征
    if numeric_cols:
        selector = VarianceThreshold(threshold=0.01)
        df_numeric = pd.DataFrame(
            selector.fit_transform(df[numeric_cols]),
            columns=np.array(numeric_cols)[selector.get_support()]
        )
        df = pd.concat([df_numeric, df[categorical_cols]], axis=1)

    return df

def visualize_class_balance(y: pd.Series, title: str) -> None:
    """类别分布可视化"""
    plt.figure(figsize=(8, 5))
    y.value_counts(normalize=True).plot(kind='bar')
    plt.title(f"Class Distribution - {title}")
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.show()


def train_risk_model(X: pd.DataFrame, y: pd.Series, model_name: str) -> xgb.XGBClassifier:
    """
    增强版模型训练流程
    包含：样本平衡、交叉验证、超参搜索、早停机制、多维度评估
    """
    # 样本平衡处理
    if y.nunique() > 1:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        X_res, y_res = X.copy(), y.copy()

    # 权重计算
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_res)

    # 数据分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=RANDOM_STATE
    )

    # XGBoost参数配置
    param_grid = {
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 1]
    }

    # 带早停的交叉验证
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=1000,
        early_stopping_rounds=50,
        random_state=RANDOM_STATE
    )

    # 网格搜索
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_model = grid_search.best_estimator_

    # 模型评估
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"\n=== Best Model Evaluation ({model_name}) ===")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"AUC: {roc_auc_score(y_val, y_proba):.4f}")
    print(f"KS: {max(roc_curve(y_val, y_proba)[1] - roc_curve(y_val, y_proba)[0]):.4f}")
    print(classification_report(y_val, y_pred))

    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax[0])
    ax[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ax[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_val, y_proba):.2f}')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].set_title("ROC Curve")
    ax[1].legend(loc="lower right")
    plt.show()

    # 特征分析（修改点：传入验证集数据X_val）
    analyze_feature_importance(best_model, X_val, model_name)

    # 模型保存
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE_PATH / f"{model_name}_model.pkl")
    print(f"Model saved to {MODEL_SAVE_PATH / model_name}_model.pkl")

    import gc
    plt.close('all')
    gc.collect()  # 强制垃圾回收

    return best_model


def analyze_feature_importance(model, X_data: pd.DataFrame, model_name: str) -> None:
    """多维特征重要性分析（支持中文）"""
    # 1. 配置中文字体（需提前安装）
    try:
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/msyh.ttc"  # Windows字体路径
        else:
            font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"  # Linux字体路径

        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
        else:
            raise FileNotFoundError(f"Font file not found: {font_path}")
    except Exception as e:
        plt.rcParams['font.family'] = 'Arial'  # 回退到英文字体
        print(f"警告：中文字体加载失败，已切换为英文显示。错误信息: {e}")

    # 2. 创建输出目录
    FEATURE_IMPORTANCE_PATH.mkdir(parents=True, exist_ok=True)
    # 3. 内置特征重要性分析
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(X_data.columns[sorted_idx][:15], importance[sorted_idx][:15])
    plt.title("特征重要性排名 (XGBoost)")
    plt.savefig(FEATURE_IMPORTANCE_PATH / f"{model_name}_feature_importance.png", bbox_inches='tight', dpi=300)
    plt.close()

    # 4. SHAP分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    plt.figure()
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.title("SHAP特征贡献度")
    plt.savefig(FEATURE_IMPORTANCE_PATH / f"{model_name}_shap_summary.png", bbox_inches='tight', dpi=300)
    plt.close()
def main():
    # 数据加载与预处理
    personal_df, enterprise_df = load_data()

    print("正在进行个人数据预处理...")
    personal_df = advanced_preprocessing(personal_df)
    print("正在进行企业数据预处理...")
    enterprise_df = advanced_preprocessing(enterprise_df, is_enterprise=True)

    # 特征工程
    print("\n生成个人特征...")
    personal_df = generate_enhanced_features(personal_df, is_enterprise=False)
    print("生成企业特征...")
    enterprise_df = generate_enhanced_features(enterprise_df, is_enterprise=True)

    # 检查数据质量
    visualize_class_balance(personal_df['credit_grade'], "Personal Credit Grade")
    visualize_class_balance(enterprise_df['risk_level'], "Enterprise Risk Level")

    # 模型训练（修改点：确保传入完整的特征数据）
    print("\n训练个人信用模型...")
    personal_model = train_risk_model(
        personal_df.drop(columns=['credit_grade']),
        personal_df['credit_grade'],
        "personal_credit"
    )

    print("\n训练企业风险模型...")
    enterprise_model = train_risk_model(
        enterprise_df.drop(columns=['risk_level']),
        enterprise_df['risk_level'],
        "enterprise_risk"
    )

    return personal_model, enterprise_model  # 返回模型对象供后续使用


if __name__ == "__main__":
    personal_model, enterprise_model = main()