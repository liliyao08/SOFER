import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import time  # 用于计时

# ===============================
# 0. 参数设置
# ===============================
RANDOM_SEED = 100         # 随机数种子
SAMPLE_NUMBER = -1        # 采用的数据数量 (-1表示使用全部数据)

# KNN参数
N_NEIGHBORS = 5           # 最近邻居的数量
WEIGHT = 'uniform'        # 权重函数，可选 'uniform' 或 'distanc

# 置换重要性参数
N_REPEATS = 10           # 置换次数
N_JOBS = -1               # 使用所有可用的CPU核心
PERMUTATION_SAMPLE_SIZE = 2000  # 仅使用前10,000个样本进行置换重要性计算

# ===============================
# 1. 数据预处理
# ===============================
def load_and_preprocess_data(filepath, sample_number=-1):
    """
    加载数据并进行预处理，包括特征值替换、归一化和数据划分。
    """
    print("Loading data...")
    data = pd.read_csv(filepath)
    feature_names=data.columns.tolist()

    if sample_number > 0:
        data = data.sample(n=sample_number, random_state=RANDOM_SEED)
    print(f"Sample number: {len(data)}")

    def replace_string_to_value(data_list):
        unique_values = data_list.unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        return data_list.replace(mapping)

    # 替换字符串为数值
    columns_to_replace = ['SW']
    for col in columns_to_replace:
        data[col] = replace_string_to_value(data[col])

    # 去除无效值
    data = data.dropna()

    # 提取特征和标签
    y = data.label
    X = data.drop('label', axis=1).drop('account', axis=1).drop('SW', axis=1)

    # 归一化处理
    print("Normalizing data...")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, feature_names

# ===============================
# 2. 模型训练与评估
# ===============================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
def train_model(X_train, X_test, y_train, y_test, feature_names, n_neighbors=5, weights='uniform'):
    """
    使用KNN训练模型，并进行预测和评估，包括计算和绘制置换重要性。
    """
    # 初始化KNN分类器
    print("Training model...")
    # model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    # model = KNeighborsClassifier(n_neighbors=1, weights=weights)
    # model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED)
    # model = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_SEED) #LASSO
    # model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # 在测试集上预测
    print("Evaluating model...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # 预测概率
    y_pred = model.predict(X_test)                   # 预测类别标签

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mse = mean_squared_error(y_test, y_pred_prob)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    # 打印评估指标
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R-squared: {r2:.3f}")

    # 混淆矩阵
    print("Confusion Matrix:")
    print(cm)

    # 绘制混淆矩阵热图
    plt.figure(figsize=(12, 10))

    # 绘制热图
    # plt.rcParams['font.family'] = 'Times New Roman'
    sns.heatmap(cm, annot=True, square=True, fmt='d', cmap='Blues',annot_kws={"size": 15})
                # xticklabels=feature_names, yticklabels=feature_names)

    plt.ylabel('Actual', fontsize=20)  # 设置 'Actual' 的字体大小为 14
    plt.xlabel('Predicted', fontsize=20)  # 设置 'Predicted' 的字体大小为 14
    # plt.title('Confusion Matrix')
    plt.show()


    # 置换重要性
    print("Calculating permutation features importances...")
    start_time = time.time()

    # 仅使用前10,000个样本进行置换重要性计算
    if len(X_test) > PERMUTATION_SAMPLE_SIZE:
        print(f"Using the first {PERMUTATION_SAMPLE_SIZE} samples from the test set for permutation importance.")
        X_test_subset = X_test.iloc[:PERMUTATION_SAMPLE_SIZE]
        y_test_subset = y_test.iloc[:PERMUTATION_SAMPLE_SIZE]
    else:
        print(f"Using all {len(X_test)} samples from the test set for permutation importance.")
        X_test_subset = X_test
        y_test_subset = y_test

    perm_importance = permutation_importance(
        model, X_test_subset, y_test_subset,
        n_repeats=N_REPEATS, 
        random_state=RANDOM_SEED, 
        scoring='f1_weighted', 
        n_jobs=N_JOBS
    )
    end_time = time.time()
    print(f"Permutation importance calculated in {end_time - start_time:.2f} seconds.")

    # 转换为DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # 打印特征重要性
    print("Feature Importances (Permutation Importance):")
    print(importance_df)

    # 绘制特征重要性柱状图
    plt.figure(figsize=(20, 12))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.xlabel('Importance', fontsize=30)
    plt.ylabel('Features', fontsize=30)
    # plt.title('Feature Importance via Permutation', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    plt.show()

    return model

# ===============================
# 3. 主程序
# ===============================
if __name__ == "__main__":

    # filepath = r"./BABD-13.csv"

    # 加载并预处理数据
    # X, y, feature_names = load_and_preprocess_data(filepath, SAMPLE_NUMBER)
    X = pd.read_csv(f'./features/X_148_0.9665.csv')
    y = pd.read_csv(f'./features/y_148_0.9665.csv')
    feature_names = np.load(f'./features/selected_features_148_0.9665.npy', allow_pickle=True)

    # 划分训练集和测试集，使用分层采样以保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    # 训练KNN模型并评估
    model = train_model(X_train, X_test, y_train, y_test,feature_names, n_neighbors=N_NEIGHBORS, weights=WEIGHT)