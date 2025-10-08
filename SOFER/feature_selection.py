from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

data = pd.read_csv('BABD-13.csv')


y = data.label
X = data.drop(['label', 'account', 'SW'], axis=1)

xgb_model = XGBClassifier(eval_metric='logloss', random_state=100)

# 初始化递归特征消除
n_features = X.shape[1]
accuracy_list = []

pruned_features_df = pd.DataFrame()

print("Starting Recursive Feature Elimination...")
ref_X,ref_y=X,y

# ref_X = pd.read_csv(f'./features/X_34_0.9647.csv')
# ref_y = pd.read_csv(f'./features/y_34_0.9647.csv')
# selected_features = np.load(f'./features/selected_features_34_0.9647.npy', allow_pickle=True)
# n_features = len(selected_features)

print(f'Number of features: {n_features}')

# 递归特征消除
while n_features > 0:

    rfe = RFE(estimator=xgb_model, n_features_to_select=n_features, step=1)
    rfe.fit(ref_X, ref_y)

    selected_features = ref_X.columns[rfe.support_]
    ref_X = ref_X[selected_features]
    print(f'Number of features selected: {len(selected_features)}')
    print(f"\t {selected_features}")

    X_train, X_test, y_train, y_test = train_test_split(ref_X, ref_y, test_size=0.2, stratify=ref_y, random_state=100)

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    xgb_model.fit(X_train_selected, y_train)
    accuracy = xgb_model.score(X_test_selected, y_test)
    accuracy_list.append((n_features, accuracy))

    print(f"Number of features: {n_features}, Accuracy: {accuracy:.4f}")

    X_train = X_train_selected
    X_test = X_test_selected

    ref_X.to_csv(f'./features/X_{n_features}_{accuracy:.4f}.csv', index=False)
    ref_y.to_csv(f'./features/y_{n_features}_{accuracy:.4f}.csv', index=False)
    np.save(f'./features/selected_features_{n_features}_{accuracy:.4f}.npy', selected_features)

    ref_X = pd.read_csv(f'features/X_{n_features}_{accuracy:.4f}.csv')
    ref_y = pd.read_csv(f'features/y_{n_features}_{accuracy:.4f}.csv')
    selected_features = np.load(f'./features/selected_features_{n_features}_{accuracy:.4f}.npy', allow_pickle=True)

    plt.figure()
    plt.plot([x[0] for x in accuracy_list], [x[1] for x in accuracy_list], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Feature Selection vs Accuracy')
    plt.grid(True)
    plt.savefig(f'./features/accuracy_plot.png')  # 保存图表
    plt.close()

    n_features-=1
    np.save(f'./features/accuracy_list.npy', accuracy_list)


