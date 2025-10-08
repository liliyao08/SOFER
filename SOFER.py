import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    mean_squared_error, precision_recall_curve, average_precision_score,
    confusion_matrix, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from kan import KAN
from torch.utils.data import DataLoader, TensorDataset
import shap
from xgboost import XGBClassifier
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

RANDOM_SEED = 100
SAMPLE_NUMBER = -1

EPISODES = 1000
LEARNING_RATE = 0.005
GRID = 5
K = 1
N_MODELS = 1


def load_and_preprocess_data(filepath, sample_number=-1):
    print("Loading data...")
    # data = pd.read_csv(filepath)
    #
    # if sample_number > 0:
    #     data = data.sample(n=sample_number, random_state=RANDOM_SEED)
    # print(f"Sample number: {len(data)}")
    #
    # def replace_string_to_value(data_list):
    #     unique_values = data_list.unique()
    #     mapping = {val: idx for idx, val in enumerate(unique_values)}
    #     return data_list.replace(mapping)
    #
    # # 替换字符串为数值
    # columns_to_replace = ['SW']
    # for col in columns_to_replace:
    #     data[col] = replace_string_to_value(data[col])
    #
    # # 去除无效值
    # data = data.dropna()
    #
    # # 提取特征和标签
    # y = data.label
    # X = data.drop('label', axis=1).drop('account', axis=1).drop('SW', axis=1)
    #
    # # 归一化处理
    # print("Normalizing data...")
    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X = pd.read_csv(f'./features/X_1_0.7086.csv.csv')
    y = pd.read_csv(f'./features/y_1_0.7086.csv.csv')
    X = X[0:50000]
    y = y[0:50000]
    feature_names = np.load(f'./features/selected_features_1_0.7086.npy', allow_pickle=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=RANDOM_SEED)
    model = XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED).fit(X_train, y_train)

    # 数据扩充
    probs=model.predict_proba(X_test)
    pseudo_labels=np.argmax(probs, axis=1)
    confidence=np.max(probs, axis=1)
    high_confidence_indicies=confidence>=0.8
    pseudo_X=X_test[high_confidence_indicies]
    pseudo_y=pseudo_labels[high_confidence_indicies]
    expanded_train_X=pd.concat([X_train, pseudo_X], axis=0)
    print(y_train.shape)
    print(pseudo_y.shape)
    pseudo_y_df = pd.DataFrame(pseudo_y, index=pseudo_X.index, columns=y_train.columns)
    expanded_train_y = pd.concat([y_train, pseudo_y_df], axis=0)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_train, y_train=expanded_train_X, expanded_train_y
    # xgb_model_expanded = XGBClassifier(eval_metric='logloss', random_state=RANDOM_SEED)
    # xgb_model_expanded.fit(expanded_train_X, expanded_train_y)
    # initial_pred = xgb_model_expanded.predict(X_test)
    # initiaL_acc = accuracy_score(y_test, initial_pred)
    # print(f"Expanded Accuracy: {initiaL_acc}")

    return X_train, X_test, y_train, y_test


# ===============================
# 2. 模型训练
# ===============================
def train_kan_model(X_train, X_test, y_train, y_test, device, episodes=EPISODES, learning_rate=LEARNING_RATE,
                    validation_split=0.2, n_models=N_MODELS):
    # 存储多个 KAN 模型
    models = []
    # 将数据转换为 tensor
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long).to(device)

    # 用于存储精度
    train_accuracies = []
    test_accuracies = []

    # KAN可视化
    def visualze_kan(X_train, X_test, y_train, y_test):
        dataset = {}
        dataset['train_input'] = torch.from_numpy(X_train.to_numpy()).type(torch.float32).to(device)
        dataset['test_input'] = torch.from_numpy(X_test.to_numpy()).type(torch.float32).to(device)
        dataset['train_label'] = torch.from_numpy(y_train.to_numpy()).type(torch.long).to(device)
        dataset['test_label'] = torch.from_numpy(y_test.to_numpy()).type(torch.long).to(device)

        model = KAN(width=[[X_train.shape[1], 0], [5, 0], [13, 0]], auto_save=True, grid=GRID, k=K,
                    device=device)

        def train_acc():
            return torch.mean(
                (torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(torch.float32))

        def test_acc():
            return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(torch.float32))

        results = model.fit(dataset, opt="Adam", steps=200, lamb=0.001, lamb_entropy=1., metrics=(train_acc, test_acc),
                            loss_fn=torch.nn.CrossEntropyLoss(), save_fig=True, beta=10,img_folder='./figures')
    visualze_kan(X_train, X_test, y_train, y_test)

    for m in range(n_models):
        model = KAN(width=[[X_train.shape[1], 0], [5, 0], [14, 0]], auto_save=True, grid=GRID, k=K,
                    device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        models.append(model)


        # 创建 TensorDataset 并使用 DataLoader 加载数据
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = int(len(X_train)/10+1) # 设置批量大小，根据显存情况调整
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        loss_function = torch.nn.CrossEntropyLoss()

    for step in range(episodes):
        for model in models:
            i=0
            for batch_X, batch_y in train_loader:
                model.train()
                optimizer.zero_grad()
                predictions = model(batch_X)
                batch_y = batch_y.squeeze()
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                print(f"\tEpoch [{step}/{episodes}], Step {(i+1)*batch_size/len(X_train)*100:.0f}%, Loss: {loss:.4f}",end='\r\n')
                i+=1


        y_pred_classes = voting_predict(models, X_test_tensor).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        test_accuracy = accuracy_score(y_true, y_pred_classes)


        test_accuracies.append(test_accuracy)


        if step % 1 == 0:
            print(f"Step {step + 1}/{episodes}, Loss: {loss.item():.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}")

    return models, X_test_tensor, y_test_tensor


def voting_predict(models, X_data):

    predictions_list = [torch.argmax(model(X_data), dim=1) for model in models]

    stacked_tensor = torch.stack(predictions_list)

    voted_predictions, _ = torch.mode(stacked_tensor, dim=0)

    return voted_predictions


def calculate_shap_values(models, X_test, feature_names, device):
    print("Calculating features importances using SHAP...")

    X_test_np = X_test.values

    def ensemble_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred_prob = []
        for model in models:
            model.eval()
            with torch.no_grad():
                preds = model(X_tensor)
                prob = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()  # 获取正类的预测概率
                y_pred_prob.append(prob)
        return np.mean(y_pred_prob, axis=0)

    background_size = 200

    if X_test_np.shape[0] < background_size:
        background = X_test_np
    else:
        background = X_test_np[np.random.choice(X_test_np.shape[0], background_size, replace=False)]

    sample_size = X_test_np.shape[0]

    explainer = shap.KernelExplainer(ensemble_predict, background, link="logit")

    shap_values = explainer.shap_values(X_test_np, nsamples=100)

    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.xlabel('Importance', fontsize=35)
    plt.ylabel('Features', fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    plt.show()


def evaluate_model(models, test_input, test_label, X_test, device):
    feature_names = np.load(f'./features/selected_features_5_0.9121.npy', allow_pickle=True)

    print("Evaluating model...")

    # 使用集成投票预测
    y_pred_classes = voting_predict(models, test_input).cpu().numpy()
    y_true = test_label.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)
    r2 = r2_score(y_true, y_pred_classes)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"R²: {r2:.4f}")


    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))


    # plt.rcParams['font.family'] = 'Times New Roman'
    sns.heatmap(cm, annot=True, square=True, fmt='d', cmap='Blues',annot_kws={"size": 15})
                # xticklabels=feature_names, yticklabels=feature_names)

    plt.ylabel('Actual', fontsize=20)  # 设置 'Actual' 的字体大小为 14
    plt.xlabel('Predicted', fontsize=20)  # 设置 'Predicted' 的字体大小为 14
    # plt.title('Confusion Matrix')
    plt.show()


    y_pred_prob = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = model(test_input)
            prob = torch.softmax(preds, dim=1)[:, 1].cpu().numpy()
            y_pred_prob.append(prob)

    y_pred_prob_avg = np.mean(y_pred_prob, axis=0)


    mse = mean_squared_error(y_true, y_pred_prob_avg)
    rmse = np.sqrt(mse)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # # 计算并绘制特征重要性（使用 SHAP）
    calculate_shap_values(models, X_test, feature_names, device)


if __name__ == "__main__":

    filepath = r"./BABD-13.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available ",torch.cuda.is_available())

    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath, SAMPLE_NUMBER)

    kan_dataset = {
        'train_input': X_train,
        'train_label': y_train,
        'test_input': X_test,
        'test_label': y_test
    }

    models, test_input, test_label = train_kan_model(X_train, X_test, y_train, y_test, device)

    evaluate_model(models, test_input, test_label, X_test, device)

    lib = ['x']
    # lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
    models[0].auto_symbolic(lib=lib)
    formulas_vanilla = models[0].symbolic_formula()[0]
    for i in range(len(formulas_vanilla)):
        print(f"class {i}: {formulas_vanilla[i]}")