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
import time  # For timing

RANDOM_SEED = 100         # Random seed for reproducibility
SAMPLE_NUMBER = -1        # Number of samples to use (-1 means use all data)

# KNN Parameters
N_NEIGHBORS = 5           # Number of nearest neighbors
WEIGHT = 'uniform'        # Weight function, options are 'uniform' or 'distance'

# Permutation Importance Parameters
N_REPEATS = 10           # Number of repetitions for permutation importance
N_JOBS = -1               # Use all available CPU cores
PERMUTATION_SAMPLE_SIZE = 2000  # Use only the first 2,000 samples for permutation importance calculation

def load_and_preprocess_data(filepath, sample_number=-1):
    """
    Load data and preprocess, including feature replacement, normalization, and data splitting.
    """
    print("Loading data...")
    data = pd.read_csv(filepath)
    feature_names = data.columns.tolist()

    if sample_number > 0:
        data = data.sample(n=sample_number, random_state=RANDOM_SEED)
    print(f"Sample number: {len(data)}")

    def replace_string_to_value(data_list):
        unique_values = data_list.unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        return data_list.replace(mapping)

    # Replace strings with numeric values
    columns_to_replace = ['SW']
    for col in columns_to_replace:
        data[col] = replace_string_to_value(data[col])

    # Remove rows with missing values
    data = data.dropna()

    # Extract features and labels
    y = data.label
    X = data.drop('label', axis=1).drop('account', axis=1).drop('SW', axis=1)

    # Normalize the data
    print("Normalizing data...")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, feature_names

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
def train_model(X_train, X_test, y_train, y_test, feature_names, n_neighbors=5, weights='uniform'):
    """
    Train the model using KNN and evaluate, including calculating and plotting permutation feature importances.
    """
    # Initialize the KNN classifier
    print("Training model...")
    # model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    # model = KNeighborsClassifier(n_neighbors=1, weights=weights)
    # model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED)
    # model = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_SEED) #LASSO
    # model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # Predict on the test set
    print("Evaluating model...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Predict probabilities
    y_pred = model.predict(X_test)                   # Predict class labels

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mse = mean_squared_error(y_test, y_pred_prob)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R-squared: {r2:.3f}")

    # Confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(12, 10))

    # Plot the heatmap
    # plt.rcParams['font.family'] = 'Times New Roman'
    sns.heatmap(cm, annot=True, square=True, fmt='d', cmap='Blues', annot_kws={"size": 15})
                # xticklabels=feature_names, yticklabels=feature_names)

    plt.ylabel('Actual', fontsize=20)  # Set 'Actual' label font size to 14
    plt.xlabel('Predicted', fontsize=20)  # Set 'Predicted' label font size to 14
    # plt.title('Confusion Matrix')
    plt.show()


    # Permutation importance
    print("Calculating permutation feature importances...")
    start_time = time.time()

    # Use only the first 2,000 samples for permutation importance calculation if the test set is large
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

    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Print feature importance
    print("Feature Importances (Permutation Importance):")
    print(importance_df)

    # Plot feature importance bar chart
    plt.figure(figsize=(20, 12))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.xlabel('Importance', fontsize=30)
    plt.ylabel('Features', fontsize=30)
    # plt.title('Feature Importance via Permutation', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":

    # filepath = r"./BABD-13.csv"

    # Load and preprocess data
    # X, y, feature_names = load_and_preprocess_data(filepath, SAMPLE_NUMBER)
    X = pd.read_csv(f'./features/X_148_0.9665.csv')
    y = pd.read_csv(f'./features/y_148_0.9665.csv')
    feature_names = np.load(f'./features/selected_features_148_0.9665.npy', allow_pickle=True)

    # Split into training and testing sets, using stratified sampling to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
    )

    # Train KNN model and evaluate
    model = train_model(X_train, X_test, y_train, y_test, feature_names, n_neighbors=N_NEIGHBORS, weights=WEIGHT)
