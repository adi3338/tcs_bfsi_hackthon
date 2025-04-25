import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Load dataset
data_path = "german_credit_data.csv"
credit_data = pd.read_csv(data_path)

# Drop unnecessary column
credit_data.drop(columns=['Unnamed: 0'], inplace=True)

# Handle missing values in categorical columns
credit_data.fillna({'Saving accounts': 'unknown', 'Checking account': 'unknown'}, inplace=True)

# Encode categorical features
encoders = {}
for category in credit_data.select_dtypes(include='object').columns:
    encoder = LabelEncoder()
    credit_data[category] = encoder.fit_transform(credit_data[category])
    encoders[category] = encoder

# Normalize numerical features
features_to_scale = ['Age', 'Credit amount', 'Duration']
scaler_tool = StandardScaler()
credit_data[features_to_scale] = scaler_tool.fit_transform(credit_data[features_to_scale])

# Define target variable based on thresholds
credit_limit = credit_data['Credit amount'].median()
time_limit = credit_data['Duration'].median()
credit_data['Target'] = ((credit_data['Credit amount'] > credit_limit) & 
                         (credit_data['Duration'] < time_limit)).astype(int)



# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(credit_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Prepare data for training
features = credit_data.drop(columns=['Target'])
target = credit_data['Target']
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=42
)

# Hyperparameter grid for tuning
tune_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

# Initialize and perform grid search
base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search_model = GridSearchCV(estimator=base_model, param_grid=tune_grid, 
                                 scoring='f1', cv=3, verbose=1, n_jobs=-1)
grid_search_model.fit(X_train_set, y_train_set)

# Evaluation
final_model = grid_search_model.best_estimator_
predicted_labels = final_model.predict(X_test_set)

performance_report = classification_report(y_test_set, predicted_labels, output_dict=True)
conf_matrix = confusion_matrix(y_test_set, predicted_labels)
optimal_params = grid_search_model.best_params_

print(performance_report, conf_matrix, optimal_params)


# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


best   = grid_search_model.best_estimator_    
scaler = scaler_tool 


# 11. Save artifacts
os.makedirs('model_artifacts', exist_ok=True)
joblib.dump(best,      'model_artifacts/best_xgb_model.pkl')
joblib.dump(encoders,  'model_artifacts/encoders.pkl')
joblib.dump(scaler,    'model_artifacts/scaler.pkl')

