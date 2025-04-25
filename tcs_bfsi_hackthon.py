import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Streamlit App
st.set_page_config(page_title="German Credit Risk Analysis", layout="wide")
st.title("German Credit Risk Analysis")

# Sidebar for user inputs
st.sidebar.header("Data and Model Settings")
raw_url = st.sidebar.text_input("german_credit_data.csv"
)

test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)
n_estimators = st.sidebar.selectbox("Number of Trees (n_estimators)", [50, 100, 200, 300], index=1)
max_depth = st.sidebar.selectbox("Max Tree Depth", [3, 5, 7, 9], index=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=2)
cv_folds = st.sidebar.selectbox("CV Folds", [3, 5, 7], index=0)
run_button = st.sidebar.button("Train Model")

# Cache data loading\@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Drop unnecessary column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

# Main pipeline
def preprocess(df):
    # Handle missing categorical
    df.fillna({'Saving accounts': 'unknown', 'Checking account': 'unknown'}, inplace=True)
    # Encode categorical
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    # Scale numerical
    scaler = StandardScaler()
    df[['Age', 'Credit amount', 'Duration']] = scaler.fit_transform(df[['Age', 'Credit amount', 'Duration']])
    # Define target
    credit_limit = df['Credit amount'].median()
    time_limit = df['Duration'].median()
    df['Target'] = ((df['Credit amount'] > credit_limit) & (df['Duration'] < time_limit)).astype(int)
    return df

# Plot functions
def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.subheader("Feature Correlation Heatmap")
    st.pyplot(fig)

def plot_confusion(cm):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low Risk', 'High Risk'])
    disp.plot(cmap='YlGnBu', ax=ax)
    st.subheader("Confusion Matrix")
    st.pyplot(fig)

# Execute
if raw_url:
    try:
        data = load_data(raw_url)
        st.write("### Raw Data Preview", data.head())
        plot_correlation(data)

        if run_button:
            df = preprocess(data.copy())
            X = df.drop(columns=['Target'])
            y = df['Target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, stratify=y, random_state=42
            )
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
            st.write(f"## Training XGBoost with params: {params}")
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **params)
            grid = GridSearchCV(model, {
                'n_estimators': [n_estimators],
                'max_depth': [max_depth],
                'learning_rate': [learning_rate]
            }, scoring='f1', cv=cv_folds, n_jobs=-1)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            y_pred = best.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            st.subheader("Classification Report")
            st.table(pd.DataFrame(report).T)
            plot_confusion(cm)
            st.write(f"### Best CV Score: {grid.best_score_:.4f}")
            st.write(f"### Best Params: {grid.best_params_}")

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
else:
    st.info("Please provide the raw GitHub CSV URL.")
