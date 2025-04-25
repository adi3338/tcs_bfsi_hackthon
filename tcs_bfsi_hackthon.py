import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Load default dataset
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    # Create binary target: 1 if Credit amount > 5000 else 0
    df['Class'] = df['Credit amount'].apply(lambda x: 1 if x > 5000 else 0)
    return df

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Dataset", "Model Training", "Prediction"])

# Upload or load default data
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded.")
else:
    df = load_data()
    st.sidebar.info("Using default German Credit dataset.")

# Preprocessing setup
# Define numeric columns for scaling
num_cols = ["Age", "Credit amount", "Duration"]

# Prepare features and target
X = df.drop(columns=['Class', 'Unnamed: 0'], errors='ignore')
y = df['Class']
# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and fit scaler on train's numeric features only
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

# Apply scaling to train and test numeric columns
X_train_scaled = X_train.copy()
X_train_scaled[num_cols] = scaler.transform(X_train[num_cols])

X_test_scaled = X_test.copy()
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# Train model on processed data
model = train_model(X_train_scaled, y_train)
# Evaluate on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# APP SECTIONS
if section == "Dataset":
    st.title("📊 Dataset Preview")
    st.dataframe(df.head(10))
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

elif section == "Model Training":
    st.title("🤖 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Top 10 Important Features")
    importances = model.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({ 'Feature': feature_names, 'Importance': importances })
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', ax=ax2)
    st.pyplot(fig2)

elif section == "Prediction":
    st.title("🔮 Predict Credit Risk")

    with st.form("custom_input_form"):
        st.markdown("#### Enter applicant details:")
        age = st.number_input("Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.median()))
        sex = st.selectbox("Sex", df["Sex"].unique())
        job = st.selectbox("Job", df["Job"].unique())
        housing = st.selectbox("Housing", df["Housing"].unique())
        saving = st.selectbox("Saving accounts", df["Saving accounts"].fillna("missing").unique())
        checking = st.selectbox("Checking account", df["Checking account"].fillna("missing").unique())
        credit = st.number_input(
            "Credit amount",
            float(df["Credit amount"].min()),
            float(df["Credit amount"].max()),
            float(df["Credit amount"].median())
        )
        duration = st.number_input(
            "Duration",
            int(df["Duration"].min()),
            int(df["Duration"].max()),
            int(df["Duration"].median())
        )
        purpose = st.selectbox("Purpose", df["Purpose"].unique())
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build input DataFrame
        input_dict = {
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving accounts": saving,
            "Checking account": checking,
            "Credit amount": credit,
            "Duration": duration,
            "Purpose": purpose
        }
        input_df = pd.DataFrame([input_dict])

        # Encode and align columns
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Scale only numeric columns
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0, 1]
        label = "🟢 Good Credit" if pred == 0 else "🔴 Bad Credit"

        st.subheader("Prediction Results")
        st.write(f"{label} (probability of bad credit: {prob:.2f})")
