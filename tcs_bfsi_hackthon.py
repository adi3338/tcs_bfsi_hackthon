import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to load and preprocess data
def load_data():
    df = pd.read_csv("german_credit_data.csv")  # Load the unzipped CSV file
    df['Class'] = df['Credit amount'].apply(lambda x: 1 if x > 5000 else 0)  # Create the target variable
    return df

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit Interface
st.title("Credit Risk Prediction App")
st.write("This app predicts credit risk based on the German Credit dataset.")

# Upload file for custom use
uploaded_file = st.file_uploader("Upload the dataset (optional)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")
else:
    df = load_data()
    st.write("Using default dataset: German Credit Data.")

# Display first few rows of the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Select features and target
X = df.drop(columns=['Class', 'Unnamed: 0'])
y = df['Class']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = train_model(X_train, y_train)
st.write("✅ Model Trained Successfully!")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Feature importance visualization
st.subheader("Feature Importance")
importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_df.head(10), x="Importance", y="Feature", ax=ax2)
st.pyplot(fig2)

# Prediction on a custom input
st.subheader("Predict Credit Risk for Custom Input")
sample_input = []
for feature in X.columns:
    sample_input.append(st.number_input(f"Enter value for {feature}", value=0.0))
sample_input = np.array(sample_input).reshape(1, -1)

if st.button('Predict'):
    prediction = model.predict(sample_input)
    result = "Good Credit" if prediction[0] == 0 else "Bad Credit"
    st.write(f"🔮 Prediction: {result}")
