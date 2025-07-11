import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="CSV ML Trainer", layout="wide")

st.title("🤖 CSV Machine Learning App")
st.markdown("Upload a CSV file, select your target column, and train a machine learning model with one click!")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("🎯 Select Target Column")
        target = st.selectbox("Choose your target (label) column", options=df.columns)

        if st.button("Train Model"):
            df_clean = df.dropna()

            X = df_clean.drop(columns=[target])
            y = df_clean[target]

            # One-hot encode categorical features
            X = pd.get_dummies(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Report
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            st.subheader("📊 Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.success("✅ Model trained and evaluated successfully!")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
