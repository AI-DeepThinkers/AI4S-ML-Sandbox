import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CLEANED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cleaned"))

def list_cleaned_csvs():
    return [f for f in os.listdir(CLEANED_DATA_PATH) if f.endswith(".csv")]

def load_csv(filepath):
    return pd.read_csv(filepath)

def is_classification_target(y):
    if y.dtype == 'O':
        return True
    if y.nunique() <= 20 and np.all(np.equal(np.mod(y, 1), 0)):
        return True
    return False

def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

def train_and_evaluate(df, target_col, feature_cols):
    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column.")
        return

    X = df[feature_cols]
    y = df[target_col]

    # Check if classification is valid
    if not is_classification_target(y):
        st.error("🚫 Target column is not suitable for classification (too many unique or continuous values).")
        return

    # Filter non-numeric features
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        st.error(f"❌ Non-numeric feature(s) selected: {', '.join(non_numeric_cols)}.\nPlease select only numeric columns.")
        return

    # Encode if needed
    if y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_
    else:
        class_names = sorted(y.unique())

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.subheader(f"{name} - Accuracy: {acc:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, class_names, f"{name} Confusion Matrix")

            st.markdown("**Classification Report**")
            st.dataframe(pd.DataFrame(
                classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            ).T)

    except Exception as e:
        st.exception(f"🛑 Model training failed: {e}")


def main():
    st.title("🔍 Level 2 – Task 2: Classification")

    st.markdown("### Step 1: Choose dataset from cleaned files")
    cleaned_csvs = sorted(list_cleaned_csvs())
    iris_default_index = next((i for i, name in enumerate(cleaned_csvs) if "iris" in name.lower()), 0)


    if not cleaned_csvs:
        st.warning("No cleaned datasets found in /data/cleaned/. Please upload one below.")
    else:
        selected_file = st.selectbox("Available Cleaned Datasets", cleaned_csvs, index=iris_default_index)
        selected_path = os.path.join(CLEANED_DATA_PATH, selected_file)
        df = load_csv(selected_path)
        st.write("✅ Loaded Dataset Preview", df.head())

        st.markdown("### Step 2: Configure Classification")
        target_col = st.selectbox("Select Target Column", options=df.columns)
        features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])
        train_and_evaluate(df, target_col, features)

    st.divider()

    st.markdown("### Optional: Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Upload CSV file for classification", type=["csv"])
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.success("📄 File uploaded successfully!")
        st.write(df_upload.head())

        target_col = st.selectbox("Select Target Column (Uploaded)", options=df_upload.columns, key="upload_target")
        features = st.multiselect("Select Feature Columns (Uploaded)", [col for col in df_upload.columns if col != target_col], key="upload_features")
        train_and_evaluate(df_upload, target_col, features)

if __name__ == "__main__":
    main()
