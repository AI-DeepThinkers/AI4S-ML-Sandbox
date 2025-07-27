import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLEANED_DIR = os.path.join(ROOT, "data/cleaned")

# Streamlit UI
st.set_page_config(page_title="🧠 Level 2 – Task 2: Classification", layout="wide")
st.title("🧠 Task 2: Classification")

# List available cleaned CSVs
csv_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith(".csv")]
if not csv_files:
    st.error("No cleaned CSV files found.")
    st.stop()

dataset = st.selectbox("📂 Select dataset", csv_files)
df = pd.read_csv(os.path.join(CLEANED_DIR, dataset))
st.write("Preview:", df.head())

# Check target type
if df['MEDV'].dtype in ['float64','int64']:
    st.info("Binning `MEDV` into price categories for classification.")
    df['target'] = pd.cut(df['MEDV'], bins=[0, 20, 35, df['MEDV'].max()],
                          labels=['Low', 'Medium', 'High'])
    df.drop(columns=['MEDV'], inplace=True)
    st.write("Targets:", df['target'].value_counts())
else:
    st.error("Dataset does not contain numeric MEDV column.")
    st.stop()

# Split and run classification
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

st.subheader("📊 Model Evaluation")
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    st.markdown(f"**{name}** – Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    cm = confusion_matrix(y_test, preds, labels=mdl.classes_)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                xticklabels=mdl.classes_, yticklabels=mdl.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} – Confusion Matrix")
    st.pyplot(fig)
