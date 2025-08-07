# streamlit_app/app_pages/level3_task2_nlp.py

import os
import streamlit as st
import pandas as pd

from Level3_Advanced.Task2_NLP.sentiment_classification import (
    load_dataset,
    train_model,
    save_model_and_vectorizer,
    load_model_and_vectorizer,
    preprocess,
    simplify_labels
)

# === 📁 Paths ===
RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/sentiment_dataset.csv"))
CLEANED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cleaned/sentiment_dataset_clean.csv"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/sentiment_model.pkl"))
VECTORIZER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/tfidf_vectorizer.pkl"))


def main():
    st.title("🧠 NLP Text Classification - Sentiment Analysis")

    # === 📊 Dataset ===
    st.header("1️⃣ Dataset Selection")

    dataset_source = st.radio("Choose dataset source:", ["Default (preloaded)", "Upload CSV"])

    if dataset_source == "Default (preloaded)":
        try:
            df = load_dataset(RAW_DATA_PATH)
            st.success("✅ Default dataset loaded.")
        except FileNotFoundError:
            st.error(f"❌ Could not find dataset at: {RAW_DATA_PATH}")
            return
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = df.rename(columns=lambda x: x.strip())
            if "Text" not in df or "Sentiment" not in df:
                st.error("❌ Uploaded file must have 'Text' and 'Sentiment' columns.")
                return
            df = df.loc[:, ["Text", "Sentiment"]].dropna().reset_index(drop=True)
            df = simplify_labels(df)
            df['clean_text'] = df['Text'].astype(str).apply(preprocess)
            st.success("✅ Uploaded and cleaned custom dataset.")
        else:
            st.info("👈 Upload a CSV file to continue.")
            return

    st.dataframe(df.head())
    st.bar_chart(df['Sentiment_Mapped'].value_counts())

    # === 🔧 Sample Size Slider ===
    st.subheader("⚙️ Sample Size for Training")
    max_samples = len(df)
    sample_size = st.slider(
        "Select number of samples for training:",
        min_value=50,
        max_value=max_samples,
        value=min(200, max_samples),
        step=10
    )
    st.caption(f"📝 Currently using **{sample_size}** samples out of {max_samples}")

    # === 🤖 Model Section ===
    st.header("2️⃣ Train or Load Model")

    model, vectorizer = None, None
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)

    if not model_ready:
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                model, vectorizer, _, _, _ = train_model(df_sample)
                save_model_and_vectorizer(model, vectorizer, os.path.dirname(MODEL_PATH))
            st.success("✅ Model trained and saved.")
            model_ready = True
    else:
        try:
            model, vectorizer = load_model_and_vectorizer(MODEL_PATH, VECTORIZER_PATH)
            st.success("✅ Loaded existing trained model.")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            model_ready = False

    # === ✏️ Predict Sentiment ===
    if model_ready:
        st.header("3️⃣ Try Sentiment Prediction")
        user_input = st.text_area("Enter a sentence to analyze sentiment:")
        if st.button("🔍 Predict Sentiment"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text.")
            else:
                cleaned_text = preprocess(user_input)
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text)[0]
                st.success(f"🎯 Predicted Sentiment: **{prediction.upper()}**")


if __name__ == "__main__":
    main()
