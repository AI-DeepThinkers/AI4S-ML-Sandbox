import sys
import os
import streamlit as st
import pandas as pd

# Set up import paths relative to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Level1_Basic.Task2_DataCleaning.cleaner import load_raw_data, clean_and_preprocess, save_cleaned_data

# Define DATA_DIR (assumed relative to project root)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))


def main():
    st.set_page_config(page_title="🧼 Data Cleaning | Level 1 - Task 2", layout="wide")
    st.title("🧼 Level 1 – Task 2: Data Cleaning & Preprocessing")

    st.markdown("""
    Clean and preprocess a raw dataset to make it suitable for analysis.

    **Steps include:**
    - Handling missing values
    - Removing or treating outliers
    - Encoding categorical variables
    - Normalizing numerical values
    """)

    raw_dir = os.path.join(DATA_DIR, "raw")
    if not os.path.exists(raw_dir):
        st.error(f"Raw data directory does not exist: {raw_dir}")
        return

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    if not raw_files:
        st.warning("❗ No raw CSV files found in data/raw/. Please add some data first.")
        return

    filename = st.selectbox("📂 Select a raw CSV file to clean:", raw_files)

    if not filename:
        st.warning("Please select a file to load.")
        return

    # Load raw data from selected file
    try:
        df_raw = load_raw_data(filename)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # No explicit target column for scraped text, so None
    target_col = None

    st.markdown("### 📄 Raw Data Preview")
    st.dataframe(df_raw)

    if st.button("🧹 Clean & Preprocess"):
        with st.spinner("Processing..."):
            rows_before, cols_before = df_raw.shape

            cleaned_df = clean_and_preprocess(df_raw, target_column=target_col)

            rows_after, cols_after = cleaned_df.shape
            outliers_removed = rows_before - rows_after

            cleaned_filename = filename.replace(".csv", "_cleaned.csv")
            save_cleaned_data(cleaned_df, filename=cleaned_filename)

        st.success(f"✅ Data cleaned and saved to `data/cleaned/{cleaned_filename}`")

        st.markdown("### 📊 Cleaning Summary")
        st.markdown(f"""
        - **Rows before cleaning:** {rows_before}  
        - **Rows after cleaning:** {rows_after}  
        - **Columns before cleaning:** {cols_before}  
        - **Columns after cleaning:** {cols_after}  
        - **🚫 Outlier rows removed:** {outliers_removed}  
        """)

        st.markdown("### ✅ Cleaned Data Preview")
        st.dataframe(cleaned_df)

        csv = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Data as CSV",
            data=csv,
            file_name=cleaned_filename,
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
