import os
import pandas as pd
import streamlit as st

from Level1_Basic.Task3_EDA.eda import (
    summary_statistics,
    plot_histograms,
    plot_scatter_plots,
    plot_box_plots,
    plot_correlation_matrix
)

def main():
    st.title("📊 Level 1 – Task 3: Exploratory Data Analysis (EDA)")

    st.markdown("""
    This tool helps you analyze a cleaned dataset using:
    - Summary statistics
    - Histograms
    - Scatter plots
    - Box plots
    - Correlation matrix
    """)

    # Directory containing cleaned CSV files
    cleaned_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cleaned"))
    csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith(".csv")]

    if not csv_files:
        st.error("⚠️ No cleaned datasets found in `data/cleaned/`. Please run Task 2 first or upload a file.")
        return

    # Default to iris_cleaned.csv if it exists
    default_file = "iris_cleaned.csv" if "iris_cleaned.csv" in csv_files else csv_files[0]
    selected_file = st.selectbox("📂 Select a cleaned dataset from disk:", csv_files, index=csv_files.index(default_file))

    # Optional file uploader
    st.markdown("---")
    uploaded_file = st.file_uploader("⬆️ Or upload your own cleaned CSV", type="csv")

    # Load dataset (uploaded or selected)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded file loaded successfully.")
    else:
        data_path = os.path.join(cleaned_dir, selected_file)
        df = pd.read_csv(data_path)
        st.success(f"✅ Loaded `{selected_file}` from cleaned data.")

    # Preview Data
    st.markdown("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.write(summary_statistics(df))

    # Histograms
    st.subheader("📊 Histograms")
    for fig in plot_histograms(df):
        st.pyplot(fig)

    # Scatter Plots
    st.subheader("🔍 Scatter Plots")
    for fig in plot_scatter_plots(df):
        st.pyplot(fig)

    # Box Plots
    st.subheader("📦 Box Plots")
    for fig in plot_box_plots(df):
        st.pyplot(fig)

    # Correlation Matrix
    st.subheader("🧮 Correlation Matrix")
    st.pyplot(plot_correlation_matrix(df))
