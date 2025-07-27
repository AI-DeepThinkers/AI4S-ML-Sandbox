import os
import sys
import streamlit as st

# Set path to access Level1_Basic and app_pages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import app pages
from app_pages import (
    level1_task1_scraping,
    level1_task2_cleaning,
    level1_task3_eda,
    level2_task1_regression,
    level2_task2_classification,
    level2_task3_clustering,
    level3_task1_timeseries,
)

# Streamlit page config
st.set_page_config(page_title="AI4SE-ML-Sandbox App", layout="wide")
st.title("ML-Sandbox Tasks")

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")

level_choice = st.sidebar.selectbox("Select Level", ["Level 1 – Basic", "Level 2 – Intermediate", "Level 3 – Advanced"])

if level_choice == "Level 1 – Basic":
    task_choice = st.sidebar.radio("Select Task", [
        "Task 1: Web Scraping",
        "Task 2: Data Cleaning",
        "Task 3: EDA"
    ])
    
    if task_choice == "Task 1: Web Scraping":
        level1_task1_scraping.main()

    elif task_choice == "Task 2: Data Cleaning":
        level1_task2_cleaning.main()

    elif task_choice == "Task 3: EDA":
        level1_task3_eda.main()

elif level_choice == "Level 2 – Intermediate":
    task_choice = st.sidebar.radio("Select Task", [
        "Task 1: Regression",
        "Task 2: Classification",
        "Task 3: Clustering"
    ])

    if task_choice == "Task 1: Regression":
        level2_task1_regression.main()

    elif task_choice == "Task 2: Classification":
        level2_task2_classification.main()

    elif task_choice == "Task 3: Clustering":
        level2_task3_clustering.main()

elif level_choice == "Level 3 – Advanced":
    task_choice = st.sidebar.radio("Select Task", [
        "Task 1: Time Series Forecasting",
        "Task 2: NLP Text Classification",
        "Task 3: Neural Network"
    ])
    if task_choice == "Task 1: Time Series Forecasting":
        level3_task1_timeseries.main()
    else:
        st.info("🚧 Task coming soon!")

