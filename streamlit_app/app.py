import os
import sys
import streamlit as st
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_pages import (
    level1_task1_scraping,
    level1_task2_cleaning,
    level1_task3_eda,
    level2_task1_regression,
    level2_task2_classification,
    level2_task3_clustering,
    level3_task1_timeseries,
    level3_task2_nlp,
)

st.set_page_config(
    page_title="Data Science Project Showcase",
    layout="wide"
)

st.title("Multi-Level Data Science Tasks")
st.caption("Explore data cleaning, ML models, NLP, and more – organized by task and level.")

# Sidebar Navigation
st.sidebar.title("Navigation")

nav_choice = st.sidebar.radio(
    "Navigate to:",
    ("HOME / README", "Level 1 – Basic", "Level 2 – Intermediate", "Level 3 – Advanced"),
    index=0,  # default to Home/README
)

if nav_choice == "HOME / README":
    readme_path = Path(__file__).parent.parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    else:
        st.warning("README.md not found.")

elif nav_choice == "Level 1 – Basic":
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

elif nav_choice == "Level 2 – Intermediate":
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

elif nav_choice == "Level 3 – Advanced":
    task_choice = st.sidebar.radio("Select Task", [
        "Task 1: Time Series Forecasting",
        "Task 2: NLP Text Classification",
        "Task 3: Neural Network"
    ])
    if task_choice == "Task 1: Time Series Forecasting":
        level3_task1_timeseries.main()
    elif task_choice == "Task 2: NLP Text Classification":
        level3_task2_nlp.main()
    else:
        st.info("🚧 Task coming soon!")

with st.sidebar:
    st.markdown("---")
    st.markdown("Built with ❤️ by [KP Matlakala](https://github.com/DeLightPlus)")
    st.markdown("Part of the #AI4SE and #DataSciencePortfolio initiatives.")

    st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/kabelo-matlakala-704349273)")
