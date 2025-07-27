import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional, Dict

# Project directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data")  # Base data folder

# --- Dataset Detection ---
def detect_dataset_type(df: pd.DataFrame) -> str:
    """Classifies dataset as time_series, text, tabular, or unknown."""
    if "Date" in df.columns or pd.api.types.is_datetime64_any_dtype(df.index):
        return "time_series"
    
    text_like_cols = sum(
        df[col].dtype == "object" and df[col].astype(str).str.len().mean() > 40
        for col in df.select_dtypes("object")
    )
    if text_like_cols >= 1:
        return "text"

    if len(df.select_dtypes(include=["int64", "float64"]).columns) >= 2:
        return "tabular"
    
    return "unknown"

# --- Load Datasets ---
def load_raw_data(filename: str) -> pd.DataFrame:
    raw_dir = os.path.join(DATA_DIR, "raw")
    file_path = os.path.join(raw_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    # Try normal CSV read
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"⚠️ Failed default read_csv for {filename}: {e}")
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # --- Fix if data loads as a single column (space-separated values)
    if df.shape[1] == 1:
        col = df.columns[0]
        try:
            df = df[col].astype(str).str.split(expand=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            print(f"⚠️ Auto-split space-separated values in {filename} → shape={df.shape}")
        except Exception as e:
            print(f"Failed to auto-split 1-column CSV: {e}")

    # --- Inject column names for known headerless datasets
    if filename == "house_prediction.csv" and df.shape[1] == 14:
        df.columns = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
        ]
        print(f"✅ Applied column names for house_prediction.csv")

    print(f"Loaded raw data from {file_path} (shape={df.shape})")
    return df


# --- Cleaning Function ---
def clean_and_preprocess(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    remove_outliers: Optional[bool] = None,
    normalize: Optional[bool] = None,
    detected_type: Optional[str] = None
) -> pd.DataFrame:
    """Clean and preprocess dataset with automatic behavior based on type."""
    df = df.copy()

    # Auto-detect type if not provided
    if not detected_type:
        detected_type = detect_dataset_type(df)

    # Set default behavior based on type
    if detected_type == "time_series":
        remove_outliers = False if remove_outliers is None else remove_outliers
        normalize = False if normalize is None else normalize
    else:
        remove_outliers = True if remove_outliers is None else remove_outliers
        normalize = True if normalize is None else normalize

    # Separate column types
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if target_column and target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Missing value handling
    if numerical_cols:
        df[numerical_cols] = SimpleImputer(strategy="mean").fit_transform(df[numerical_cols])

    if categorical_cols:
        df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols])

    if target_column:
        df[[target_column]] = SimpleImputer(strategy="most_frequent").fit_transform(df[[target_column]])

    # Time series: fill NA forward/backward
    if detected_type == "time_series":
        df = df.ffill().bfill()

    # Outlier removal
    if remove_outliers and numerical_cols:
        z_scores = np.abs(stats.zscore(df[numerical_cols]))
        df = df[(z_scores < 3).all(axis=1)]

    # Normalization
    if normalize and numerical_cols:
        df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])

    return df

# --- Save Cleaned Data ---
def save_cleaned_data(df: pd.DataFrame, filename: str = "cleaned_data.csv") -> None:
    cleaned_dir = os.path.join(DATA_DIR, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)
    file_path = os.path.join(cleaned_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Cleaned data saved at {file_path}")

# --- Summary Generator ---
def generate_cleaning_summary(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame, dataset_type: Optional[str] = None) -> Dict[str, int]:
    return {
        "Dataset type": dataset_type or detect_dataset_type(df_raw),
        "Rows before cleaning": df_raw.shape[0],
        "Rows after cleaning": df_cleaned.shape[0],
        "Columns before cleaning": df_raw.shape[1],
        "Columns after cleaning": df_cleaned.shape[1],
        "Outlier rows removed": df_raw.shape[0] - df_cleaned.shape[0],
    }
