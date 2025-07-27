import os
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data/raw")
CLEANED_DIR = os.path.join(ROOT_DIR, "data/cleaned")

RAW_PATH = os.path.join(DATA_DIR, "house_prediction.csv")
CLEANED_PATH = os.path.join(CLEANED_DIR, "house_prices_cleaned.csv")

boston_columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

def preprocess_and_save():
    # Use delim_whitespace=True for space-separated values
    df = pd.read_csv(RAW_PATH, header=None, names=boston_columns, delim_whitespace=True)

    # Drop rows where target 'MEDV' is missing
    df = df.dropna(subset=["MEDV"])

    # Fill missing numeric cols (just in case)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Save cleaned data
    os.makedirs(CLEANED_DIR, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)

    return df

if __name__ == "__main__":
    df_clean = preprocess_and_save()
    print(f"✅ Cleaned data saved to {CLEANED_PATH}. Shape: {df_clean.shape}")
