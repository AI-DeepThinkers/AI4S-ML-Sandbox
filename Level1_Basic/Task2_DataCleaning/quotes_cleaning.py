# quotes_cleaning.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
df = pd.read_csv("data/quotes.csv")
print("\n🔹 Original Dataset:")
print(df.head())

# --- Step 1: Handle Missing Data ---
print("\n🔍 Checking for missing values...")
print(df.isnull().sum())
df.dropna(inplace=True)  # Drop rows with any missing value for simplicity

# --- Step 2: Clean Tags ---
# Convert tags from comma-separated string to list
df["Tags"] = df["Tags"].apply(lambda x: [tag.strip() for tag in x.split(",")] if pd.notnull(x) else [])

# --- Step 3: Encode Author ---
# Label Encoding (if needed for ML models)
le = LabelEncoder()
df["Author_encoded"] = le.fit_transform(df["Author"])

# One-Hot Encoding example (optional alternative)
# df = pd.get_dummies(df, columns=["Author"], prefix="Author")

# --- Step 4: Vectorize Quotes ---
# Convert text to TF-IDF feature vectors
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(df["Quote"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# --- Combine TF-IDF with encoded author ---
df_cleaned = pd.concat([df[["Author", "Author_encoded"]], tfidf_df], axis=1)

# --- Save cleaned data ---
df_cleaned.to_csv("data/quotes_cleaned.csv", index=False)
print("\n✅ Cleaned data saved to data/quotes_cleaned.csv")

