import os
import re
import string
import pandas as pd
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns






# === 📥 NLTK Downloads ===
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# === 🌐 Globals ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

LABEL_MAP = {
    'positive': [
        'positive', 'joy', 'excitement', 'admiration', 'elation', 'hopeful',
        'inspired', 'affection', 'grateful', 'proud', 'enjoyment', 'serenity',
        'satisfaction', 'kind', 'curiosity', 'contentment', 'enthusiasm',
        'inspiration', 'vibrancy', 'spark', 'zest', 'awe'
    ],
    'negative': [
        'anger', 'fear', 'sadness', 'hate', 'frustrated', 'disgust', 'loneliness',
        'grief', 'regret', 'isolation', 'heartbreak', 'numbness', 'lostlove',
        'bitter', 'bitterness', 'melancholy', 'devastated', 'sorrow'
    ],
    'neutral': [
        'neutral', 'contemplation', 'reflection', 'confusion', 'ambivalence',
        'nostalgia', 'boredom', 'solitude', 'calmness', 'coziness', 'playful'
    ]
}

FLATTENED_LABEL_MAP = {fine.lower(): general for general, fines in LABEL_MAP.items() for fine in fines}

# === 🔧 Preprocessing Functions ===

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def simplify_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Sentiment_Mapped'] = df['Sentiment'].str.strip().str.lower().map(FLATTENED_LABEL_MAP)
    df = df.dropna(subset=['Sentiment_Mapped']).copy()
    df['clean_text'] = df['Text'].astype(str).apply(preprocess)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.reset_index(drop=True, inplace=True)
    df = simplify_labels(df)
    df['clean_text'] = df['Text'].astype(str).apply(preprocess)
    return df

# === 📦 I/O Functions ===

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return clean_dataframe(df)

def save_clean_dataset(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_model_and_vectorizer(model, vectorizer, dir_path='../../models'):
    os.makedirs(dir_path, exist_ok=True)
    joblib.dump(model, os.path.join(dir_path, 'sentiment_model.pkl'))
    joblib.dump(vectorizer, os.path.join(dir_path, 'tfidf_vectorizer.pkl'))

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# === 🤖 Model Functions ===
def train_model(df: pd.DataFrame):
    # Shuffle dataset just in case
    df = shuffle(df, random_state=42)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['Sentiment_Mapped']

    # Train/test split with stratification for balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Use multinomial logistic regression (better for multiclass)
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',   # Optional: adjusts for class imbalance
        solver='lbfgs',            # Recommended for multiclass
        multi_class='multinomial',
        random_state=42
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    return model, vectorizer, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred, display_plot=True):
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
    if display_plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['negative', 'neutral', 'positive'],
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

# === 🧪 Script Execution ===

if __name__ == "__main__":
    raw_path = '../../data/raw/sentiment_dataset.csv'
    clean_path = '../../data/cleaned/sentiment_dataset_clean.csv'
    models_path = '../../models'

    print(f"📥 Loading dataset from {raw_path}")
    df = load_dataset(raw_path)
    print(f"✅ Cleaned dataset has shape: {df.shape}")
    save_clean_dataset(df, clean_path)
    print(f"📁 Saved cleaned dataset to {clean_path}")

    model, vectorizer, X_test, y_test, y_pred = train_model(df)
    evaluate_model(y_test, y_pred)
    save_model_and_vectorizer(model, vectorizer, models_path)
    print("✅ All steps complete.")
