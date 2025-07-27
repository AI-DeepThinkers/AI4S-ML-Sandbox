import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set paths
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../.."))
CLEANED_PATH = os.path.join(ROOT_DIR, "data/cleaned/house_prices_cleaned.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CLEANED_PATH)
X = df.drop(columns=["MEDV"])
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)

    # Predicted vs Actual plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{name}: Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_actual_vs_predicted.png"))
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{name}: Residual Distribution")
    plt.xlabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_residuals.png"))
    plt.close()

print("✅ Models trained, saved, and plots generated.")
print("📁 Output saved to:", MODEL_DIR)
