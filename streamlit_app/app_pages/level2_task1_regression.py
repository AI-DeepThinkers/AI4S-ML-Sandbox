import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load cleaned dataset
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CLEANED_PATH = os.path.join(ROOT_DIR, "data/cleaned/house_prediction_cleaned.csv")

def load_data():
    df = pd.read_csv(CLEANED_PATH)
    return df

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model": model, "name": name, "mse": mse, "r2": r2, "predictions": preds}

def main():
    st.title("🏡 Level 2 – Task 1: Regression Model for House Price Prediction")

    df = load_data()
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.markdown("### 🔍 Model Evaluation")
    results = []

    results.append(evaluate_model("Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test))
    results.append(evaluate_model("Decision Tree", DecisionTreeRegressor(random_state=42), X_train, X_test, y_train, y_test))
    results.append(evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test))

    for res in results:
        st.markdown(f"#### ✅ {res['name']}")
        st.write(f"**Mean Squared Error (MSE):** {res['mse']:.4f}")
        st.write(f"**R-squared (R²):** {res['r2']:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, res["predictions"], alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title(f"{res['name']} Predictions")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
