# 📈 Level 3 – Task 1: Time Series Forecasting

## 🧠 Objective

The goal of this advanced task is to **analyze**, **decompose**, and **forecast** time-series data using statistical and machine learning approaches. This task focuses on practical forecasting techniques for real-world datasets (e.g., stock prices), evaluating model performance, and visualizing results.

---

## 📂 Project Structure

Level3_Task1_TimeSeries/
├── forecast_utils.py # Data loading, smoothing, ARIMA modeling
├── visualizer.py # Plots: time series, decomposition, smoothing, forecast
streamlit_app/
└── app_pages/
└── level3_task1_timeseries.py # Interactive Streamlit dashboard
data/
├── raw/
│ └── stock_prices.csv
└── cleaned/
└── stock_prices_cleaned.csv

---

## 🔧 Tools Used

- **Python**
- **pandas** – Data wrangling
- **matplotlib** – Visualization
- **statsmodels** – Time series modeling (ARIMA, decomposition, etc.)
- **scikit-learn** – RMSE evaluation
- **Streamlit** – Interactive dashboard

---

## 📌 Features Implemented

### ✅ 1. Data Cleaning & Preprocessing
> Using a shared data pipeline, `stock_prices.csv` was cleaned and saved as `stock_prices_cleaned.csv` with:
- Date parsing
- Missing value handling
- Sorting by time

---

### ✅ 2. Time Series Plotting
> Visualizes the selected column over time.

- Helps detect trends, spikes, or irregularities.

---

### ✅ 3. Time Series Decomposition
> Decomposes the series into:
- **Trend** – Long-term movement
- **Seasonality** – Repeating patterns
- **Residual** – Noise after removing trend & seasonality

Tool used: `statsmodels.seasonal_decompose`

---

### ✅ 4. Smoothing Techniques
- **Moving Average** – Smooths noise using a sliding window
- **Exponential Smoothing** – Weighted smoothing with recent data emphasized

This helps reveal patterns and stabilize fluctuations.

---

### ✅ 5. ARIMA Forecasting
- Fit an **ARIMA** model with user-defined `(p, d, q)` parameters
- Train/test split adjustable via slider
- Forecast plotted against test data
- Evaluation via **RMSE (Root Mean Squared Error)**

---

### ✅ 6. Streamlit Dashboard
> An interactive web UI allows users to:
- Explore data
- Apply smoothing
- Run ARIMA forecasts
- Visualize results in real time

---

## 📷 Sample Visuals

- 📈 Line plot showing historical stock prices
- 🔄 Decomposition into trend, seasonality, residual
- 📉 ARIMA forecast vs actual values
- 📏 RMSE metric for model evaluation

---

## 📝 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app
```bash
streamlit run streamlit_app/app_pages/level3_task1_timeseries.py
```

3. Explore in browser
Select time series column

Tune ARIMA parameters and train size

Observe decomposition and forecast

📊 Interpretation Tips
Trend shows overall direction.

Seasonality highlights regular intervals (e.g., monthly dips).

Residuals should look random — patterns mean model is missing info.

Forecast close to actual values → good ARIMA configuration.

Use RMSE to compare different model setups.

