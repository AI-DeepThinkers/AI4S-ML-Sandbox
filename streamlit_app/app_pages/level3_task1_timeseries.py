import os, sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Append module path
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../Level3_Advanced/Task1_TimeSeries")))
import forecast_utils as fu
import visualizer as vz

# Data path
DATA_PATH = os.path.join("data", "cleaned", "stock_prices_cleaned.csv")

def main():
    st.title("📈 Level 3 – Task 1: Time Series Forecasting")

    df = fu.load_data(DATA_PATH)
    st.write("### 📊 Data Preview", df.head())

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric time series found for forecasting.")
        return

    ts = st.selectbox("Choose series to forecast", numeric_cols)
    series = df[ts].dropna()


    st.write("### 1. Time Series Plot")
    st.pyplot(vz.plot_series(series, label=ts))

    st.write("### 2. Decomposition")
    period = st.slider("Seasonal Period (days)", 2, 365, 30)
    st.pyplot(vz.plot_decomposition(series, period=period))

    st.write("### 3. Smoothing Techniques")
    window = st.slider("Moving Average Window", 5, 60, 20)
    ma_series = fu.apply_moving_average(series, window)
    exp_series = fu.apply_exponential_smoothing(series, seasonal_periods=period)
    st.pyplot(vz.plot_smoothing(series, ma_series, exp_series))

    st.write("### 4. ARIMA Forecasting")
    p = st.number_input("AR (p)", 0, 5, value=1)
    d = st.number_input("Difference (d)", 0, 2, value=1)
    q = st.number_input("MA (q)", 0, 5, value=1)
    train_pct = st.slider("Train Size (%)", 50, 90, 80)

    train, test = fu.train_test_split(series, train_pct)

    if st.button("Run Forecast"):
        with st.spinner("Training ARIMA model..."):
            model_fit = fu.fit_arima_model(train, order=(p, d, q))
            forecast = fu.forecast_arima(model_fit, steps=len(test))
            rmse = fu.calculate_rmse(test, forecast)
        
        st.success(f"📏 Forecast RMSE: {rmse:.3f}")
        st.pyplot(vz.plot_forecast(train, test, forecast))

        # Future forecasting
        steps = st.number_input("Future Forecast Days", 1, 180, 30)
        future_forecast = fu.forecast_arima(fu.fit_arima_model(series, order=(p, d, q)), steps=steps)
        future_index = pd.date_range(start=series.index[-1], periods=steps+1, freq="D")[1:]
        future_series = pd.Series(future_forecast.values, index=future_index)

        st.write("### 📅 Future Forecast")
        st.line_chart(future_series)
        st.write(future_series)

if __name__ == "__main__":
    main()
