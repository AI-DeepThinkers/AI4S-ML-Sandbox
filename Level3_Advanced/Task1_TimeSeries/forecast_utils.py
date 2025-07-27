import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def load_data(path):
    df = pd.read_csv(path)

    # Try to detect a Date column
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        raise ValueError("No date column found in the dataset.")

    df = df.sort_index()
    return df

def apply_moving_average(series, window):
    return series.rolling(window).mean()

def apply_exponential_smoothing(series, seasonal_periods=30):
    model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=seasonal_periods)
    fit = model.fit()
    return fit.fittedvalues

def train_test_split(series, train_pct):
    split = int(len(series) * train_pct / 100)
    train = series[:split]
    test = series[split:]
    return train, test

def fit_arima_model(train, order):
    model = ARIMA(train, order=order)
    fit = model.fit()
    return fit

def forecast_arima(model_fit, steps):
    return model_fit.forecast(steps=steps)

def calculate_rmse(test, forecast):
    mse = mean_squared_error(test, forecast)
    return np.sqrt(mse)
