import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_series(series, label="Series"):
    fig, ax = plt.subplots(figsize=(12, 6))  # wider + taller
    ax.plot(series, label=label)
    ax.set_ylabel("Value")
    ax.set_title(f"{label} Over Time")
    ax.legend()
    return fig


def plot_decomposition(series, period=30):
    result = seasonal_decompose(series, model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(10, 8)
    return fig

def plot_smoothing(original, moving_avg, exp_smooth):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(original, label="Original", alpha=0.5)
    ax.plot(moving_avg, label="Moving Average", linestyle="--")
    ax.plot(exp_smooth, label="Exp. Smoothing", linestyle=":")
    ax.legend()
    return fig

def plot_forecast(train, test, forecast):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train, label="Train")
    ax.plot(test, label="Test")
    ax.plot(forecast, label="Forecast", linestyle="--")
    ax.legend()
    return fig
