import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summary_statistics(df):
    return df.describe()

def plot_histograms(df):
    histograms = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20, edgecolor='black')
        ax.set_title(f'Distribution of {col}')
        histograms.append(fig)
    return histograms

def plot_scatter_plots(df):
    scatter_plots = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=numeric_cols[i], y=numeric_cols[j], ax=ax)
            ax.set_title(f'{numeric_cols[i]} vs {numeric_cols[j]}')
            scatter_plots.append(fig)
    return scatter_plots

def plot_box_plots(df):
    box_plots = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f'Box Plot of {col}')
        box_plots.append(fig)
    return box_plots

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix for numerical features only.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    correlation_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))  # Reduce height
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix (Numerical Features Only)', fontsize=14)
    return fig

