import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load dataset from the given file path.
    """
    return pd.read_csv(file_path)

def summary_statistics(df):
    """
    Calculate summary statistics for the dataframe.
    """
    return df.describe()

def plot_histograms(df):
    """
    Plot histograms for all numerical features.
    """
    histograms = []
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20, edgecolor='black')
        ax.set_title(f'Distribution of {column}')
        histograms.append(fig)
    return histograms

def plot_scatter_plots(df):
    """
    Plot scatter plots for pairwise relationships of numerical features.
    """
    scatter_plots = []
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            x_col, y_col = numeric_columns[i], numeric_columns[j]
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f'{x_col} vs {y_col}')
            scatter_plots.append(fig)
    return scatter_plots

def plot_box_plots(df):
    """
    Plot box plots to visualize the spread and detect outliers.
    """
    box_plots = []
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=column, ax=ax)
        ax.set_title(f'Box Plot of {column}')
        box_plots.append(fig)
    return box_plots

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix for numerical features.
    """
    correlation_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    return fig
