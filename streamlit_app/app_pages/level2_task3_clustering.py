import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
DATA_DIR = os.path.join("data", "cleaned")

def load_dataset():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    csv_files.sort()

    default_index = next((i for i, name in enumerate(csv_files) if "iris" in name.lower()), 0)
    selected_file = st.selectbox("Select dataset", csv_files, index=default_index)

    df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
    return df, selected_file

def preprocess_data(df):
    df_numeric = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_numeric)
    return pd.DataFrame(scaled, columns=df_numeric.columns)

def plot_elbow(data):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    st.pyplot(plt)

def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    reduced_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    reduced_df['Cluster'] = labels

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60)
    plt.title("Clusters Visualized with PCA")
    st.pyplot(plt)

def main():
    st.title("🔵 Level 2 – Task 3: K-Means Clustering")

    df, filename = load_dataset()
    st.write("### 📊 Dataset Preview:", filename)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns for clustering.")
        return

    scaled_df = preprocess_data(df)

    st.write("### 📉 Elbow Method to Determine Optimal K")
    plot_elbow(scaled_df)

    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_df)

    sil_score = silhouette_score(scaled_df, cluster_labels)
    st.success(f"Silhouette Score for k={k}: {sil_score:.4f}")

    df['Cluster'] = cluster_labels

    st.write("### 📍 Cluster Visualization (via PCA)")
    plot_clusters(scaled_df, cluster_labels)

    st.write("### 📌 Cluster Summary")
    try:
        st.dataframe(df.groupby('Cluster')[numeric_cols].mean())
    except Exception as e:
        st.error(f"Could not compute cluster summary: {e}")

if __name__ == "__main__":
    main()
