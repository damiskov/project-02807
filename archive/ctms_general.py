"""
Applying general clustering algorithms to PCA-reduced CTM representations.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import Dict

from loguru import logger

# sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# models
from models.cluster import ClusterModel
from models.cluster import KMeansClusterModel
from models.cluster import DBSCANClusterModel
from models.cluster import HierarchicalClusterModel

# utils
from utils.load import load_ctms_dataset
from utils.tf_idf import tfidf_cluster_summary
from utils.tf_idf import tfidf_cluter_per_column

# --- helpers --

def ctm_to_array(ctm_series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of ctms to a 2D numpy array."""
    # flatten each ctm array if necessary
    flattened_ctms = [ctm.flatten() if isinstance(ctm, np.ndarray) else ctm for ctm in ctm_series]
    return np.vstack(flattened_ctms)

def preprocess_ctms(
    ctms_df: pd.DataFrame,
    n_components: int = 500,
    remove_outliers: bool = True,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Unified preprocessing pipeline: PCA → Outlier Removal.
    
    Args:
        ctms_df: DataFrame with CTMs in 'ctm' column.
        n_components: Number of PCA components to keep.
        remove_outliers: Whether to remove outliers using IQR method.
        iqr_multiplier: Multiplier for IQR when removing outliers.
    
    Returns:
        Tuple of (processed_df, pca_array) where:
            - processed_df: DataFrame after outlier filtering
            - pca_array: 2D numpy array of PCA-reduced, optionally filtered data
    """
    df = ctms_df.copy()
    
    # PCA on original data
    ctm_array = ctm_to_array(df["ctm"])
    logger.info(f"Reducing dimensionality from {ctm_array.shape[1]} → {n_components} dims")

    scaler = StandardScaler()
    ctm_scaled = scaler.fit_transform(ctm_array)

    pca = PCA(n_components=n_components)
    ctm_pca = pca.fit_transform(ctm_scaled)
    
    variance_retained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA retained {variance_retained:.2%} of variance")

    # Remove outliers in PCA space if requested
    if remove_outliers:
        mask = np.ones(len(ctm_pca), dtype=bool)
        for i in range(ctm_pca.shape[1]):
            Q1 = np.percentile(ctm_pca[:, i], 25)
            Q3 = np.percentile(ctm_pca[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            mask &= (ctm_pca[:, i] >= lower_bound) & (ctm_pca[:, i] <= upper_bound)
        
        df = df[mask]
        ctm_pca = ctm_pca[mask]
        logger.info(f"Removed {(~mask).sum()} outliers. Kept {mask.sum()} samples.")
    
    return df, ctm_pca


def apply_clustering(
    preprocessed_df: pd.DataFrame,
    pca_array: np.ndarray,
    model: ClusterModel,
) -> pd.DataFrame:
    """
    Apply clustering to preprocessed (PCA-reduced, outlier-filtered) data.

    Args:
        preprocessed_df: DataFrame after preprocessing.
        pca_array: 2D numpy array of PCA-reduced data.
        model: ClusterModel instance for clustering.
    
    Returns:
        DataFrame with 'cluster_label' column added.
    """
    df = preprocessed_df.copy()
    logger.info(f"Applying {model.name} clustering on {pca_array.shape}")

    model.fit(pca_array)
    cluster_labels = model.predict(pca_array)
    df['cluster_label'] = cluster_labels
    logger.info(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
    return df

# --- tuning functions ---

def test_kmeans(
    ctms_df: pd.DataFrame,
    k_values: list[int]
) -> dict[int, float]:
    """
    Tune KMeans clustering by evaluating silhouette scores for different k values.

    Args:
        ctms_df: DataFrame with PCA-reduced ctms in a column 'ctm_pca'.
        k_values: List of integers representing the number of clusters to try.
    Returns:
        Dictionary mapping k values to their silhouette scores.
    """
    
    scores = {}
    ctm_array = ctm_to_array(ctms_df["ctm"])

    for k in tqdm(k_values, desc="Tuning KMeans k values"):
        kmeans_model = KMeansClusterModel(n_clusters=k)
        kmeans_model.fit(ctm_array)
        labels = kmeans_model.predict(ctm_array)
        score = silhouette_score(ctm_array, labels)
        scores[k] = score

    return scores

def test_dbscan(
    ctms_df: pd.DataFrame,
    eps_values: list[float],
    min_samples: int = 5
) -> Dict[float, float]:
    """
    Tune DBSCAN clustering by evaluating silhouette scores for different eps values.
    
    Args:
        ctms_df: DataFrame with PCA-reduced ctms in a column 'ctm_pca'.
        eps_values: List of floats representing the eps values to try.
        min_samples: Minimum number of samples for a core point in DBSCAN.
    Returns:
        Dictionary mapping eps values to their silhouette scores.
    """
    scores = {}
    ctm_array = ctm_to_array(ctms_df["ctm"])

    for eps in tqdm(eps_values, desc="Tuning DBSCAN eps values"):
        dbscan_model = DBSCANClusterModel(eps=eps, min_samples=min_samples)
        dbscan_model.fit(ctm_array)
        labels = dbscan_model.labels_

        # Check if more than 1 cluster is found
        if len(set(labels)) > 1 and -1 not in set(labels):  # Exclude noise points for silhouette score
            score = silhouette_score(ctm_array, labels)
            scores[eps] = score
        else:
            scores[eps] = -1  # Invalid score for single cluster or all noise

    return scores

def test_hierarchical(
    ctms_df: pd.DataFrame,
    n_clusters_values: list[int]
) -> dict[int, float]:
    """
    Tune Hierarchical clustering by evaluating silhouette scores for different n_clusters values.

    Args:
        ctms_df: DataFrame with CTM embeddings in a column "ctm".
        n_clusters_values: List of integers representing the number of clusters to try.
    Returns:
        Dictionary mapping n_clusters values to their silhouette scores.
    """
    
    scores = {}
    ctm_array = ctm_to_array(ctms_df["ctm"])

    for n_clusters in tqdm(n_clusters_values, desc="Tuning Hierarchical n_clusters values"):
        hierarchical_model = HierarchicalClusterModel(n_clusters=n_clusters)
        hierarchical_model.fit(ctm_array)
        labels = hierarchical_model.predict(ctm_array)
        score = silhouette_score(ctm_array, labels)
        scores[n_clusters] = score

    return scores

    

# --- main pipelines ---

def cluster_tuning(
    ctms_df: pd.DataFrame,
    kmeans_k_values: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    dbscan_eps_values: list[float] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    hierarchical_n_clusters_values: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
):
    """
    Main function which performs the following pipeline:
    1. Standardizes the embeddings.
    2. Reduces dimensionality using PCA.
    3. Logs the explained variance ratio.
    4. Applies:
        - KMeans clustering
        - DBSCAN clustering
        - Hierarchical clustering for visualization
    5. Logs silhoueetette scores for each clustering method.
    6. Saves the results to JSON files.

    logs: dimensionality reduction (PCA dim for >=90% explained var) and clustering results (silhouette scores).
    """

    reduce_ctm_dim(ctms_df)
    kmeans_scores = test_kmeans(ctms_df, kmeans_k_values)
    dbscan_scores = test_dbscan(ctms_df, dbscan_eps_values)
    # NOTE: Hierarchical takes a long time
    # hierarchical_scores = test_hierarchical(ctms_df, hierarchical_n_clusters_values)
    return {
        "kmeans": kmeans_scores,
        "dbscan": dbscan_scores,
        # "hierarchical": hierarchical_scores
    }
    

def cluster_analysis(
    ctms_df: pd.DataFrame,
    clustering_model: ClusterModel,
):
    """
    Main function which performs the following pipeline:
    1. Standardizes the embeddings.
    2. Reduces dimensionality using PCA.
    3. Applies the specified clustering algorithm.
    4. Performs TF-IDF analysis per cluster on relevant columns.
    """

    ctms_reduced_df = reduce_ctm_dim(ctms_df)
    clustered_df = apply_clustering(ctms_reduced_df, clustering_model)
    # log cluster df head and columns
    logger.info(f"Clustered DataFrame head:\n{clustered_df[["name", "themes", "keywords", "involved_companies"]].head()}")
    logger.info(f"Clustered DataFrame columns: {clustered_df.columns.tolist()}")


    tfidf_results = tfidf_cluster_summary(
        clustered_df,
        save_path=f"results/ctms_general/_{clustering_model.name}_tfidf_summary.json",
        k=10,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8,
    )
    tfidf_cluter_per_column(
        clustered_df,
        text_columns=["name", "themes", "keywords", "involved_companies"],
        save_path=f"results/ctms_general/{clustering_model.name}_tfidf_per_column.json",
        k=10,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8,
    )

    return tfidf_results

def plot_clusters(
    ctm_df: pd.DataFrame,
    clustering_model: ClusterModel,
    dim: int = 2,
    n_components: int = 500,
    remove_outliers: bool = True,
    iqr_multiplier: float = 1.5,
):
    """
    Complete pipeline: Preprocess → Cluster → Plot
    
    Args:
        ctm_df: DataFrame with CTM data.
        clustering_model: ClusterModel instance.
        dim: Number of dimensions for plot (2 or 3).
        n_components: Number of PCA components.
        remove_outliers: Whether to remove outliers.
        iqr_multiplier: IQR multiplier for outlier removal.
    """
    if dim not in [2, 3]:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    # Single unified preprocessing step
    preprocessed_df, pca_array = preprocess_ctms(
        ctm_df,
        n_components=n_components,
        remove_outliers=remove_outliers,
        iqr_multiplier=iqr_multiplier
    )
    
    # Cluster on preprocessed data
    clustered_df = apply_clustering(preprocessed_df, pca_array, clustering_model)
    
    # Plot
    if dim == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            pca_array[:, 0],
            pca_array[:, 1],
            c=clustered_df['cluster_label'],
            cmap='tab10',
            alpha=0.7
        )
        plt.title(f"2D PCA Visualization of Clusters using {clustering_model.name}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label='Cluster Label')
        plt.tight_layout()
        plt.savefig(f"results/ctms_general/{clustering_model.name}_clusters_2d.png")
        plt.close()
    else:  # dim == 3
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            pca_array[:, 0],
            pca_array[:, 1],
            pca_array[:, 2],
            c=clustered_df['cluster_label'],
            cmap='tab10',
            alpha=0.7
        )
        ax.set_title(f"3D PCA Visualization of Clusters using {clustering_model.name}")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")
        plt.colorbar(scatter, label='Cluster Label', ax=ax)
        plt.tight_layout()
        plt.savefig(f"results/ctms_general/{clustering_model.name}_clusters_3d.png")
        plt.close()


if __name__ == "__main__":

    ctm_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    kmeans_model = KMeansClusterModel(n_clusters=10)

    plot_clusters(
        ctm_df,
        clustering_model=kmeans_model,
        dim=3,
        n_components=920,  # Keeps ~95% of variance in much lower dimensions
    )