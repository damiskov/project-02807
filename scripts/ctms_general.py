"""
Applying general clustering algorithms to PCA-reduced CTM representations.
"""
"""
Applying general clustering algorithms to PCA-reduced embedding representations.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def reduce_ctm_dim(ctms_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise and reduce dimensionality of embeddings using PCA."""
    df = ctms_df.copy()

    # Series -> matrix (order preserved)
    ctm_array = ctm_to_array(df["ctm"])
    logger.info(f"Reducing dimensionality of ctms with shape: {ctm_array.shape}")

    # Standardise
    scaler = StandardScaler()
    ctm_scaled = scaler.fit_transform(ctm_array)

    # PCA
    pca = PCA()  
    ctm_pca = pca.fit_transform(ctm_scaled)
    # Determine dims for >=90% variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    dim_90_var = np.argmax(explained_variance >= 0.9) + 1
    logger.info(f"PCA reduced ctms to {dim_90_var} dimensions for >=90% explained variance.")

    logger.info(f"dimensions reduced by factor of {ctm_array.shape[1] / dim_90_var:.2f}")

    # Store row-wise reduced vectors; order matches original index
    df[f"ctm_pca"] = [row for row in ctm_pca[:, :dim_90_var]]

    return df


def apply_clustering(
    ctms_df: pd.DataFrame,
    model: ClusterModel,
) -> pd.DataFrame:
    """
    Apply one of the general clustering algorithms to the PCA-reduced embeddings.
    Returns the dataframe with an additional 'cluster_label' column.

    Args:
        ctms_df: DataFrame with PCA-reduced ctms in a column 'ctm_pca'.
            - MUST contain a "ctm_pca" column with reduced ctms as numpy arrays.
        model: An instance of ClusterModel to fit and predict clusters.
    Returns:
        DataFrame with an additional 'cluster_label' column.
    """

    df = ctms_df.copy()
    ctm_array = ctm_to_array(df["ctm_pca"])
    logger.info(f"Applying {model.name} clustering on PCA-reduced ctms with shape: {ctm_array.shape}")

    model.fit(ctm_array)
    cluster_labels = model.predict(ctm_array)
    df['cluster_label'] = cluster_labels
    logger.info(f"Clustering complete. Number of clusters found: {len(np.unique(cluster_labels))}")
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
    embeddings_df: pd.DataFrame,
    model_name: str,
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

    reduce_embedding_dim(embeddings_df, model_name)
    kmeans_scores = test_kmeans(embeddings_df, model_name, kmeans_k_values)
    dbscan_scores = test_dbscan(embeddings_df, model_name, dbscan_eps_values)
    # NOTE: Hierarchical takes a long time
    # hierarchical_scores = test_hierarchical(embeddings_df, model_name, hierarchical_n_clusters_values)
    return {
        "kmeans": kmeans_scores,
        "dbscan": dbscan_scores,
        # "hierarchical": hierarchical_scores
    }
    

def cluster_analysis(
    embeddings_df: pd.DataFrame,
    clustering_model: ClusterModel,
):
    """
    Main function which performs the following pipeline:
    1. Standardizes the embeddings.
    2. Reduces dimensionality using PCA.
    3. Applies the specified clustering algorithm.
    4. Performs TF-IDF analysis per cluster on relevant columns.
    """

    ctms_reduced_df = reduce_ctm_dim(embeddings_df)
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


if __name__ == "__main__":

    ctm_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    kmeans_model = KMeansClusterModel(n_clusters=6)
    tfidf_results = cluster_analysis(
        ctm_df,
        kmeans_model,
    )
