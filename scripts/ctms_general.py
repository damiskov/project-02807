"""
Clean 3-function clustering pipeline for CTM-based similarity analysis.
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from typing import List, Optional

# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# your models
from models.cluster import ClusterModel
from models.cluster import KMeansClusterModel, DBSCANClusterModel, HierarchicalClusterModel

# utils
from utils.tf_idf import tfidf_cluster_summary, tfidf_cluter_per_column
from utils.load import load_ctms_dataset
# from utils.metrics import davies_bouldin_score, silhouette_score,
from utils.metrics import column_wise_summary


# -------- plot settings ----------

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # --- Figure ---
    "figure.figsize": (6, 4),
    "figure.dpi": 300,

    # --- Fonts ---
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "font.family": "serif",       # or "sans-serif"
    "mathtext.fontset": "stix",   # clean math font

    # --- Axes ---
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": False,

    # --- Ticks ---
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    # --- Legend ---
    "legend.fontsize": 10,
    "legend.frameon": False,

    # --- Colors / Colormap ---
    "image.cmap": "viridis",
})

# -------- preprocessing ----------

def ctm_to_array(ctm_series: pd.Series) -> np.ndarray:
    """Convert CTMs into a stacked 2D array."""
    return np.vstack([
        ctm.flatten() if isinstance(ctm, np.ndarray) else ctm
        for ctm in ctm_series
    ])


def preprocess_ctms(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    iqr_multiplier: float = 2.5
):
    """Convert CTMs -> PCA -> optional outlier filtering."""
    X = ctm_to_array(df["ctm"])

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA full, then select cutoff at ≥90% explained variance
    pca_full = PCA().fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = np.argmax(cum >= 0.90) + 1
    logger.info(f"CTMs: PCA -> {k} dims (≥90% variance)")

    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_scaled)

    # Outlier removal
    if remove_outliers:
        mask = np.ones(len(X_pca), dtype=bool)
        for j in range(X_pca.shape[1]):
            q1, q3 = np.percentile(X_pca[:, j], [25, 75])
            iqr = q3 - q1
            lb, ub = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
            mask &= (X_pca[:, j] >= lb) & (X_pca[:, j] <= ub)

        logger.info(f"Removed {(~mask).sum()} outliers.")
        return df[mask].reset_index(drop=True), X_pca[mask]

    return df.reset_index(drop=True), X_pca


# ------- cluster tuning -----------

def tune_clusters(
    df: pd.DataFrame,
    k_list: Optional[List[int]] = None,
    eps_list: Optional[List[float]] = None,
    n_clusters_list: Optional[List[int]] = None,
    save_path: str = "results/ctms_general",

):
    """
    Unified cluster tuning function.
    Returns silhouette scores for KMeans, DBSCAN, Hierarchical.
    """
    logger.info("Preprocessing CTMs...")
    df_proc, X = preprocess_ctms(df)

    silhouette_scores = {
        "kmeans": {},
        "dbscan": {},
        "hierarchical": {}
    }
    davies_bouldin_scores = {
        "kmeans": {},
        "dbscan": {},
        "hierarchical": {}
    }

    # ---- KMeans ----
    if k_list:
        for k in tqdm(k_list, desc="Tuning KMeans"):
            # model = KMeansClusterModel(n_clusters=k)
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X)
            silhouette_scores["kmeans"][k] = silhouette_score(X, labels)
            davies_bouldin_scores["kmeans"][k] = davies_bouldin_score(X, labels)

    # ---- DBSCAN ----
    if eps_list:
        for eps in tqdm(eps_list, desc="Tuning DBSCAN"):
            model = DBSCANClusterModel(eps=eps, min_samples=5)
            model.fit(X)
        labels = model.labels_

        # silhouette requires ≥2 clusters and no noise-label issues
        if len(set(labels)) > 1 and -1 not in labels:
            silhouette_scores["dbscan"][eps] = silhouette_score(X, labels)
            davies_bouldin_scores["dbscan"][eps] = davies_bouldin_score(X, labels)
        else:
            score = -1

        silhouette_scores["dbscan"][eps] = score
        davies_bouldin_scores["dbscan"][eps] = None  # invalid

    # ---- Hierarchical ----
    # for n in tqdm(n_clusters_list, desc="Tuning Hierarchical"):
    #     model = HierarchicalClusterModel(n_clusters=n)
    #     labels = model.fit_predict(X)
    #     score = silhouette_score(X, labels)
    #     results["hierarchical"][n] = score

    score_path = f"{save_path}/ctm_tuning_scores.json"
    with open(score_path, "w") as f:
        json.dump({
            "silhouette": silhouette_scores,
            "davies_bouldin": davies_bouldin_scores
        }, f, indent=2)

    return silhouette_scores, davies_bouldin_scores


# -------- cluster plotting ----------

def plot_clusters(
    df: pd.DataFrame,
    model: ClusterModel,
    dim: int = 2,
    remove_outliers: bool = True,
    save_dir: str = "results/ctms_general/plots",
    model_name: str = "ctm"
):
    """
    Preprocess -> cluster -> plot PCA (2D/3D).
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    df_proc, X = preprocess_ctms(df, remove_outliers=remove_outliers)

    labels = model.fit_predict(X)
    df_proc["cluster_label"] = labels

    # ---- 2D ----
    if dim == 2:
        plt.figure(figsize=(10, 8))
        s = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.title(f"2D PCA clusters (CTM)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        save_path = f"{save_dir}/{model_name}_2D_clusters.png"
        plt.savefig(save_path, dpi=300)
        return df_proc

    # ---- 3D ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="tab10", alpha=0.7)
    ax.set_title(f"3D PCA clusters (CTM)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    save_path = f"{save_dir}/{model_name}_3D_clusters.png"
    plt.savefig(save_path, dpi=300)
    return df_proc


# -------- TF-IDF metadata evaluation ----------

def evaluate_metadata(
    df: pd.DataFrame,
    model: ClusterModel,
    save_dir: str = "results/ctms_general/json"
):
    """
    Preprocess -> cluster -> TF-IDF global + per-column.
    Saves two JSON files.
    """
    df_proc, X = preprocess_ctms(df)
    df_proc["cluster_label"] = model.fit_predict(X)

    path_global = f"{save_dir}/ctms_tfidf_global.json"
    path_cols = f"{save_dir}/ctms_column_summary.json"

    tfidf_cluster_summary(
        df_proc,
        save_path=path_global,
        k=10,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8,
    )

    column_wise_summary(
        df_proc,
        columns=["name", "themes", "keywords", "involved_companies", "first_release_year"],
        save_path=path_cols,
    )

    return df_proc


# ---- driver code ----

def tuning_pipeline_driver():
    """
    Driver code for tuning CTM clustering models.
    """
    ctm_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    

    # n_clusters = [2, 4, 6, 10]
    # eps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    k_list = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    logger.info("Tuning clustering for CTM representations…")
    tuning_results = tune_clusters(
        ctm_df,
        k_list=k_list
    )
    logger.info(f"Tuning results: {tuning_results}")

def plotting_pipeline_driver():
    """
    Driver code for plotting embedding clustering results.
    """
    ctms_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    k = 6 
    # model = KMeansClusterModel(n_clusters=k)
    model = KMeans(n_clusters=k, random_state=42)

    logger.info(f"Plotting clustering for ctms with K={k}…")
    clustered_df = plot_clusters(
        ctms_df,
        model=model,
        dim=2
    )
    logger.info(f"Clustered DataFrame head:\n{clustered_df.head()}")


if __name__ == "__main__":
    df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    print(df[["name", "themes", "keywords", "involved_companies", "first_release_year"]].head())
    # tuning_pipeline_driver()
    # plotting_pipeline_driver()
    evaluate_metadata(
        df,
        model=KMeans(n_clusters=6, random_state=42),
        save_dir="results/ctms_general/json"
    )