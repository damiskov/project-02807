"""
Clean 3-function clustering pipeline for CTM-based similarity analysis.
"""
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from typing import List, Optional

# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


# your models
from models.cluster import ClusterModel
from models.cluster import KMeansClusterModel, DBSCANClusterModel, HierarchicalClusterModel

# utils
from utils.tf_idf import tfidf_cluster_summary, tfidf_cluter_per_column
from utils.load import load_ctms_dataset
# from utils.metrics import davies_bouldin_score, silhouette_score,
from utils.metrics import column_wise_summary
from utils.mahalanobis import mahalanobis_mask
from utils.metrics import plot_merged_attribute_heatmap


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
    n_components: Optional[int] = None,
    remove_outliers: bool = True,
    alpha: float = 0.001  # significance level for Mahalanobis
):
    X = ctm_to_array(df["ctm"])

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA: keep 90% variance
    pca_full = PCA().fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = np.argmax(cum >= 0.90) + 1
    logger.info(f"CTMs: PCA -> {k} dims (≥90% variance)")

    if n_components is not None:
        k = n_components
   
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_scaled)

    if remove_outliers:
        mask, d2, threshold = mahalanobis_mask(X_pca, alpha=alpha)
        logger.info(f"Removed {(~mask).sum()} outliers (Mahalanobis, χ² threshold={threshold:.2f})")
        return df[mask].reset_index(drop=True), X_pca[mask]

    return df.reset_index(drop=True), X_pca


# ------- cluster tuning -----------

def tune_clusters(
    df: pd.DataFrame,
    k_list: Optional[List[int]] = None,
    n_components: Optional[int] = None,
    save_dir: str = "results/ctms_general/json",
    save: bool = True

):
    """
    Unified cluster tuning function.
    Returns silhouette scores for KMeans, DBSCAN, Hierarchical.
    """
    logger.info("Preprocessing CTMs...")
    df_proc, X = preprocess_ctms(df, n_components=n_components, remove_outliers=True)

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

    if save:
        score_path = f"{save_dir}/ctm_kmeans_tuning_scores.json"
        with open(score_path, "w") as f:
            json.dump({
                "silhouette": silhouette_scores["kmeans"],
                "davies_bouldin": davies_bouldin_scores["kmeans"]
            }, f, indent=2)

    return silhouette_scores, davies_bouldin_scores


# -------- cluster plotting ----------

def plot_clusters(
    df: pd.DataFrame,
    model: ClusterModel,
    n_components: Optional[int] = None,
    dim: int = 2,
    remove_outliers: bool = True,
    save_dir: str = "results/ctms_general/figs",
    model_name: str = "ctm",
    save: bool = True
):
    """
    Preprocess -> cluster -> plot PCA (2D/3D).
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    df_proc, X = preprocess_ctms(
        df,
        remove_outliers=remove_outliers,
        n_components=n_components
    )

    labels = model.fit_predict(X)
    df_proc["cluster_label"] = labels
    
    # randomly sample points if too many
    if X.shape[0] > 1000:
        sample_indices = np.random.choice(X.shape[0], size=1000, replace=False)
        X = X[sample_indices]
        labels = labels[sample_indices]
        df_proc = df_proc.iloc[sample_indices].reset_index(drop=True)

    # ---- 2D ----
    if dim == 2:
        plt.figure(figsize=(10, 8))
        s = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.title(f"2D PCA clusters (CTM)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.colorbar(s, label="Cluster Label")
        plt.show()
        save_path = f"{save_dir}/{model_name}_2D_clusters.png"
        if save:
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
    plt.colorbar(s, label="Cluster Label")
    plt.show()
    save_path = f"{save_dir}/{model_name}_3D_clusters.png"
    if save:
        plt.savefig(save_path, dpi=300)
    return df_proc


# -------- TF-IDF metadata evaluation ----------

def evaluate_metadata(
    k: int = 6,
    n_components: Optional[int] = 30,
    save_dir: str = "results/ctms_general/json"
):
    """
    Preprocess -> cluster -> TF-IDF global + per-column.
    Saves two JSON files.
    """
    model = KMeans(n_clusters=k, random_state=42)
    
    df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    df_proc, X = preprocess_ctms(df, n_components=n_components, remove_outliers=True)
    df_proc["cluster_label"] = model.fit_predict(X)

    path_global = f"{save_dir}/ctms_tfidf_global.json"
    path_cols = f"{save_dir}/ctms_column_summary.json"
    
    # check the metadata columns exist
    required_columns = ["name", "themes", "keywords", "involved_companies", "first_release_year"]
    for col in required_columns:
        if col not in df_proc.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        
    # print the columns
    print("DataFrame columns:", df_proc.columns.tolist())
    
 

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

def tuning_pipeline_driver(
    k_list: List[int],
    n_components: Optional[int] = None
):
    """
    Driver code for tuning CTM clustering models.
    """
    ctm_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    

    logger.info("Tuning clustering for CTM representations…")
    tuning_results = tune_clusters(
        ctm_df,
        k_list=k_list,
        n_components=n_components
    )
    logger.info(f"Tuning results: {tuning_results}")

def plotting_pipeline_driver(
    k: int = 6,
    dim: int = 3,
    n_components: Optional[int] = None
):
    """
    Driver code for plotting embedding clustering results.
    """
    ctms_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    # model = KMeansClusterModel(n_clusters=k)
    model = KMeans(n_clusters=k, random_state=42)

    logger.info(f"Plotting clustering for ctms with K={k}, dim={dim}, n_components={n_components}…")
    clustered_df = plot_clusters(
        ctms_df,
        n_components=n_components,
        model=model,
        dim=dim,
        remove_outliers=True,
        
    )
    logger.info(f"Clustered DataFrame head:\n{clustered_df.head()}")
    
def javis_awful_heatmap_driver(
    k: int = 6,
    n_components: Optional[int] = 30,
    save_dir: str = "results/ctms_general/figs"
):
    """
    Driver code for plotting merged attribute heatmap.
    """
    ctms_df = load_ctms_dataset("data/videogame_sequences/sequence_dataset.parquet")
    model = KMeans(n_clusters=k, random_state=42)

    logger.info(f"Generating merged attribute heatmap for CTMs with K={k}, n_components={n_components}…")
    clustered_df = evaluate_metadata(
        k=k,
        n_components=n_components,
        save_dir="results/ctms_general/json"
    )

    save_path = f"{save_dir}/ctms_merged_attribute_heatmap_k{k}.png"
    # plot_merged_attribute_heatmap(
    #     clustered_df,
    #     labels=clustered_df["cluster_label"].values,
    #     save_path=save_path,
    #     top_n=5
    # )
    # logger.info(f"Saved merged attribute heatmap to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CTM clustering pipeline")
    
    parser.add_argument("--mode", choices=["tune", "plot", "evaluate", "javi"], required=True,
                        help="Pipeline mode: tune, plot, evaluate, or javi")
    
    # Tuning arguments
    parser.add_argument("--k-list", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25],
                        help="List of k values for tuning")
    
    # Plotting arguments
    parser.add_argument("--k", type=int, default=6, help="Number of clusters for plotting")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3, help="Plot dimension (2 or 3)")
    
    # Common arguments
    parser.add_argument("--n-components", type=int, default=30, help="Number of PCA components")
    
    # Evaluation arguments
    parser.add_argument("--save-dir", type=str, default="results/ctms_general/json",
                        help="Save directory for evaluation results")
    
    args = parser.parse_args()
    
    if args.mode == "tune":
        tuning_pipeline_driver(
            k_list=args.k_list,
            n_components=args.n_components
        )
    elif args.mode == "plot":
        plotting_pipeline_driver(
            k=args.k,
            dim=args.dim,
            n_components=args.n_components
        )
    elif args.mode == "evaluate":
        evaluate_metadata(
            k=args.k,
            n_components=args.n_components,
            save_dir=args.save_dir
        )
        
    elif args.mode == "javi":
        javis_awful_heatmap_driver(
            k=args.k,
            n_components=args.n_components,
            save_dir="results/ctms_general/figs"
        )

if __name__ == "__main__":
    main()