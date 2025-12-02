import ast
import json
import textwrap

import numpy as np
import pandas as pd
from tqdm import tqdm

from loguru import logger

from typing import Dict, List, Any
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt


def kl_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute symmetric KL divergence between two matrices without masking.

    All entries contribute, including zeros.

    Parameters:
    - P, Q: np.ndarray, same shape
    - eps: small constant to avoid log(0)

    Returns:
    - float: symmetric KL divergence (0 = identical, larger = more different)
    """
    # Flatten matrices into 1D vectors
    P_flat = P.flatten()
    Q_flat = Q.flatten()

    # Smooth to avoid log(0)
    P_flat = np.maximum(P_flat, eps)
    Q_flat = np.maximum(Q_flat, eps)

    # Normalize to valid probability distributions (already sum to 1 if CTMs)
    # P_flat /= P_flat.sum()
    # Q_flat /= Q_flat.sum()

    # Compute KL divergences
    kl_PQ = np.sum(P_flat * np.log(P_flat / Q_flat))
    kl_QP = np.sum(Q_flat * np.log(Q_flat / P_flat))

    # Symmetric KL
    return 0.5 * (kl_PQ + kl_QP)

def create_kl_adjacency(ctms: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise symmetric KL divergence between CTMs without masking.

    Parameters:
    - ctms: list of np.ndarray, each representing a CTM

    Returns:
    - np.ndarray: symmetric adjacency matrix of shape (n, n)
    """
    n = len(ctms)
    adjacency = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing KL adjacency (unmasked)"):
        for j in range(i + 1, n):
            dist = kl_divergence(ctms[i], ctms[j])
            adjacency[i, j] = dist
            adjacency[j, i] = dist  # ensure symmetry

    return adjacency

def cosine_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute standard cosine distance between two matrices (no masking).
    Returns a distance in [0, 1], smaller = more similar.
    """
    a_flat = A.flatten()
    b_flat = B.flatten()

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    # Handle degenerate norms (all zeros)
    if norm_a == 0 or norm_b == 0:
        return 1.0

    sim = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    dist = 1 - sim  # smaller = more similar
    return dist

def create_cosine_adjacency(ctms: List[np.ndarray]) -> np.ndarray:
    """
    Compute adjacency matrix using standard cosine distance (no mask).
    """
    n = len(ctms)
    adjacency = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="Computing cosine adjacency"):
        for j in range(i+1, n):
            dist = cosine_distance(ctms[i].ravel(), ctms[j].ravel())  # flatten arrays if needed
            adjacency[i, j] = dist
            adjacency[j, i] = dist
    return adjacency


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Davies-Bouldin index.
    Returns np.nan if invalid clustering.
    """
    # unique clusters
    cluster_ids = np.unique(labels)
    K = len(cluster_ids)

    if K < 2:
        return np.nan

    # compute centroids 
    centroids = []
    for c in cluster_ids:
        pts = X[labels == c]
        centroids.append(np.mean(pts, axis=0))
    centroids = np.vstack(centroids)

    # compute intra-cluster scatter S_i
    S = np.zeros(K)
    for idx, c in enumerate(cluster_ids):
        pts = X[labels == c]
        if len(pts) == 1:
            S[idx] = 0.0
        else:
            dists = np.linalg.norm(pts - centroids[idx], axis=1)
            S[idx] = np.mean(dists)

    # compute pairwise centroid distances M_ij
    # Shape: (K, K)
    diff = centroids[:, None, :] - centroids[None, :, :]
    M = np.linalg.norm(diff, axis=2)

    # prevent divide-by-zero (identical centroids)
    M[M == 0] = 1e-12

    # compute R_ij = (S_i + S_j) / M_ij
    S_i_j = S[:, None] + S[None, :]
    R = S_i_j / M

    # compute D_i = max_{j != i} R_ij
    np.fill_diagonal(R, -np.inf)  # ignore diagonal
    D = np.max(R, axis=1)

    return float(np.mean(D))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score.
    """
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return -1.0

    n = X.shape[0]

    # pairwise distance matrix
    dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    silhouettes = np.zeros(n)

    for k in unique_labels:
        cluster_idx = np.where(labels == k)[0]
        other_idx = np.where(labels != k)[0]

        for i in cluster_idx:
            # a(i): mean intra-cluster distance 
            if len(cluster_idx) > 1:
                a = np.mean(dmat[i, cluster_idx][cluster_idx != i]) # exclude self
            else:
                a = 0.0

            # b(i): mean distance to points of nearest other cluster
            b = np.inf
            for k2 in unique_labels:
                if k2 == k:
                    continue
                idx2 = np.where(labels == k2)[0]
                b = min(b, np.mean(dmat[i, idx2]))

            silhouettes[i] = (b - a) / max(a, b)

    return float(np.mean(silhouettes))



def column_wise_summary(
    df: pd.DataFrame,
    columns: List[str],
    save_path: str,
) -> None:
    """
    For each cluster and each column:
    - Count occurrences of each unique element.
    - Fully supports list-like columns (themes, keywords, companies).
    - Saves a JSON summary of value frequencies.

    Output format:
    {
        "0": {
            "themes": {"Action": 10, "Sci-fi": 4, ...},
            "keywords": {...},
            ...
        },
        "1": { ... },
        ...
    }
    """
    
    clusters = sorted(df["cluster_label"].unique())

    summary: Dict[Any, Dict[str, Dict[str, int]]] = {}

    for c in clusters:
        # Convert cluster label to JSON-serializable type
        c_key = str(int(c)) if isinstance(c, (np.integer, np.floating)) else str(c)
        subset = df[df["cluster_label"] == c]
        cluster_num = len(subset)
        
        summary[c_key] = {
            col: {} for col in columns
        }
        summary[c_key]["cluster_size"] = cluster_num
        
        for col in columns:
            logger.debug(f"Processing cluster {c_key}, column '{col}'…")
            subset = df[df["cluster_label"] == c]

            # Ensure the nested dict for this column exists
            if col not in summary[c_key]:
                summary[c_key][col] = {}

            if subset.empty:
                logger.warning(f"Cluster {c_key} has no entries for column '{col}'.")
                # Keep as empty dict for empty subsets
                continue
            

            # Flatten list values OR handle scalar values
            flat_values = []
            for item in subset[col]:
                if isinstance(item, str):
                    # Try to parse JSON string
                    try:
                        parsed = ast.literal_eval(item)
                        if isinstance(parsed, list):
                            flat_values.extend(parsed)
                        else:
                            flat_values.append(parsed)
                    except (ValueError, SyntaxError):
                        # Not JSON, treat as plain string
                        flat_values.append(item)
                elif isinstance(item, list):
                    flat_values.extend(item)
                else:
                    flat_values.append(item)

            counts = dict(Counter(flat_values))
            # Convert keys and values to JSON-serializable types
            counts = {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, np.integer))}
            # Sort by count (descending)
            counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
            # Store counts under a dedicated key to avoid overwriting
            summary[c_key][col]["counts"] = counts

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=4)

    return None


def plot_merged_attribute_heatmap(
    df,
    labels,
    save_path,
    top_n=5,
):
    """
    Merges 'themes', 'keywords', 'involved_companies', and 'game' into a single 
    analysis pool and plots distinctive terms per cluster.
    """
    local_df = df.copy()
    local_df['cluster'] = labels
    
    # Define the columns we want to merge
    # Ensure these exist in your dataframe; remove any that don't
    target_cols = ["themes", "keywords", "involved_companies", "game"]
    valid_cols = [c for c in target_cols if c in local_df.columns]
    
    # --- STEP 1: NORMALIZE EVERYTHING TO LISTS ---
    for col in valid_cols:
        # Fill NaNs with empty list representation
        local_df[col] = local_df[col].fillna("[]")
        
        def standardize_to_list(val):
            try:
                # Case A: It's already a list -> Keep it
                if isinstance(val, list):
                    return val
                
                # Case B: It's a string
                if isinstance(val, str):
                    val = val.strip()
                    # Case B1: Stringified list "['Action', 'FPS']" -> Convert to list
                    if val.startswith('[') and val.endswith(']'):
                        return ast.literal_eval(val)
                    # Case B2: Regular string "Halo 3" -> Wrap in list ['Halo 3']
                    # Check if empty string
                    if val == "" or val == "[]":
                        return []
                    return [val]
                
                # Case C: Other (numbers, etc) -> Wrap in list
                return [str(val)]
            except:
                return []

        local_df[col] = local_df[col].apply(standardize_to_list)

    # --- STEP 2: MERGE COLUMNS ---
    # Now that all columns are actual lists, we can sum them row-wise
    # This concatenates ['Halo'] + ['Action'] -> ['Halo', 'Action']
    local_df['all_tags'] = local_df[valid_cols].sum(axis=1)

    # --- STEP 3: EXPLODE ---
    exploded_df = local_df.explode('all_tags')
    
    # Clean up: remove short garbage strings or numbers that might have slipped in
    exploded_df['all_tags'] = exploded_df['all_tags'].astype(str)
    exploded_df = exploded_df[exploded_df['all_tags'].str.len() > 2]

    # --- STEP 4: CALCULATE METRICS (LIFT) ---
    global_counts = exploded_df['all_tags'].value_counts(normalize=True)
    unique_clusters = sorted(local_df['cluster'].unique())
    
    lift_grid = np.zeros((len(unique_clusters), top_n))
    word_grid = np.full((len(unique_clusters), top_n), "", dtype=object)
    
    for i, c in enumerate(unique_clusters):
        c_counts = exploded_df[exploded_df['cluster'] == c]['all_tags'].value_counts(normalize=True)
        
        # Calculate Lift
        lift = c_counts / global_counts.reindex(c_counts.index).fillna(1)
        
        top_k = lift.sort_values(ascending=False).head(top_n)
        
        for rank, (word, score) in enumerate(top_k.items()):
            if rank < top_n:
                lift_grid[i, rank] = score
                # Wrap text for heatmap
                clean_word = textwrap.fill(str(word), width=12) 
                word_grid[i, rank] = clean_word

    # --- STEP 5: PLOT ---
    plt.figure(figsize=(14, len(unique_clusters) * 1.4))
    
    ax = sns.heatmap(
        lift_grid, 
        annot=word_grid, 
        fmt="", 
        cmap="Blues", 
        linewidths=1.0,
        linecolor='white',
        yticklabels=[f"Cluster {c+1}" for c in unique_clusters],
        xticklabels=[f"Rank {i+1}" for i in range(top_n)],
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"orientation": "horizontal", "pad": 0.05, "aspect": 40, "label": "Lift Score (Distinctiveness)"}
    )
    
    ax.set_title("Most Distinctive Attributes (Themes, Keywords, Games, Companies) by Cluster", fontsize=16, pad=20)
    plt.yticks(rotation=0)
    ax.xaxis.tick_top()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.tight_layout()
    plt.show()