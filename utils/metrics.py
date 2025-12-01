import ast
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from numpy.typing import NDArray
from typing import Dict, List, Any
from collections import Counter


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
        summary[c_key] = {}
        
        for col in columns:
            subset = df[df["cluster_label"] == c]

            if subset.empty:
                summary[c_key][col] = {}
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
            summary[c_key][col] = counts

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=4)

    return None