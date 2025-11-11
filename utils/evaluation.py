"""
Evaluation utilities for clustering models.
- Silhouette Score
- Modularity Score
"""
import numpy as np

# --- Silhouette Score ---
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    In-house implementation of the Silhouette Score.

    Args:
        X (np.ndarray): Data points (n_samples, n_features).
        labels (np.ndarray): Cluster labels for each data point.

    Returns:
        float: Mean silhouette score in [-1, 1].
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters == n_samples:
        return 0.0  # Undefined silhouette score

    # Compute pairwise distance matrix
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    s = np.zeros(n_samples)
    for i in range(n_samples):
        own_cluster = labels[i]

        # Intra-cluster distances (exclude self)
        in_mask = (labels == own_cluster)
        in_mask[i] = False
        a_i = np.mean(dist_matrix[i, in_mask]) if np.any(in_mask) else 0.0

        # Nearest other cluster mean distance
        b_i = np.min([
            np.mean(dist_matrix[i, labels == other])
            for other in unique_labels if other != own_cluster
        ])

        s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(s))



def compute_modularity_score(
    adjacency_matrix: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute the modularity score of a clustering given the adjacency matrix.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the graph.
        labels (np.ndarray): Cluster labels for each node.

    Returns:
        float: Modularity score.
    """
    m = np.sum(adjacency_matrix) / 2
    unique_labels = np.unique(labels)
    Q = 0.0

    degrees = np.sum(adjacency_matrix, axis=1)

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        for i in cluster_indices:
            for j in cluster_indices:
                A_ij = adjacency_matrix[i, j]
                k_i = degrees[i]
                k_j = degrees[j]
                Q += A_ij - (k_i * k_j) / (2 * m)

    Q /= (2 * m)
    return Q