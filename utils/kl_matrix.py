import numpy as np
from tqdm import tqdm
from typing import List


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
