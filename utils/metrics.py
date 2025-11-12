import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
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
