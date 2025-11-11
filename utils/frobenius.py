import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
from typing import List


def create_frobenius_adjacency_matrix(
    ctms: List[NDArray]
) -> np.ndarray:
    """
    Given a list of CTMs (Chord trajectory matrices),

    this function creates a Frobenius adjacency matrix.
    
    Args:
        ctms (List[NDArray]): List of chord trajectory matrices.
    
    Returns:
        np.ndarray: Frobenius adjacency matrix where each entry (i, j) represents the
                    Frobenius norm of the difference between the CTMs of pieces i and j.
    """

    n_pieces = len(ctms)
    frobenius_matrix = np.zeros((n_pieces, n_pieces))
    
    for i in tqdm(range(n_pieces), desc="Calculating Frobenius graph"):
        for j in range(n_pieces):
            if i != j:
                diff_matrix = ctms[i] - ctms[j]
                frobenius_matrix[i, j] = frobenius_norm(diff_matrix)
                
    return frobenius_matrix

def frobenius_norm(matrix):
    """Calculate the Frobenius norm of a matrix."""
    return sum(sum(cell**2 for cell in row) for row in matrix) ** 0.5

def masked_cosine_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute a masked cosine distance between two matrices.
    Zeros in both matrices are ignored.
    
    Returns a distance in [0, 1], smaller = more similar.
    """
    # Mask: True where at least one entry is non-zero
    mask = (A != 0) | (B != 0)
    
    if not np.any(mask):
        # Both matrices completely zero → maximal distance
        return 1.0

    # Flatten only the relevant entries
    a_flat = A[mask].flatten()
    b_flat = B[mask].flatten()

    # Cosine similarity
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    
    # Handle degenerate norms
    if norm_a == 0 or norm_b == 0:
        return 1.0

    sim = np.dot(a_flat, b_flat) / (norm_a * norm_b)
    
    # Convert to distance
    dist = 1 - sim  # smaller = more similar

    return dist

def create_masked_cosine_adjacency(ctms: List[np.ndarray]) -> np.ndarray:
    """
    Compute adjacency matrix using masked cosine distance for qualitative comparison.
    """
    n = len(ctms)
    adjacency = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="Computing masked cosine adjacency"):
        for j in range(i+1, n):
            dist = masked_cosine_distance(ctms[i], ctms[j])
            adjacency[i, j] = dist
            adjacency[j, i] = dist
    return adjacency

def kl_divergence_masked(P: np.ndarray, Q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Symmetric KL divergence ignoring shared zeros.
    """
    # Mask to ignore entries where both are zero
    mask = (P != 0) | (Q != 0)
    P = P[mask]
    Q = Q[mask]
    
    # Smooth and normalize
    P = np.maximum(P, eps)
    Q = np.maximum(Q, eps)
    P /= P.sum()
    Q /= Q.sum()

    kl_PQ = np.sum(P * np.log(P / Q))
    kl_QP = np.sum(Q * np.log(Q / P))
    return 0.5 * (kl_PQ + kl_QP)


def create_kl_adjacency(ctms: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise symmetric KL divergence between CTMs.
    """
    n = len(ctms)
    adjacency = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing KL divergence adjacency"):
        for j in range(i + 1, n):
            dist = kl_divergence_masked(ctms[i], ctms[j])
            adjacency[i, j] = dist
            adjacency[j, i] = dist

    return adjacency

# for testing
if __name__ == "__main__":
    
    from utils.load import load_dataset

    matrices, meta = load_dataset()
    # apply the frobenius norm to each matrix and add as a new column
    meta['frobenius_norm'] = matrices['matrix'].apply(frobenius_norm)
    print(meta[['composer', 'frobenius_norm']].head())