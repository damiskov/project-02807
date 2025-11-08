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


# for testing
if __name__ == "__main__":
    from load import load_dataset

    matrices, meta = load_dataset("../data/ctms", "../data/metadata/movies_metadata.csv")
    # apply the frobenius norm to each matrix and add as a new column
    meta['frobenius_norm'] = matrices['matrix'].apply(frobenius_norm)
    print(meta[['title', 'frobenius_norm']].head())
