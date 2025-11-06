"""
Internal utilities for generating MinHash signatures and computing similarities.
"""
import numpy as np


def make_hash_functions(
    k: int,
    max_val: int = 2**32 - 1,
    seed: int = 42
) -> callable:
    """Generate k different hash functions of the form h(x) = (a*x + b) % p."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, max_val, size=k, dtype=np.uint64)
    b = rng.integers(0, max_val, size=k, dtype=np.uint64)

    def hash_family(x):
        x = np.uint64(x)
        return (a * x + b) % max_val

    return hash_family

def shingle_to_int(shingle, base=131) -> int:
    """Encode a shingle (list of ints) into a single integer."""
    h = 0
    for val in shingle:
        h = h * base + val
    return h

def compute_minhash(shingles, k=100, seed=42) -> np.ndarray:
    """
    Compute k MinHash values from a set of shingles.
    
    Args:
        shingles (list[np.ndarray]): List of shingles (subsequences).
        k (int): Number of hash functions / MinHash values to compute.
        seed (int): Random seed for hash function generation.
    Returns:
        np.ndarray: Array of k MinHash values. (MinHash signature)
    """

    # convert shingles to integer IDs    
    shingle_ids = [shingle_to_int(s) for s in shingles]
    if not shingle_ids:
        return np.zeros(k, dtype=np.uint64)
    
    # create k hash functions
    hash_functions = make_hash_functions(k, seed=seed)

    # compute min hash per function
    minhashes = np.full(k, np.inf)
    for x in shingle_ids:
        hashes = hash_functions(x)
        minhashes = np.minimum(minhashes, hashes)

    return minhashes.astype(np.uint64)


def pairwise_minhash_similarity(signatures: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of pairwise Jaccard similarity estimates from MinHash signatures.
    Args:
        signatures (np.ndarray): Array of shape (N, k), where
                                 N = number of sequences,
                                 k = number of minhash values per sequence.

    Returns:
        np.ndarray: (N, N) similarity matrix with values in [0, 1].
    """
    N, k = signatures.shape

    # expand dimensions for broadcasting
    A = signatures[:, np.newaxis, :]  #  (N, 1, k)
    B = signatures[np.newaxis, :, :]  #  (1, N, k)

    # elementwise comparison across k and count matches
    matches = (A == B).sum(axis=2)    # shape (N, N)
    similarity = matches / k          # normalize to [0,1]
    return similarity