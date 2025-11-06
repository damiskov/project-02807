"""
Internal utilities for generating MinHash signatures from shingles.
"""
import numpy as np


def make_hash_functions(k: int, max_val: int = 2**32 - 1, seed: int = 42):
    """Generate k different hash functions of the form h(x) = (a*x + b) % p."""
    rng = np.random.default_rng(seed)
    prime_large = int(2**32)
    a = rng.integers(1, prime_large, size=k, dtype=np.uint64)
    b = rng.integers(0, prime_large, size=k, dtype=np.uint64)

    def hash_family(x):
        x = np.uint64(x)
        return (a * x + b) % prime_large    

    return hash_family

def shingle_to_int(shingle, base=131):
    """Encode a shingle (list of ints) into a single integer."""
    h = 0
    for val in shingle:
        h = h * base + val
    return h

def compute_minhash(shingles, k=100, seed=42):
    """Compute k MinHash values from a set of shingles."""

    # convert shingles to integer IDs    
    shingle_ids = [shingle_to_int(s) for s in shingles]
    if not shingle_ids:
        return np.zeros(k, dtype=np.uint64)
    
    # create k hash functions
    hash_f = make_hash_functions(k, seed=seed)

    # compute min hash per function
    minhashes = np.full(k, np.inf)
    for x in shingle_ids:
        hashes = hash_f(x)
        minhashes = np.minimum(minhashes, hashes)

    return minhashes.astype(np.uint64)
