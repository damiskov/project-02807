"""
Functions for generating shingles from the musical sequences.
"""
import numpy as np

from utils.load import load_sequences

from typing import Optional


def generate_shingles(sequence: np.ndarray, shingle_size: int = 5) -> list[np.ndarray]:
    """
    Generate shingles (subsequences) from a musical sequence.

    Args:
        sequence (np.ndarray): The input musical sequence.
        shingle_size (int): The size of each shingle.

    Returns:
        list[np.ndarray]: A list of shingles.
    """
    shingles = []
    for i in range(len(sequence) - shingle_size + 1):
        shingle = sequence[i:i + shingle_size]
        shingles.append(shingle)
    return shingles



def test_shingles():
    sequences_df = load_sequences("data/sequences")[:10]
    for _, row in sequences_df.iterrows():
        shingles = generate_shingles(row['sequence'])
        print(f"Shingles for {row['piece_name']}:\n{shingles}")
    sequences_df = load_sequences("data/sequences")
    print(sequences_df.head())

if __name__=="__main__":
    test_shingles()
