"""
Plotting raw midi data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from utils.load import load_sequences

def plot_raw_midi_sequence(sequence: np.ndarray, title: str = "Raw MIDI Sequence"):
    """Plot a raw MIDI sequence as a piano roll."""
    plt.figure(figsize=(12, 6))
    plt.imshow(sequence.T, aspect='auto', cmap='gray_r', origin='lower')
    plt.colorbar(label='Velocity')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('MIDI Note Number')
    plt.yticks(ticks=np.arange(0, 128, 12), labels=np.arange(0, 128, 12))
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    
    sequences_df = load_sequences(base_dir="data/sequences", file_extension=".npy", metadata="data/metadata/movies_metadata.csv")
    logger.info(f"Loaded {len(sequences_df)} sequences.")

    # Plot a random sample of 3 sequences
    sample_sequences = sequences_df.sample(n=3, random_state=42)
    for idx, row in sample_sequences.iterrows():
        plot_raw_midi_sequence(row['sequence'], title=f"Raw MIDI Sequence: {row.get('title', 'Unknown Title')}")