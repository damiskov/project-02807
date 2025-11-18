"""
Plotting raw midi data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from utils.load import load_metadata

from pretty_midi import PrettyMIDI

def load_raw_midi(
    file_path: str,
    fs: int = 1
) -> np.ndarray:
    """Load a raw MIDI sequence from a .midi file."""
    midi_data = PrettyMIDI(file_path)
    # Convert to piano roll representation
    piano_roll = midi_data.get_piano_roll(fs=fs)  # fs: frames per second
    return piano_roll
    

def plot_midi_roll(piano_roll: np.ndarray, title: str = "MIDI Piano Roll"):
    """Plot a piano roll representation of MIDI data."""
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('MIDI Note Number')
    plt.colorbar(label='Velocity')
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    metadata = load_metadata("data/metadata/movies_metadata.csv")
    path = "/Users/davidmiles-skov/Desktop/fall_25/02807_comp_tools/project-02807/data/midis/tt0080684.mid"
    piano_roll = load_raw_midi(path)
    plot_midi_roll(piano_roll, title="Piano Roll for tt0080684.mid")