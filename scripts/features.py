"""
Generates features (chord adjacency matrix) from midi files.
"""

import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

PAUSE_CODE = 128
NUM_NOTES = 129

def extract_note_sequence(midi_path: Path, add_pauses=True, silence_thresh=0.05):
    """Return a list of MIDI note pitches from the file (0-127) + pauses (128)."""

    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = [(n.start, n.end, n.pitch)
             for inst in midi.instruments for n in inst.notes]
    logger.debug(f"Notes extracted from {midi_path}: {notes[:10]}...")
    
    if not notes:
        return []
    
    notes.sort(key=lambda x: x[0])
    seq = [notes[0][2]]
    
    for i in range(1, len(notes)):

        prev_end = notes[i - 1][1]
        start = notes[i][0]
        
        if add_pauses and (start - prev_end) > silence_thresh:
            seq.append(PAUSE_CODE)
        
        seq.append(notes[i][2])

    logger.debug(f"Final note sequence for {midi_path}: {seq}")
    
    return seq

def chord_trajectory_matrix(sequence):
    """Construct 129x129 chord trajectory matrix."""
    
    M = np.zeros((NUM_NOTES, NUM_NOTES), dtype=np.int32)
    
    for i in range(len(sequence) - 1):
    
        a, b = sequence[i], sequence[i + 1]
        M[a, b] += 1
    
    return M

def process_dataset(base_dir="data/midis", metadata="data/metadata/musicnet_metadata.csv", out_dir="data/features"):
    
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    meta = pd.read_csv(metadata)
    records = []

    for midi_path in tqdm(base_dir.rglob("*.mid")):
    
        composer = midi_path.parent.name
        try:
            seq = extract_note_sequence(midi_path)

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            continue

        if not seq:
            continue

       

        M = chord_trajectory_matrix(seq)
        if M.sum() == 0:
            logger.warning(f"Empty trajectory matrix for {midi_path}, skipping.")
        import sys; sys.exit()

        composer_dir = out_dir / composer.lower()
        composer_dir.mkdir(exist_ok=True)
        np.save(composer_dir / f"{composer.lower()}_{midi_path.stem}_traj.npy", M)

        records.append({
            "composer": composer,
            "file": str(midi_path),
            "length": len(seq),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "summary.csv", index=False)
    logger.success(f"Processed {len(df)} MIDI files -> saved to {out_dir}/summary.csv")

if __name__ == "__main__":
    process_dataset()
