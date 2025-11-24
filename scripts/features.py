"""
Generates features (chord adjacency matrix) from midi files.
"""

import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import logging

PAUSE_CODE = 128
NUM_NOTES = 129

def extract_note_sequence(midi_path: Path, add_pauses=True, silence_thresh=0.05):
    """Return a list of MIDI note pitches from the file (0-127) + pauses (128)."""

    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = [(n.start, n.end, n.pitch)
             for inst in midi.instruments for n in inst.notes]
    # logger.debug(f"Notes extracted from {midi_path}: {notes[:10]}...")
    
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
    # logger.debug(f"Final note sequence for {midi_path}: {seq}")   
    return seq

def chord_trajectory_matrix(sequence):
    """Construct 129x129 chord trajectory matrix."""
    
    M = np.zeros((NUM_NOTES, NUM_NOTES), dtype=np.int32)
    
    for i in range(len(sequence) - 1):
    
        a, b = sequence[i], sequence[i + 1]
        M[a, b] += 1
    
    return M

def save_ctms(base_dir="data/midi_downloads", out_dir="data/features"):
    """
    Process videogame MIDIs organized by console/game directories to extract
    chord trajectory matrices (CTMs) and save them as .npy per videogame.
    
    Directory structure:
        base_dir/
            console1/
                gameA/
                    *.mid
                gameB/
            console2/
                gameC/
    
    CTMs for each MIDI are saved in:
        out_dir/game_name/*.npy
    """
    logger = logging.getLogger(__name__)

    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    records = []

    # Iterate over all consoles and games
    for console_dir in base_dir.iterdir():
        if not console_dir.is_dir():
            continue
        for game_dir in console_dir.iterdir():
            if not game_dir.is_dir():
                continue

            game_out_dir = out_dir / game_dir.name
            game_out_dir.mkdir(exist_ok=True, parents=True)

            for midi_path in tqdm(game_dir.glob("*.mid"), desc=f"Processing {game_dir.name}"):
                try:
                    seq = extract_note_sequence(midi_path)
                except Exception as e:
                    logger.error(f"Error processing {midi_path}: {e}")
                    continue

                if not seq:
                    logger.warning(f"Empty sequence for {midi_path}, skipping.")
                    continue

                M = chord_trajectory_matrix(seq)
                if M.sum() == 0:
                    logger.warning(f"Empty trajectory matrix for {midi_path}, skipping.")
                    continue

                # Use MIDI filename as ID
                midi_id = midi_path.stem
                np.save(game_out_dir / f"{midi_id}_traj.npy", M)

                records.append({
                    "id": midi_id,
                    "console": console_dir.name,
                    "game": game_dir.name,
                    "file": str(midi_path),
                    "length": len(seq),
                })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"Processed {len(df)} MIDI files -> saved summary to {out_dir}/summary.csv")

def save_sequences(base_dir="data/midis", out_dir="data/sequences"):
    """
    Process movie theme MIDIs to extract note sequences and save them as .npy.
    Filenames are based on IMDb ID (from MIDI filename stem).
    """
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    records = []

    for midi_path in tqdm(base_dir.rglob("*.mid"), desc="Processing MIDIs"):
        try:
            seq = extract_note_sequence(midi_path)
        except Exception as e:
            logger.error(f"Error processing {midi_path}: {e}")
            continue

        if not seq:
            logger.warning(f"Empty sequence for {midi_path}, skipping.")
            continue

        # Use IMDb ID from filename as output filename
        imdb_id = midi_path.stem
        np.save(out_dir / f"{imdb_id}_seq.npy", np.array(seq))

        records.append({
            "id": imdb_id,
            "file": str(midi_path),
            "length": len(seq),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "sequences_summary.csv", index=False)
    logger.success(f"Processed {len(df)} MIDI files -> saved to {out_dir}/sequences_summary.csv")



if __name__ == "__main__":
    save_sequences()
