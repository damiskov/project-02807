# """
# Convert MIDI files to WAV format using FluidSynth CLI.

# NOTE: This requires FluidSynth to be installed on your system.
# """

# import os
# from pathlib import Path
# from midi2audio import FluidSynth

# import subprocess
# from pathlib import Path

# def convert_midi_cli(midi_path, wav_path, soundfont_path):
#     wav_path.parent.mkdir(parents=True, exist_ok=True)

#     cmd = [
#         "fluidsynth",
#         "-a", "file",
#         "-F", str(wav_path),
#         "-r", "44100",
#         "-o", "synth.polyphony=4096",
#         soundfont_path,
#         str(midi_path)
#     ]


#     subprocess.run(cmd, check=True)


# def convert_tree_cli(midi_root, wav_root, soundfont):
#     midi_root = Path(midi_root)
#     wav_root = Path(wav_root)

#     for midi_path in midi_root.rglob("*.mid"):
#         rel = midi_path.relative_to(midi_root)
#         wav_path = wav_root / rel.with_suffix(".wav")

#         if wav_path.exists():
#             print(f"[SKIP] {wav_path}")
#             continue

#         print(f"[CONVERT] {midi_path} -> {wav_path}")

#         try:
#             convert_midi_cli(midi_path, wav_path, soundfont)
#         except Exception as e:
#             print(f"[ERROR] {midi_path}: {e}")



# # NOTE: change paths as needed
# if __name__ == "__main__":
#     midi_root = "dataset_construction/data/midi"
#     wav_root = "dataset_construction/data/wav"
#     soundfont_path = "dataset_construction/data/soundfonts/FluidR3_GM.sf2"

#     convert_tree_cli(midi_root, wav_root, soundfont_path)

import os
import subprocess
from pathlib import Path


def convert_midi_cli(midi_path, wav_path, soundfont_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp
    tmp_path = wav_path.with_suffix(".tmp.wav")

    cmd = [
        "fluidsynth",
        "-a", "file",
        "-F", str(tmp_path),
        "-r", "44100",
        "-o", "synth.polyphony=4096",
        soundfont_path,
        str(midi_path)
    ]

    subprocess.run(cmd, check=True)

    # rename temp only if successful
    tmp_path.replace(wav_path)


def wav_is_valid(wav_path: Path) -> bool:
    """Check whether a WAV file is valid and non-empty."""
    return wav_path.exists() and wav_path.stat().st_size > 0


def convert_tree_cli(midi_root, wav_root, soundfont):
    midi_root = Path(midi_root)
    wav_root = Path(wav_root)

    converted = 0
    skipped = 0

    for midi_path in midi_root.rglob("*.mid"):
        rel = midi_path.relative_to(midi_root)
        wav_path = wav_root / rel.with_suffix(".wav")

        # Skip if WAV already exists and is valid
        if wav_is_valid(wav_path):
            print(f"[SKIP] {wav_path} (already exists)")
            skipped += 1
            continue

        print(f"[CONVERT] {midi_path} -> {wav_path}")

        try:
            convert_midi_cli(midi_path, wav_path, soundfont)
            converted += 1
        except Exception as e:
            print(f"[ERROR] {midi_path}: {e}")

    print(f"\nDONE: {converted} converted, {skipped} skipped.")


# NOTE: change paths as needed
if __name__ == "__main__":
    midi_root = "dataset_construction/data/midi"
    wav_root = "dataset_construction/data/wav"
    soundfont_path = "dataset_construction/data/soundfonts/FluidR3_GM.sf2"

    convert_tree_cli(midi_root, wav_root, soundfont_path)
