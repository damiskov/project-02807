"""
Convert MIDI files to WAV format using FluidSynth CLI.

NOTE: This requires FluidSynth to be installed on your system.
"""

import os
from pathlib import Path
from midi2audio import FluidSynth

import subprocess
from pathlib import Path

def convert_midi_cli(midi_path, wav_path, soundfont_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fluidsynth",
        "-a", "file",
        "-F", str(wav_path),
        "-r", "44100",
        soundfont_path,
        str(midi_path)
    ]

    subprocess.run(cmd, check=True)


def convert_tree_cli(midi_root, wav_root, soundfont):
    midi_root = Path(midi_root)
    wav_root = Path(wav_root)

    for midi_path in midi_root.rglob("*.mid"):
        rel = midi_path.relative_to(midi_root)
        wav_path = wav_root / rel.with_suffix(".wav")

        if wav_path.exists():
            print(f"[SKIP] {wav_path}")
            continue

        print(f"[CONVERT] {midi_path} -> {wav_path}")

        try:
            convert_midi_cli(midi_path, wav_path, soundfont)
        except Exception as e:
            print(f"[ERROR] {midi_path}: {e}")



# NOTE: change paths as needed
if __name__ == "__main__":
    midi_root = "sandbox/data/midi"
    wav_root = "sandbox/data/wav"
    soundfont_path = "sandbox/soundfonts/FluidR3_GM.sf2"

    convert_tree_cli(midi_root, wav_root, soundfont_path)
