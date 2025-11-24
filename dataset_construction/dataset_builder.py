from dataclasses import dataclass

import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger


# audio processing
import torch
import scipy.signal
import soundfile as sf

# models
from msclap import CLAP
from transformers import (
    ASTFeatureExtractor,
    ASTModel,
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)

import pretty_midi


@dataclass
class DatasetBuilder_02807:
    """
    Builds the dataset for 02807, group 44.

    Encapsulates the midi + metadata -> audio dataset building logic.

    Requires:
        metadata path: E.g., data/metadata/...
        midi root: E.g., data/midi/...
        wav root: E.g., data/wav/...

    The roots are recursively searched for files.
    """

    # paths
    metadata_path: str
    midi_root: str
    wav_root: str
    metadata_df: pd.DataFrame = None

    def __post_init__(self):
        # load metadata
        try:
            self.metadata_df = pd.read_csv(self.metadata_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load metadata from {self.metadata_path}: {e}")

    # ---- main data extraction functions ---

    def build_sequence_dataset(
        self,
        out_dir: str,
        add_pauses: bool = True,
        silence_threshold: float = 0.1,
        max_files: int | None = None,
    ):
        """
        For all midi files in midi_root:
            - load midi
            - extract note sequence
            - append to dataframe along with metadata

        Save the resulting dataframe to out_dir/sequence_dataset.parquet

        Args:
            out_dir: directory to save the parquet file
            add_pauses: if True, insert pause codes between notes
            silence_threshold: minimum silence (seconds) to insert a pause
            max_files: if set, limit to this many files
        """
        data = []
        failures = []

        midi_root_path = Path(self.midi_root)

        midi_files = list(midi_root_path.rglob("*.mid"))
        
        if max_files is not None:
            midi_files = midi_files[:max_files]

        for midi_path in tqdm(midi_files, desc="Building sequence dataset"):
            rel_path = midi_path.relative_to(midi_root_path)
            # load midi
            try:
                midi_data = self._load_midi(str(midi_path))
            except Exception as e:
                logger.error(f"Failed to load MIDI {midi_path}: {e}")
                failures.append(str(midi_path))
                continue

            
            # extract note sequence
            try:
                note_sequence = self._extract_note_sequence(
                    midi_data,
                    add_pauses=add_pauses,
                    silence_threshold=silence_threshold,
                )
            except Exception as e:
                logger.error(f"Failed to extract note sequence from {midi_path}: {e}")
                failures.append(str(midi_path))
                continue
            
            # NOTE: Unsure about this

            # get corresponding metadata
            metadata_row = self.metadata_df[self.metadata_df['filename'] == rel_path.name]

            if metadata_row.empty:
                logger.warning(f"No metadata found for {rel_path.name}, skipping.")
                continue

            metadata_dict = metadata_row.to_dict(orient='records')[0]

            # append to data
            data.append({
                "note_sequence": note_sequence,
                **metadata_dict,
            })

    def build_embedding_dataset(
        self,
        out_dir: str,
        max_files: int | None = None,
    ):
        """
        For all wav files in wav_root:
            - load audio
            - extract embeddings (CLAP, AST, WavLM)
            - append to dataframe along with metadata
        Save the resulting dataframe to out_dir/embedding_dataset.parquet

        Args:
            out_dir: directory to save the parquet file
            max_files: if set, limit to this many files
        """
        
        data = []
        failures = []

        wav_root_path = Path(self.wav_root)

        wav_files = list(wav_root_path.rglob("*.wav"))

        if max_files is not None:
            wav_files = wav_files[:max_files]

        # load models
        clap_model = self._load_clap()
        ast_extractor, ast_model = self._load_ast()
        wavlm_extractor, wavlm_model = self._load_wavlm()

        for wav_path in tqdm(wav_files, desc="Building embedding dataset"):
            rel_path = wav_path.relative_to(wav_root_path)

            # load audio
            try:
                y, sr = self._load_audio(str(wav_path), target_sr=48_000, duration=30.0)
                y_16, sr_16 = self._load_audio(str(wav_path), target_sr=16_000, duration=30.0)
            except Exception as e:
                logger.error(f"Failed to load audio {wav_path}: {e}")
                failures.append(str(wav_path))
                continue

            # extract embeddings
            try:
                ast_emb = self._get_ast_embedding(ast_extractor, ast_model, y_16, sr_16)
                wavlm_emb = self._get_wavlm_embedding(wavlm_extractor, wavlm_model, y_16, sr_16)
                clap_emb = self._get_clap_embedding(clap_model, y, sr)
            except Exception as e:
                logger.error(f"Failed to extract embeddings from {wav_path}: {e}")
                failures.append(str(wav_path))
                continue
            
            # NOTE: Unsure about this
            # get corresponding metadata
            metadata_row = self.metadata_df[self.metadata_df['filename'] == rel_path.with_suffix('.mid').name]

            if metadata_row.empty:
                logger.warning(f"No metadata found for {rel_path.name}, skipping.")
                continue

            metadata_dict = metadata_row.to_dict(orient='records')[0]

            # append to data
            data.append({
                "ast": ast_emb,
                "wavlm": wavlm_emb,
                "clap": clap_emb,
                **metadata_dict,
            })
        # save to parquet
        out_path = Path(out_dir) / "embedding_dataset.parquet"
        df = pd.DataFrame(data)
        df.to_parquet(out_path)
        logger.info(f"Saved embedding dataset to {out_path}")


    # ---- Data Loading ----

    def _load_audio(
        self,
        path: str,
        target_sr: int = 48_000,
        mono: bool = True,
        duration: float | None = 30.0,
    ) -> tuple[np.ndarray, int]:
        """
        Load audio with libsndfile and resample with scipy.

        Args:
            path: path to .wav file
            target_sr: target sampling rate (Hz)
            mono: if True, average stereo to mono
            duration: max duration in seconds (None = full file)

        Returns:
            (audio, sr):
                audio: float32 numpy array, shape (num_samples,)
                sr:    int, sampling rate (== target_sr)
        """
        audio, sr = sf.read(path, always_2d=False)

        if audio.ndim == 2 and mono:
            audio = audio.mean(axis=1)

        if duration is not None:
            max_samples = int(duration * sr)
            audio = audio[:max_samples]

        if sr != target_sr:
            audio = scipy.signal.resample_poly(audio, target_sr, sr)

        return audio.astype(np.float32), target_sr
    
    def _load_midi(self, path: str):
        """
        Load MIDI file using pretty_midi.

        Args:
            path: path to .mid file
        Returns:
            pretty_midi.PrettyMIDI instance
        """
        midi_data = pretty_midi.PrettyMIDI(path)
        return midi_data


    # ---- Model Loading ----

    def _load_clap(self) -> CLAP:
        """Load CLAP model from msclap."""

        logger.info("Loading CLAP (msclap)...")
        m = CLAP(version="2023", use_cuda=False)
        return m
    
    def _load_ast(self) -> ASTModel:
        """Load AST model from transformers."""

        logger.info("Loading AST...")
        feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        logger.info("AST loaded.")
        return feature_extractor, model

    def _load_wavlm(self) -> Tuple[Wav2Vec2FeatureExtractor, WavLMModel]:
        """Load WavLM model from transformers."""
        logger.info("Loading WavLM (microsoft/wavlm-base-plus)...")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus"
        )
        model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus"
        ).eval()
        logger.info("WavLM loaded.")
        return extractor, model
    

    # ---- generate embeddings ----

    def _get_clap_embedding(clap_model: CLAP, y: np.ndarray, sr: int = 48_000) -> np.ndarray:
        """
        CLAP embedding (msclap).

        Args:
            clap_model: CLAP instance
            y: waveform, shape (num_samples,)
            sr: sampling rate of y

        Returns:
            embedding: shape (1024,)
        """
        target_sr = clap_model.args.sampling_rate  # 48000

        if sr != target_sr:
            y = scipy.signal.resample_poly(y, target_sr, sr).astype(np.float32)

        audio = torch.from_numpy(y).float().unsqueeze(0)  # (1, samples)

        with torch.no_grad():
            emb = clap_model.clap.audio_encoder(audio)[0]  # (1, 1024)

        return emb.squeeze(0).cpu().numpy()
    
    
    def _get_ast_embedding(extractor, model, y, sr):
        """
        AST requires 16 kHz audio.
        Resample if needed.
        """
        target_sr = 16000
        
        if sr != target_sr:
            y16 = scipy.signal.resample_poly(y, target_sr, sr).astype(np.float32)
        else:
            y16 = y

        inputs = extractor(y16, sampling_rate=target_sr, return_tensors="pt")

        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state.mean(dim=1)

        return emb.squeeze(0).cpu().numpy()


    def _get_wavlm_embedding(
        self,
        wavlm_extractor: Wav2Vec2FeatureExtractor,
        wavlm_model: WavLMModel,
        y: np.ndarray,
        sr: int = 48_000,
    ) -> np.ndarray:
        """
        WavLM embedding.

        Args:
            wavlm_extractor: WavLM feature extractor
            wavlm_model:     WavLM model
            y:               audio samples, shape (num_samples,)
            sr:              sampling rate (Hz)

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        inputs = wavlm_extractor(
            y, sampling_rate=sr, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = wavlm_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        return embedding

    # ---- note sequences ----

    def _extract_note_sequence(
        self,
        midi: pretty_midi.PrettyMIDI,
        add_pauses: bool = True,
        silence_threshold=0.1,
        pause_code: int = 128,
        num_notes: int = 129,
    ) -> np.ndarray:
        
        """Return a list of MIDI note pitches from the file (0-127) + pauses (128)."""

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
            
            if add_pauses and (start - prev_end) > silence_threshold:
                seq.append(pause_code)
            
            seq.append(notes[i][2])
    
        return np.array(seq)





