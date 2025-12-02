from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
import re

# ======= updated loading functions =======


# --- metadata handling ---
def clean_text(s: str) -> str:
    """Clean text (metadata)"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def join_list(x):
    """Convert lists to space-joined strings and handle missing values."""
    if isinstance(x, list):
        return " ".join(map(str, x))
    return "" if pd.isna(x) else str(x)

def clean_metadata_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a specific metadata column in the DataFrame."""
    df["metadata"] = (
        df["name"].apply(join_list) + " " +
        df["themes"].apply(join_list) + " " +
        df["keywords"].apply(join_list) + " " +
        df["involved_companies"].apply(join_list)
    )
    df["metadata"] = df["metadata"].apply(clean_text)
    return df

def load_embeddings_dataset(path: str) -> pd.DataFrame:
    """Load embeddings CSV and preprocess certain columns."""
    df = pd.read_parquet(path)
    df = clean_metadata_column(df)
    return df
        

def load_ctms_dataset(path: str) -> pd.DataFrame:
    """Load CTM-based dataset"""
    df = pd.read_parquet(path)
    df["ctm"] = df["ctm"].apply(lambda x: np.stack(x, axis=0).astype(np.int32)) # need to convert list of arrays to single array
    df = clean_metadata_column(df)
    return df

# --- old shit ---

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load movie metadata CSV and keep IMDb rating only."""

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.index = df['id']
    df.sort_index(inplace=True)
    df.drop('id', axis=1, inplace=True)

    return df

def load_feature_matrices(base_dir: str | Path) -> pd.DataFrame:
    """
    Recursively load all .npy chord trajectory matrices for movie themes into a DataFrame.
    Each row: piece_id (IMDb), matrix (np.ndarray), path.
    """
    base_dir = Path(base_dir)
    records = []
    for npy_path in tqdm(base_dir.rglob("*.npy"), desc="Loading matrices"):
        # Use filename stem as IMDb ID
        piece_id = npy_path.stem.replace("_traj", "")
        
        try:
            matrix = np.load(npy_path)
            records.append({
                "id": piece_id,
                "matrix": matrix,
                "path": str(npy_path)
            })
        except Exception as e:
            logger.warning(f"Error loading {npy_path}: {e}")
            continue

    df = pd.DataFrame(records, index=[r['id'] for r in records])
    df.sort_index(inplace=True)
    df.index.name = 'id'
    logger.success(f"Loaded {len(df)} matrices from {base_dir}")
    return df

def load_dataset(features_dir="data/ctms", metadata_csv="data/metadata/movies_metadata.csv"):
    """Convenience wrapper to load both matrices and metadata."""
    matrices_df = load_feature_matrices(features_dir)
    metadata_df = load_metadata(metadata_csv)
    return matrices_df, metadata_df

def load_sequences(base_dir: str | Path = "data/sequences", file_extension: str = ".npy") -> pd.DataFrame:
    """
    Recursively load all musical sequences for movie themes from .npy files into a DataFrame.
    Each row: piece_id (IMDb), sequence (np.ndarray), path.
    """
    logger.info(f"Loading sequences from {base_dir} with extension {file_extension}")
    base_dir = Path(base_dir)
    records = []

    for npy_path in tqdm(base_dir.rglob(f"*{file_extension}"), desc="Loading sequences"):
        # Use filename stem as IMDb ID
        piece_id = npy_path.stem.replace("_seq", "")
        
        try:
            sequence = np.load(npy_path)
            records.append({
                "id": piece_id,
                "sequence": sequence,
                "path": str(npy_path)
            })
        except Exception as e:
            logger.warning(f"Error loading {npy_path}: {e}")
            continue

    df = pd.DataFrame(records, index=[r['id'] for r in records])
    df.sort_index(inplace=True)
    df.index.name = 'id'
    logger.success(f"Loaded {len(df)} sequences from {base_dir}")
    return df