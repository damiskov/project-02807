from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
import re

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load movie metadata CSV and keep IMDb rating only."""
    import ast

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.index = df['id']
    df.sort_index(inplace=True)
    df.drop('id', axis=1, inplace=True)

    if 'ratings' in df.columns:
        def extract_imdb(ratings_str):
            try:
                ratings_list = ast.literal_eval(ratings_str)
                for r in ratings_list:
                    if r['Source'] == 'Internet Movie Database':
                        # convert "8.6/10" → float 8.6
                        return float(r['Value'].split('/')[0])
            except Exception:
                return np.nan
            return np.nan

        df['imdbrating'] = df['ratings'].apply(extract_imdb)
        df.drop('ratings', axis=1, inplace=True)

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
    # align based on 'id' if presen
    return matrices_df, metadata_df


def load_sequences(base_dir: str | Path = "data/sequences", file_extension: str = ".npy") -> pd.DataFrame:
    """
    Recursively load all musical sequences for movie themes from .npy files into a DataFrame.
    Each row: piece_id (IMDb), sequence (np.ndarray), path.
    """
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



if __name__ == "__main__":
    matrices, meta = load_dataset()
    logger.info(f"Matrix sample:\n{matrices.head()}")
    logger.info(f"Metadata sample:\n{meta.head()}")

    # save the first matrix to verify
    # random int
    # random_int = np.random.randint(0, len(matrices))
    # sample_matrix = matrices.iloc[random_int]['matrix']
    # name = matrices.iloc[random_int]['piece_name']
    
    # import matplotlib.pyplot as plt
    # plt.imshow(sample_matrix, cmap='hot', interpolation='nearest')
    # plt.title(f"Chord Trajectory Matrix Sample: {name}")
    # plt.colorbar()
    # plt.show()
