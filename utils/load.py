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

    return df

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

def load_feature_matrices_aligned(base_dir="data/ctms", metadata_csv="data/metadata/videogames_metadata.csv"):
    """
    Load all CTM matrices and align them with videogame metadata.
    Assumes structure: base_dir/v<game_id>/song_name.npy
    Returns matrices_df with game metadata merged.
    """
    base_dir = Path(base_dir)
    records = []

    for npy_path in tqdm(base_dir.rglob("*.npy"), desc="Loading CTMs"):
        piece_id = npy_path.stem  # e.g., v281495_song_A
        try:
            matrix = np.load(npy_path)
            # Extract videogame ID from directory name (vXXXX)
            match = npy_path.parts[-2]  # parent directory, e.g., v281495
            game_id = int(match.lstrip("v"))

            records.append({
                "id": piece_id,
                "matrix": matrix,
                "path": str(npy_path),
                "game_id": game_id
            })
        except Exception as e:
            logger.warning(f"Error loading {npy_path}: {e}")
            continue

    matrices_df = pd.DataFrame(records)
    matrices_df.sort_values("id", inplace=True)
    matrices_df.set_index("id", inplace=True)

    # Load metadata and align by game_id
    metadata_df = pd.read_csv(metadata_csv)
    metadata_df.set_index("id", inplace=True)
    # Keep only rows corresponding to available game_ids
    metadata_df = metadata_df.loc[metadata_df.index.intersection(matrices_df["game_id"].unique())]

    # Merge metadata into matrices_df
    matrices_df = matrices_df.merge(metadata_df, left_on="game_id", right_index=True, how="left")

    logger.success(f"Loaded {len(matrices_df)} matrices aligned with {len(metadata_df)} games")
    return matrices_df, metadata_df

def load_feature_matrices(base_dir: str | Path) -> pd.DataFrame:
    """
    Recursively load all .npy chord trajectory matrices for videogame soundtracks into a DataFrame.
    Each row: piece_id (console_vID_midiname), matrix (np.ndarray), path.
    
    Assumes folder structure:
        base_dir/
            console_name/
                v<ID>/
                    *.npy
    """
    base_dir = Path(base_dir)
    records = []

    for npy_path in tqdm(base_dir.rglob("*.npy"), desc="Loading matrices"):
        try:
            # Extract console and game ID
            game_dir = npy_path.parents[0].name       # v<ID>
            console_name = npy_path.parents[1].name  # console_name

            # Only keep numeric ID after 'v'
            match = game_dir.lower().startswith("v") and game_dir[1:].isdigit()
            if match:
                game_id = game_dir
            else:
                game_id = game_dir

            midiname = npy_path.stem.replace("_traj", "")
            piece_id = f"{console_name}_{game_id}_{midiname}"

            # Load matrix
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
