"""
Produces all baseline plots.


"""
import numpy as np
import pandas as pd
from loguru import logger
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from utils.load import load_sequences, load_metadata
from utils.pitch import extract_pitch_based_features
from utils.pitch import plot_pca_variance
from utils.pitch import plot_clusters, plot_dendrogram

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_pitch_features():
    # extract sequences
    sequences_df = load_sequences("data/sequences", metadata="data/metadata/movies_metadata.csv")
    records = []

    # generate feature

    for idx, row in sequences_df.iterrows():
        seq = row['sequence']
        features = extract_pitch_based_features(seq, max_interval=12)

        records.append({
            "id": idx,
            "pitch_features": features,
            **{i: row[i] for i in sequences_df.columns} # we want to keep all metadata columns
        })


    features_df = pd.DataFrame(records, index=[r['id'] for r in records])
    features_df.sort_index(inplace=True)

    # apply pca to pitch features

    X = np.vstack(features_df["pitch_features"].values)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=0.95, random_state=42)   # keep 95% of variance
    X_pca = pca.fit_transform(X_std)
    logger.info(f"PCA reduced from {X.shape[1]} → {X_pca.shape[1]} dimensions "
                f"({pca.explained_variance_ratio_.sum():.2%} variance kept)")
    
    features_df["pca_features"] = list(X_pca)

    return features_df


# Get all unique genres

def plot_genre_pca_3d(df: pd.DataFrame):
    """3D scatter plots of PCA features colored by genre."""

    all_genres = set()

    for genres in df['genre'].str.split(','):
        all_genres.update([g.strip() for g in genres])

    pca_array = np.array(df['pca_features'].tolist())

    # Create subplots for top genres
    genre_counts = Counter()
    for genres in df['genre'].str.split(','):
        for g in genres:
            genre_counts[g.strip()] += 1

    top_genres = [g for g, _ in genre_counts.most_common(6)]

    fig = plt.figure(figsize=(15, 10))
    for idx, genre in enumerate(top_genres, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # Filter movies with this genre
        mask = df['genre'].str.contains(genre, na=False)
        
        ax.scatter(pca_array[mask, 0], 
                pca_array[mask, 1], 
                pca_array[mask, 2],
                alpha=0.6)
        ax.set_title(f'{genre} ({mask.sum()} movies)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    plt.tight_layout()
    plt.show()

def plot_pca_by_genre(
    df: pd.DataFrame,
    n_dims: int = 3,
    pc_indices: list = None,
    genre_column: str = 'genre'
):
    """
    Plot PCA features colored by genre with flexible dimensionality.

    Parameters:
    -----------
    df : DataFrame
        DataFrame with 'pca_features' and genre column
    n_dims : int
        Number of dimensions to plot (2 or 3)
    pc_indices : list or None
        Which PCs to plot (e.g., [0, 2, 3] for PC1, PC3, PC4).
        If None, uses first n_dims components
    genre_column : str
        Column name for genre labels

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    pca_array = np.array(df['pca_features'].tolist())

    # Select which PCs to plot
    if pc_indices is None:
        pc_indices = list(range(n_dims))
    elif len(pc_indices) != n_dims:
        raise ValueError(f"pc_indices must have length {n_dims}")

    # Get unique genres and colors
    
    unique_genres = df[genre_column].unique()

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))

    fig = plt.figure(figsize=(12, 8))

    if n_dims == 3:
        ax = fig.add_subplot(111, projection='3d')

        for genre, color in zip(unique_genres, colors):
            mask = df[genre_column] == genre
            ax.scatter(pca_array[mask, pc_indices[0]],
                       pca_array[mask, pc_indices[1]],
                       pca_array[mask, pc_indices[2]],
                       c=[color],
                       label=genre,
                       s=50,
                       alpha=0.6)

        ax.set_xlabel(f'PC{pc_indices[0]+1}')
        ax.set_ylabel(f'PC{pc_indices[1]+1}')
        ax.set_zlabel(f'PC{pc_indices[2]+1}')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    elif n_dims == 2:
        ax = fig.add_subplot(111)

        for genre, color in zip(unique_genres, colors):
            mask = df[genre_column] == genre
            ax.scatter(pca_array[mask, pc_indices[0]],
                       pca_array[mask, pc_indices[1]],
                       c=[color],
                       label=genre,
                       s=50,
                       alpha=0.6)

        ax.set_xlabel(f'PC{pc_indices[0]+1}')
        ax.set_ylabel(f'PC{pc_indices[1]+1}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    else:
        raise ValueError("n_dims must be 2 or 3")

    plt.title(f'{n_dims}D PCA Visualization by Genre')
    plt.tight_layout()
    return fig, ax


def genre_highlighted_pca_3d(df: pd.DataFrame):
    """3D scatter plot of PCA features highlighting major genres."""
    pca_array = np.array(df['pca_features'].tolist())

    # Define genres of interest
    genres_of_interest = ['Comedy', 'Action', 'Horror']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all movies in gray first
    ax.scatter(pca_array[:, 0], pca_array[:, 1], pca_array[:, 2],
            c='lightgray', alpha=0.3, s=30, label='Other')

    # Highlight each genre of interest
    markers = ['o', '^', 's', 'D']
    colors = ['red', 'blue', 'green', 'purple']

    for genre, marker, color in zip(genres_of_interest, markers, colors):
        mask = df['genre'].str.contains(genre, na=False)
        ax.scatter(pca_array[mask, 0], 
                pca_array[mask, 1], 
                pca_array[mask, 2],
                c=color,
                marker=marker,
                label=genre,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.title('3D PCA Visualization - Major Genres Highlighted')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    features_df = generate_pitch_features()
    logger.info(f"Features DataFrame sample:\n{features_df.head()}")
    plot_genre_pca_3d(features_df)
   # plot_pca_by_genre(features_df, n_dims=2)
