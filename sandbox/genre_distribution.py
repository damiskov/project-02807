import pandas as pd 
from loguru import logger
import matplotlib.pyplot as plt

from utils.load import load_metadata

def get_metadata() -> pd.DataFrame:
    """Load metadata for use in genre distribution plotting."""
    metadata_df = load_metadata("data/metadata/movies_metadata.csv")
    return metadata_df

def plot_genre_distribution(metadata_df: pd.DataFrame):


    # Extract genres
    all_genres = []
    for genres in metadata_df['genre']:
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(',')]
            all_genres.extend(genres)
        else:
            logger.warning(f"Non-string genre entry: {genres}")

    # Count genre occurrences
    genre_counts = pd.Series(all_genres).value_counts()
    # Plot genre distribution
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar')
    plt.title('Genre Distribution')
    plt.xlabel('Genres')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    metadata_df = get_metadata()
    plot_genre_distribution(metadata_df)