from dataclasses import dataclass
from numpy.typing import NDArray
from typing import List, Optional

import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
import matplotlib.pyplot as plt


# ---- Node -----
@dataclass 
class MovieNode:
    """Class representing a music piece node in the graph."""
    id: str
    title: str
    released: str


# ---- Graph -----
@dataclass
class MovieGraph:
    nodes: List[MovieNode]
    adjacency_matrix: np.ndarray

    def _distance_to_similarity(self, sigma: Optional[float] = None) -> np.ndarray:
        """Convert Frobenius distances to similarities using a Gaussian kernel."""
        D = self.adjacency_matrix
        if sigma is None:
            # median heuristic
            sigma = np.median(D[D > 0])
        S = np.exp(-D**2 / (2 * sigma**2))
        np.fill_diagonal(S, 0)  # no self-links
        return S

    def plot_graph_networkx(
        self,
        adjacency_matrix: np.ndarray,
        threshold: Optional[float] = None,
        k: Optional[int] = None,
        node_size: int = 300,
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None,
        layout: str = "mds",  # "spring" or "mds"
        random_state: int = 0
    ):
        """
        Plot a movie similarity graph.

        Parameters
        ----------
        threshold : float, optional
            Keep only edges with similarity > threshold.
        k : int, optional
            Use k-nearest-neighbor sparsification instead of threshold.
        layout : {"spring", "mds"}
            Layout algorithm for node positions.
        """
        from sklearn.manifold import MDS
        from sklearn.neighbors import kneighbors_graph
        # --- Convert distances → similarities ---
        S = self._distance_to_similarity()

        # --- Sparsify ---
        if k is not None:
            A = kneighbors_graph(S, n_neighbors=k, mode="connectivity", include_self=False)
            A = 0.5 * (A + A.T)  # symmetrize
            G = nx.from_scipy_sparse_array(A)
        else:
            if threshold is not None:
                S = np.where(S >= threshold, S, 0)
            G = nx.from_numpy_array(S)

        # --- Assign node metadata ---
        mapping = {i: node.title for i, node in enumerate(self.nodes)}
        G = nx.relabel_nodes(G, mapping)

        # --- Choose layout ---
        if layout == "spring":
            pos = nx.spring_layout(G, seed=random_state)
        elif layout == "mds":
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)
            D = adjacency_matrix
            coords = mds.fit_transform(D)
            pos = {mapping[i]: coords[i] for i in range(len(self.nodes))}
        else:
            raise ValueError("layout must be 'spring' or 'mds'")

        # --- Draw ---
        plt.figure(figsize=figsize)
        nx.draw_networkx(
            G,
            pos,
            node_size=node_size,
            with_labels=True,
            font_size=9,
            edge_color="gray",
            alpha=0.7
        )
        plt.title("Movie Similarity Graph")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
    
# helper: create list of nodes from metadata
def create_movie_nodes(metadata: pd.DataFrame) -> List[MovieNode]:
    nodes = []
    for movie_id, row in metadata.iterrows():
        node = MovieNode(
            id=movie_id,
            title=row['title'],
            released=row['released']
        )
        nodes.append(node)
    return nodes

if __name__ == "__main__":

    from utils.load import load_feature_matrices, load_metadata
    from utils.frobenius import create_frobenius_adjacency_matrix

    # Load your movie data
    ctms_df = load_feature_matrices("../data/ctms")   # each row: matrix for a movie
    meta_df = load_metadata("../data/metadata/movies_metadata.csv")        # metadata keyed by IMDb ID

    # Use first n movies
    n = 50
    ctms = ctms_df['matrix'].iloc[:n].tolist()
    meta = meta_df.iloc[:n]

    # Create nodes and adjacency matrix
    nodes = create_movie_nodes(meta)
    frobenius_adjacency_matrix = create_frobenius_adjacency_matrix(ctms)

    # Plot the graph
    graph = MovieGraph(nodes, frobenius_adjacency_matrix)
    mean_norm = np.mean(frobenius_adjacency_matrix[frobenius_adjacency_matrix > 0])
    threshold = mean_norm * 0.7
    graph.plot_graph_networkx(frobenius_adjacency_matrix, threshold=threshold, node_size=300, figsize=(15,10))
