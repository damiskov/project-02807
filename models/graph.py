from dataclasses import dataclass
from numpy.typing import NDArray
from typing import List, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ---- Node -----
@dataclass 
class MusicNode:
    """Class representing a music piece node in the graph."""

    composer: str
    piece_name: str
    composition: str
    ensemble: str
    id: int



# ---- Graph -----
@dataclass
class MusicGraph:
    """Class representing a music similarity graph."""

    nodes: List[MusicNode]
    adjacency_matrix: NDArray

    def display(self):
        """TODO: Implement a pretty way to display the graph."""
        pass


    def plot_graph_networkx(
        self,
        adjacency_matrix: np.ndarray,
        threshold: Optional[float] = None, 
        node_size: int = 100, figsize: tuple = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        
        """
        Plot the music similarity graph based on the adjacency matrix.
        
        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix.
            threshold (Optional[float]): If provided, edges with weights above this threshold will be excluded.
            node_size (int): Size of the nodes in the graph.
            figsize (tuple): Size of the figure.
            save_path (Optional[str]): If provided, the path to save the plotted graph image.
        """
        
        G = nx.Graph()
        
        for i, node in enumerate(self.nodes):
            G.add_node(i, name=node.piece_name)

        n_pieces = len(self.nodes)
        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                weight = adjacency_matrix[i, j]
                if threshold is None or weight <= threshold:
                    G.add_edge(i, j, weight=weight)
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=node_size)
        
        if save_path:
            plt.savefig(save_path)
        else:   
            plt.show()

    



if __name__ == "__main__":

    from utils.load import load_dataset
    from utils.frobenius import create_frobenius_adjacency_matrix
    from utils.frobenius import frobenius_norm

    matrices, meta = load_dataset()

    number_of_files_to_use = len(matrices) 
    #number_of_files_to_use = 50
    ctms = matrices.iloc[:number_of_files_to_use]
    meta = meta.iloc[:number_of_files_to_use]

    file_names = ctms['piece_name'].tolist()

    frobenius_adjacency_matrix = create_frobenius_adjacency_matrix(file_names, ctms['matrix'].tolist())

    graph = MusicGraph(file_names, frobenius_adjacency_matrix
                       )
    mean_norm = np.mean(frobenius_adjacency_matrix[frobenius_adjacency_matrix > 0])
    threshold = mean_norm * 0.7
    graph.plot_graph_networkx(frobenius_adjacency_matrix, threshold=threshold, node_size=300, figsize=(15, 10))
