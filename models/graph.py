from typing import List, Union, Optional
import numpy as np
from numpy.typing import NDArray
from utils.frobenius import frobenius_norm
from utils.load import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class MusicGraph:
    
    def __init__(self, file_names: Union[List[str], str], adjacency_matrices: NDArray):
        self.file_names = file_names
        self.adjacency_matrices = adjacency_matrices
        
    def apply_method(self, method_name: str, **kwargs) -> None:

        methods = {
            "clustering": self.cluster_graph
        }
        
        methods[method_name](**kwargs)
    
    def create_frobenius_graph(self) -> np.ndarray:
        n_pieces = len(self.file_names)
        frobenius_matrix = np.zeros((n_pieces, n_pieces))
        
        for i in tqdm(range(n_pieces), desc="Calculating Frobenius graph"):
            for j in range(n_pieces):
                if i != j:
                    diff_matrix = self.adjacency_matrices[i] - self.adjacency_matrices[j]
                    frobenius_matrix[i, j] = frobenius_norm(diff_matrix)
                    
        return frobenius_matrix
    
    def plot_graph(self, frobenius_matrix: np.ndarray, threshold: Optional[float] = None, 
                  node_size: int = 100, figsize: tuple = (12, 8)) -> None:
        G = nx.Graph()
        for i, name in enumerate(self.file_names):
            simple_name = name.split('/')[-1].split('.')[0]
            G.add_node(i, name=simple_name)
        
        n_pieces = len(self.file_names)
        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                weight = frobenius_matrix[i, j]
                if threshold is None or weight < threshold:
                    G.add_edge(i, j, weight=weight)
        
        plt.figure(figsize=figsize)
        
        #pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        #pos = nx.kamada_kawai_layout(G)
        pos = nx.forceatlas2_layout(G)

        nx.draw(G, pos, 
                node_size=node_size,
                node_color='lightblue',
                with_labels=True,
                labels={node: G.nodes[node]['name'] for node in G.nodes()},
                edge_color='red',
                width=[80/G[u][v]['weight'] for u, v in G.edges()],
                font_size=8)
        
        plt.title(f"Music Similarity Graph (Edge width indicates similarity), with threshold = {threshold}")
        plt.savefig("figures/graph/music_similarity_graph_threshold_0_7_times_mean_norm.png")
        plt.show()

if __name__ == "__main__":
    matrices, meta = load_dataset()

    number_of_files_to_use = len(matrices) 
    #number_of_files_to_use = 50
    matrices = matrices.iloc[:number_of_files_to_use]
    meta = meta.iloc[:number_of_files_to_use]

    file_names = matrices['piece_name'].tolist()
    adjacency_matrices = np.array(matrices['matrix'].tolist())
    
    graph = MusicGraph(file_names, adjacency_matrices)
    frobenius_matrix = graph.create_frobenius_graph()

    mean_norm = np.mean(frobenius_matrix[frobenius_matrix > 0])
    threshold = mean_norm * 0.7
    graph.plot_graph(frobenius_matrix, threshold=threshold, node_size=300, figsize=(15, 10))
