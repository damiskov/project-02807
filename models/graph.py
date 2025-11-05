from dataclasses import dataclass
from numpy.typing import NDArray
from typing import List, Optional

import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---- Node -----
@dataclass 
class MusicNode:
    """Class representing a music piece node in the graph."""

    composer: str
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


    def plot_graph_by_composer(self, adjacency_matrix, percentile=20, 
                          figsize=(20, 20), node_size=300):
        G = nx.Graph()
        
        composers = list(set([node.composer for node in self.nodes]))
        composer_to_color = {composer: plt.cm.tab20(i / len(composers)) 
                            for i, composer in enumerate(composers)}
        
        for i, node in enumerate(self.nodes):
            G.add_node(i, name=f"{node.composer}_{node.composition}", composer=node.composer)
        
        threshold = np.percentile(adjacency_matrix[np.triu_indices_from(adjacency_matrix, k=1)], percentile)
        n_pieces = len(self.nodes)
        
        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                weight = adjacency_matrix[i, j]
                if threshold is None or weight <= threshold:
                    G.add_edge(i, j, weight=weight)
        
        _, ax = plt.subplots(figsize=figsize)
        
        pos = nx.spring_layout(G, k=15, iterations=100, seed=42)
    
        nx.draw_networkx_edges(G, pos, alpha=0.7, width=0.9, edge_color='gray', ax=ax)
        
        for composer in composers:
            nodelist = [i for i, node in enumerate(self.nodes) if node.composer == composer]
            nx.draw_networkx_nodes(G, pos, 
                                nodelist=nodelist,
                                node_size=node_size, 
                                node_color=[composer_to_color[composer]],
                                alpha=0.8,
                                edgecolors='white',
                                linewidths=2,
                                label=composer,
                                ax=ax)
        
        nx.draw_networkx_labels(G, pos, 
                            labels={i: str(i) for i in G.nodes()},
                            font_size=8,
                            font_weight='bold',
                            ax=ax)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, 
                title='Composers', frameon=True, fancybox=True, shadow=True)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        plt.savefig('figures/graph/music_graph.png')

    
# helper: create list of nodes from metadata
def create_music_nodes(
    metadata: pd.DataFrame
) -> List[MusicNode]:
    nodes = []

    for idx, row in metadata.iterrows():

        node = MusicNode(
            composer=row['composer'],
            composition=row['composition'],
            ensemble=row['ensemble'],
            id=idx
        )
        nodes.append(node)
    return nodes

if __name__ == "__main__":

    from utils.load import load_dataset
    from utils.frobenius import create_frobenius_adjacency_matrix

    matrices, meta = load_dataset()

    n = 125 # Sample n random pieces
    ctms = matrices["matrix"].sample(frac=1, random_state=42).head(n) # shuffle to get random samples
    meta = meta.iloc[:n]
    
    nodes = create_music_nodes(meta)
    frobenius_adjacency_matrix = create_frobenius_adjacency_matrix(ctms.tolist())

    graph = MusicGraph(nodes, frobenius_adjacency_matrix)
    graph.plot_graph_by_composer(frobenius_adjacency_matrix, percentile=20, node_size=300, figsize=(15, 10))
