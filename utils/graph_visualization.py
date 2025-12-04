import numpy as np
from graphviz import Graph

def graph_viz (adj_matrix, labels, filename, file_type="svg", threshold=0):

    cluster_labels = labels.tolist()
    # add more colors if more clusters are expected
    cluster_colors = ['#ff1493', '#39FF14', '#0088ff', '#FFA500', '#000000', '#BC13FE']

    n = adj_matrix.shape[0]


    upper_tri_indices = np.triu_indices(n, k=1)        # exclude diagonal
    edge_distances = adj_matrix[upper_tri_indices]

    # show the edges that have a distance value inferior to the threshold
    edges_vis_threshold = np.percentile(edge_distances, 20)


    weights = adj_matrix[adj_matrix > 0]
    if len(weights) == 0:
        weights = np.array([1])
    min_w, max_w = weights.min(), weights.max()


    dot = Graph('100_nodes', engine='sfdp')       
    dot.attr(outputorder='edgesfirst')
    dot.attr(overlap='voronoi')

    dot.attr('node',
            shape='circle',
            style='filled',
            fillcolor='#ff1493',     
            color='#ff69b4',         
            penwidth='0',             
            width='0.1',
            height='0.1',
            fixedsize='true',
            margin='0',              
            fontsize='0',            
            label='',                
            fontcolor='none',        
            z='100')


    # dot.attr(splines='curved')
    dot.attr(splines='polyline')
    dot.attr(overlap='prism1000')          
    # dot.attr(esep='1.0')


    dot.attr('graph', sep='10')
    dot.attr(nodesep='2.0')     # horizontal distance between nodes 
    dot.attr(ranksep='3.0')    
    dot.attr(sep='+10')         # global “push-out” force 
    # dot.attr(esep='+20')



    for i in range(n):
        dot.node(str(i))

    for i in range(n):
        for j in range(i+1, n):      

            # edges 

            w = adj_matrix[i, j]
            if w > 0:
                # Normalize weights
                normalized = (w - min_w) / (max_w - min_w) if max_w > min_w else 1.0

                # Opacity: 30% -> 100%, if we want to change the opacity depending on distance
                alpha = int(0.3 + 0.7 * normalized * 255)     # 77 → 255 in hex
                opacity_hex = f"{alpha:02X}"

                # Thickness of the edge, between 0.8 - 4.0, in this case
                penwidth = 0.8 + 3.2 * normalized

                distance = 0.1 + 1.0 * (1 - normalized)**2

                # Color with opacity
                color = f"#444444{opacity_hex}"

                # threshold 0 is the default
                if w<edges_vis_threshold and threshold==0:

                    cluster_clr = cluster_colors[cluster_labels[i]]

                    dot.edge(str(i), str(j),
                            #  penwidth=f"{penwidth:.2f}",
                            penwidth="0.3",
                            color=cluster_clr)

            # nodes

            cluster_clr = cluster_colors[cluster_labels[i]]

            if cluster_labels[i]==cluster_labels[j]:
                dot.node(str(i),
                    fillcolor=cluster_clr,
                    color=cluster_clr,
                    penwidth='0')
                
                dot.node(str(j),
                    fillcolor=cluster_clr,
                    color=cluster_clr,
                    penwidth='0')
                
                # dot.edge(str(i), str(j),
                #             #  penwidth=f"{penwidth:.2f}",
                #             penwidth="0.3",
                #             color=cluster_clr)


    # save svg
    dot.render(filename, format=file_type, cleanup=True)
    
    
import numpy as np
from graphviz import Graph
import networkx as nx
import matplotlib.pyplot as plt
from fa2_modified import ForceAtlas2
from matplotlib.patches import Patch
from matplotlib import cm
from collections import Counter

from typing import Optional


def visualize_graph(sim_matrix, labels, filename: Optional[str]=None):

    cluster_labels = labels.tolist()
    cluster_sizes = dict(Counter(cluster_labels))

    tab10 = [
        '#1f77b4',  
        '#ff7f0e',  
        '#2ca02c',  
        '#d62728',  
        '#9467bd',  
        '#8c564b',  
        '#e377c2',  
        '#7f7f7f',  
        '#bcbd22',  
        '#17becf'   
    ]

    # add more colors if more clusters are expected
    # cluster_colors = ['#ff1493', '#39FF14', '#0088ff', '#FFA500', '#000000', '#BC13FE']
    cluster_colors = tab10

    node_colors = [cluster_colors[cluster_labels[i]] for i in range(sim_matrix.shape[0])]

    # Adjust node size based on community size (isolated nodes are larger)
    # node_sizes = [
    #     50 if louvain_partition[node] == -1 else 10 + community_sizes[louvain_partition[node]] ** 0.5
    #     for node in G_undirected.nodes()
    # ]

    G = nx.Graph()

    for i in range(len(sim_matrix)):  #add nodes
        G.add_node(i) 

    node_sizes = [
        10 for _ in G.nodes()
    ]

    edges_toshow = {}

    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):

            clr = cluster_colors[cluster_labels[i]]

            if sim_matrix[i, j] > 0:
                if clr not in edges_toshow:
                    edges_toshow[clr] = []
                
                edges_toshow[clr].append((i,j))
                
                edges_toshow[clr].append((i,j))
                G.add_edge(i, j, weight=sim_matrix[i, j])


    # Step 3: Use the ForceAtlas2 algorithm to compute node layout
    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # Use linear distances
        adjustSizes=False,  # Prevent overlap
        edgeWeightInfluence=1.0,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1,
        scalingRatio=75,
        strongGravityMode=False,
        gravity=3,
        verbose=True
    )

    G_undirected = G
    positions = forceatlas2.forceatlas2_networkx_layout(G_undirected, pos=None, iterations=2000)

    # Draw the network graph
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(
        G_undirected, 
        pos=positions, 
        node_color=node_colors, 
        node_size=[
            node_sizes[i]
            for i, node in enumerate(G_undirected.nodes())
        ], 
        alpha=0.8,
    )

    for c in list(edges_toshow.keys()):
        nx.draw_networkx_edges(
            G_undirected, 
            edgelist=edges_toshow[c],
            pos=positions, 
            edge_color=c, 
            width=0.3,
            alpha=[
                0.05 
                for _ in edges_toshow[c]
            ]
        )

    communities = [i for i in range(6)]

    # plt.title("Network Visualization using ForceAtlas2 (Colored by Cluster)")
    legend_elements = [
        # Replace 16 with community sizes
        Patch(facecolor=cluster_colors[community], label=f"Cluster {community}: ({cluster_sizes[community]} nodes)")
        for community in list(cluster_sizes.keys())
    ]
    # legend_elements.append(Patch(facecolor=(0.7, 0.7, 0.7), label="Other Communities"))
    # plt.legend(handles=legend_elements, loc="lower left", fontsize=8)

    plt.axis('off')  # Hide axes
    plt.show()
    if filename is not None:
        plt.savefig(f"../results/embedding_general/graph_clustering/{filename}", dpi=300, bbox_inches="tight")