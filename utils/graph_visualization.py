import numpy as np
from graphviz import Graph
import plotly.graph_objects as go
import numpy as np

def graph_viz (adj_matrix, labels, filename, file_type="svg", threshold=0):

    cluster_labels = labels.tolist()
    # add more colors if more clusters are expected
    cluster_colors = [
        '#ff1493', '#39FF14', '#0088ff', '#FFA500', 
        '#000000', '#BC13FE', '#00CED1', '#FFD700', '#FF4500', '#32CD32'
    ]

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


def plot_spectral_clusters(sim_matrix, labels_arr, edge_percentile=90, show_edges=False, save_path="plot.png"):
    """
    Plot a 2D spectral layout of clusters from a similarity matrix.

    Args:
        sim_matrix (np.ndarray): Symmetric similarity/adjacency matrix.
        labels_arr (np.ndarray): Cluster labels for each node.
        edge_percentile (float): Percentile threshold for showing edges.
        show_edges (bool): Whether to plot edges between nodes.
        save_path (str): Path to save the plot image (PNG, requires kaleido).
    """
    if sim_matrix is None or labels_arr is None:
        raise ValueError("sim_matrix and labels_arr must be provided.")

    # Spectral layout using last two eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.array(sim_matrix))
    X_pos = eigvecs[:, -2:]
    X_pos = (X_pos - X_pos.min(axis=0)) / (X_pos.max(axis=0) - X_pos.min(axis=0))

    node_x = X_pos[:, 0]
    node_y = X_pos[:, 1]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=12,
            color=labels_arr,
            colorscale='Viridis',
            line_width=1
        ),
        hoverinfo='text',
        text=[f"Cluster {label}" for label in labels_arr]
    )

    data_traces = [node_trace]

    if show_edges:
        edge_x, edge_y = [], []
        threshold = np.percentile(sim_matrix, edge_percentile)
        for i in range(sim_matrix.shape[0]):
            for j in range(i + 1, sim_matrix.shape[1]):
                if sim_matrix[i, j] >= threshold:
                    edge_x += [X_pos[i, 0], X_pos[j, 0], None]
                    edge_y += [X_pos[i, 1], X_pos[j, 1], None]
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        data_traces.insert(0, edge_trace)  # draw edges behind nodes

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title="2D Spectral Layout of Clusters",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=600
    )

    fig.write_image(save_path, engine="kaleido")
    return
