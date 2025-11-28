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