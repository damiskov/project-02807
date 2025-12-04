import os
import json
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # --- Figure ---
    "figure.figsize": (6, 4),
    "figure.dpi": 300,

    # --- Fonts ---
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "font.family": "serif",       # or "sans-serif"
    "mathtext.fontset": "stix",   # clean math font

    # --- Axes ---
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": False,

    # --- Ticks ---
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    # --- Legend ---
    "legend.fontsize": 10,
    "legend.frameon": False,

    # --- Colors / Colormap ---
    "image.cmap": "viridis",
})



def cluster_jaccard_similarity_matrix(topic_words_dict, print_top=10):
    """
    Compute a symmetric Jaccard similarity matrix between clusters.

    Parameters
    ----------
    topic_words_dict : dict
        {cluster_id: [list of words]}
    print_top : int
        How many top similarities to print.

    Returns
    -------
    sim_matrix : np.ndarray
        Symmetric matrix of shape (n_clusters, n_clusters)
    cluster_ids : list
        Order of cluster IDs corresponding to matrix rows/cols
    """

    cluster_ids = list(topic_words_dict.keys())
    n = len(cluster_ids)

    # Initialize matrix with zeros
    sim_matrix = np.zeros((n, n), dtype=float)

    # Jaccard for each pair
    for i, c1 in enumerate(cluster_ids):
        set1 = set(topic_words_dict[c1])

        for j, c2 in enumerate(cluster_ids):
            if i > j:
                continue  # fill only upper triangle, mirror later

            if i == j:
                sim_matrix[i, j] = 1.0
                continue

            set2 = set(topic_words_dict[c2])
            jaccard = len(set1 & set2) / len(set1 | set2)

            sim_matrix[i, j] = jaccard
            sim_matrix[j, i] = jaccard  # ensure symmetry

    # Print top similarities
    if print_top > 0:
        flat_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                flat_pairs.append(((cluster_ids[i], cluster_ids[j]), sim_matrix[i, j]))

        flat_pairs.sort(key=lambda x: x[1], reverse=True)

        print("\nTop Jaccard similarities:")
        for (c1, c2), score in flat_pairs[:print_top]:
            print(f"{c1} ↔ {c2}: {score:.3f}")

    return sim_matrix, cluster_ids

def plot_similarity_matrix(sim_matrix, cluster_ids, title="Cluster Jaccard Similarity Matrix", save_path=None):
    """
    Plot a heatmap of the similarity matrix.

    Parameters
    ----------
    sim_matrix : np.ndarray
        Symmetric similarity matrix.
    cluster_ids : list
        List of cluster IDs corresponding to matrix rows/cols.
    title : str
        Title of the plot.
    """ 

    plt.figure(figsize=(10, 8))
    # sns.heatmap(sim_matrix, xticklabels=cluster_ids, yticklabels=cluster_ids,
    #             cmap="viridis", annot=True, fmt=".2f", square=True)
    plt.imshow(sim_matrix, cmap="viridis", interpolation="nearest")
    # plt.colorbar(label="Jaccard Similarity")
    plt.title(title)
    plt.xlabel("Clusters")
    plt.ylabel("Clusters")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_similarity_sum(sim_matrix):
    """
    Compute the sum of the off-diagonal similarity values in the Jaccard matrix.
    
    Higher value = more overlap between clusters.
    
    Final external validation metric.
    
    Parameters
    ----------
    sim_matrix : np.ndarray
        Symmetric similarity matrix of shape (n_clusters, n_clusters).
    
    Returns
    -------
    float
        Sum of all off-diagonal elements (upper triangle only to avoid double counting).
    """
    n = sim_matrix.shape[0]
    off_diagonal_sum = 0.0
    
    # Sum upper triangle only (excluding diagonal)
    for i in range(n):
        for j in range(i + 1, n):
            off_diagonal_sum += sim_matrix[i, j]
    
    return off_diagonal_sum

if __name__ == "__main__":

    
    root_dir = r"results\temp"
    output_dir = r"results\temp"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    similarity_sums = {}
    # Recursively walk through the directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                print(f"\nProcessing: {filepath}")
                
                try:
                    # Load JSON file
                    with open(filepath, 'r', encoding='utf-8') as f:
                        topic_words_dict = json.load(f)
                    
                    # Compute similarity matrix
                    sim_matrix, cluster_ids = cluster_jaccard_similarity_matrix(
                        topic_words_dict, 
                        print_top=10
                    )
                    
                    # Create plot title from filename
                    base_name = os.path.splitext(file)[0]
                    title = f"Cluster Jaccard Similarity: {base_name}"
    
                    # Save the figure
                    output_filename = f"{base_name}_similarity_heatmap.png"
                    output_path = os.path.join(output_dir, output_filename)
                    plot_similarity_matrix(sim_matrix, cluster_ids, title=title, save_path=output_path)
                    print(f"Saved: {output_path}")
                    similarity_sum = compute_similarity_sum(sim_matrix)
                    print(f"Similarity sum for filename {file}: {similarity_sum:.3f}")
                    similarity_sums[file] = similarity_sum
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    
    # Save similarity sums to a JSON file
    sums_output_path = os.path.join(output_dir, "similarity_sums.json")
    with open(sums_output_path, 'w', encoding='utf-8') as f:
        json.dump(similarity_sums, f, indent=4)
    