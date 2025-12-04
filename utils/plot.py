import matplotlib.pyplot as plt
from matplotlib import cm

def plot_metadata_distributions(data, top_n=10, save_dir=None, model_name=None):
    """
    Plot distributions for metadata categories from a JSON-like dict.
    For each category, creates a single figure with subplots showing each cluster 
    compared to the global distribution, using tab10 color scheme.

    Expected structure:
    {
      "0": {
          "name": {"counts": {...}},
          "themes": {"counts": {...}},
          "keywords": {"counts": {...}},
          "involved_companies": {"counts": {...}},
          "first_release_year": {"counts": {...}}
      },
      "1": {...},  # optionally more clusters
    }

    Parameters
    ----------
    data : dict
        Parsed JSON dictionary.
    top_n : int
        Number of top values to plot for categorical variables.
    """
    
    if save_dir and not model_name:
        raise ValueError("If save_dir is provided, model_name must also be provided for file naming.")
    
    # Get tab10 colormap
    tab10 = cm.get_cmap('tab10')
    
    cluster_sizes = {id: data[id].get('cluster_size', 0) for id in data}
    cluster_ids = sorted(data.keys(), key=lambda x: int(x))
    n_clusters = len(cluster_ids)
    
    # Collect all categories (excluding cluster_size)
    categories = set()
    for cluster_dict in data.values():
        for key in cluster_dict.keys():
            if key != "cluster_size" and "counts" in cluster_dict.get(key, {}):
                categories.add(key)
    
    # Compute global distributions for each category
    global_distributions = {}
    for category in categories:
        global_dist = {}
        for cluster_id in cluster_ids:
            cluster_dict = data[cluster_id]
            if category in cluster_dict and "counts" in cluster_dict[category]:
                counts = cluster_dict[category]["counts"]
                for item, count in counts.items():
                    global_dist[item] = global_dist.get(item, 0) + count
        global_distributions[category] = global_dist
    
    # Process each category
    for category in sorted(categories):
        print(f"\n=== {category} ===")
        
        # Determine if this is year data
        is_year = category == "first_release_year"
        
        # Create subplot grid (clusters + 1 for global)
        n_plots = n_clusters + 1
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else list(axes)
        
        # Plot global distribution first
        ax = axes[0]
        global_counts = global_distributions[category]
        
        if is_year:
            # Year histogram
            years = []
            freqs = []
            for k, v in global_counts.items():
                try:
                    years.append(int(k))
                    freqs.append(v)
                except ValueError:
                    continue
            
            if years:
                sorted_data = sorted(zip(years, freqs))
                years, freqs = zip(*sorted_data)
                ax.bar(years, freqs, color='gray', alpha=0.7)
                ax.set_title(f"Global: {category}", fontweight='bold')
                ax.set_xlabel("Year")
                ax.set_ylabel("Count")
        else:
            # Categorical bar chart
            sorted_items = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            labels = [x[0] for x in sorted_items]
            values = [x[1] for x in sorted_items]
            
            if labels:
                ax.barh(labels, values, color='gray', alpha=0.7)
                ax.invert_yaxis()
                ax.set_title(f"Global: {category}", fontweight='bold')
                ax.set_xlabel("Count")
        
        # Plot each cluster
        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes[idx + 1]
            cluster_dict = data[cluster_id]
            
            if category not in cluster_dict or "counts" not in cluster_dict[category]:
                ax.axis('off')
                continue
            
            counts = cluster_dict[category]["counts"]
            
            if not counts:
                ax.axis('off')
                continue
            
            # Get cluster color from tab10
            color = tab10(int(cluster_id) % 10)
            
            if is_year:
                # Year histogram
                years = []
                freqs = []
                for k, v in counts.items():
                    try:
                        years.append(int(k))
                        freqs.append(v)
                    except ValueError:
                        continue
                
                if years:
                    sorted_data = sorted(zip(years, freqs))
                    years, freqs = zip(*sorted_data)
                    ax.bar(years, freqs, color=color, alpha=0.7)
                    ax.set_title(f"Cluster {int(cluster_id)+1} (n={cluster_sizes[cluster_id]})", fontweight='bold')
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Count")
            else:
                # Categorical bar chart
                sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                labels = [x[0] for x in sorted_items]
                values = [x[1] for x in sorted_items]
                
                if labels:
                    ax.barh(labels, values, color=color, alpha=0.7)
                    ax.invert_yaxis()
                    ax.set_title(f"Cluster {int(cluster_id)+1} (n={cluster_sizes[cluster_id]})", fontweight='bold')
                    ax.set_xlabel("Count")
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_dir:
            save_path = f"{save_dir}/{model_name}_{category}_distributions.png"
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")
        plt.show()
    
    # Plot cluster sizes as separate figure
    print(f"\n=== Cluster Sizes ===")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [tab10(int(cid) % 10) for cid in cluster_ids]
    ax.bar(cluster_ids, [cluster_sizes[cid] for cid in cluster_ids], color=colors, alpha=0.7)
    ax.set_title("Cluster Sizes", fontweight='bold')
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Size")
    plt.tight_layout()
    if save_dir:
        save_path = f"{save_dir}/{model_name}_cluster_sizes.png"
        plt.savefig(save_path, dpi=300)
    plt.show()