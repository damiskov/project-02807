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

ctms_baseline = {
      "2": 0.49901745927628105,
      "3": 0.4779488235542436,
      "4": 0.4650267192720845,
      "5": 0.3503332158249229,
      "6": 0.36203184026406576,
      "7": 0.3547597568847556,
      "8": 0.3023987915449259,
      "9": 0.310902755045113,
      "10": 0.31268342082139283,
      "12": 0.2739557094404988,
      "15": 0.24650370085405868,
      "20": 0.2502384818544209,
      "25": 0.23527212132742312
}

clap_baseline = {
    "2": 0.1201457679271698,
    "3": 0.12367318570613861,
    "4": 0.11450513452291489,
    "5": 0.1083076223731041,
    "6": 0.09659291803836823,
    "7": 0.09937311708927155,
    "8": 0.09705543518066406,
    "9": 0.10479807108640671,
    "10": 0.10226788371801376,
    "12": 0.10192717611789703,
    "15": 0.10046276450157166,
    "20": 0.09797404706478119,
    "25": 0.09526319056749344
}


ast_baseline = {
    "2": 0.12599098682403564,
    "3": 0.10102314502000809,
    "4": 0.10141900181770325,
    "5": 0.09824373573064804,
    "6": 0.11006613075733185,
    "7": 0.1059110015630722,
    "8": 0.11041785776615143,
    "9": 0.11022673547267914,
    "10": 0.10124001652002335,
    "12": 0.09984270483255386,
    "15": 0.10165844112634659,
    "20": 0.09252970665693283,
    "25": 0.09114072471857071
}

wavlm_baseline = {
    "2": 0.23162107169628143,
    "3": 0.14573049545288086,
    "4": 0.15363413095474243,
    "5": 0.1091696172952652,
    "6": 0.10414450615644455,
    "7": 0.1082657128572464,
    "8": 0.09707444161176682,
    "9": 0.09497757256031036,
    "10": 0.08627411723136902,
    "12": 0.08525720238685608,
    "15": 0.08184213191270828,
    "20": 0.07526630908250809,
    "25": 0.06724324822425842
}


ctms_graph = {
    "2": 0.639,
    "3": 0.457,
    "4": 0.359,
    "5": 0.319,
    "6": 0.325,
    "7": 0.324,
    "8": 0.371,
    "9": 0.361,
    "10": 0.355,
    "12": 0.332,
    "15": 0.334,
    "20": 0.321,
    "25": 0.304
}

ast_graph = {
  '2': 0.6808619801717065,
  '3': 0.6357408471254145,
  '4': 0.6764959349061609,
  '5': 0.6070345979600968,
  '6': 0.5971074155101531,
  '7': 0.5807216933058785,
  '8': 0.5256281645105547,
  '9': 0.4758871124828204,
  '10': 0.46396203665731295,
  '11': 0.4553476118687444,
  '12': 0.3778388286734161,
  '13': 0.3864068321614001,
  '14': 0.43156334829223636
}
wavlm_graph = {
  '2': 0.83109602819182,
  '3': 0.5354476735636673,
  '4': 0.6798843159514972,
  '5': 0.6315500563028463,
  '6': 0.5331390523497598,
  '7': 0.48368252467439804,
  '8': 0.48600982215778,
  '9': 0.4924276618184107,
  '10': 0.4378269192212701,
  '11': 0.3857216647589591,
  '12': 0.4172891600440942,
  '13': 0.45596421171996554,
  '14': 0.39956034865587253
}
clap_graph = {
  '2': 0.7127449836966612,
  '3': 0.7471303485569838,
  '4': 0.7376870080209464,
  '5': 0.7427913162219855,
  '6': 0.724043843215695,
  '7': 0.6691259730236184,
  '8': 0.5357954465742636,
  '9': 0.5883097311041875,
  '10': 0.5774117744633976,
  '11': 0.4814419263499013,
  '12': 0.4238938211265818,
  '13': 0.4384692077720241,
  '14': 0.43430249790589764
}

def plot_silhouette_comparison(save_path=None):
    """
    Plot silhouette scores vs K for all methods (CTMs and embeddings).
    
    Creates a minimalist publication-ready figure comparing baseline and 
    graph-based clustering across different representations.
    
    Parameters
    ----------
    save_path : str, optional
        Path to save the figure. If None, displays only.
    """
    methods = {
        'CTMs (Baseline)': ctms_baseline,
        'CTMs (Graph)': ctms_graph,
        'AST (Baseline)': ast_baseline,
        'AST (Graph)': ast_graph,
        'CLAP (Baseline)': clap_baseline,
        'CLAP (Graph)': clap_graph,
        'WavLM (Baseline)': wavlm_baseline,
        'WavLM (Graph)': wavlm_graph,
    }
    colors = {
        'CTMs (Baseline)': '#1f77b4',  # blue
        'CTMs (Graph)': '#ff7f0e',     # orange
        'AST (Baseline)': '#2ca02c',   # green
        'AST (Graph)': '#8c564b',      # brown
        'CLAP (Baseline)': '#d62728',  # red
        'CLAP (Graph)': '#e377c2',     # pink
        'WavLM (Baseline)': '#9467bd', # purple
        'WavLM (Graph)': '#17becf',    # cyan
    }
    
    markers = {
        'CTMs (Baseline)': 'o',
        'CTMs (Graph)': 's',
        'AST (Baseline)': '^',
        'AST (Graph)': 'v',
        'CLAP (Baseline)': 'D',
        'CLAP (Graph)': 'p',
        'WavLM (Baseline)': '*',
        'WavLM (Graph)': 'h',
    }
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot each method
    for method_name, scores in methods.items():
        if not scores:  # just in case
            print(f"No scores for {method_name}, skipping.")
            continue
        
            
        k_values = sorted([int(k) for k in scores.keys()])
        silhouette_values = [scores[str(k)] for k in k_values]
        
        ax.plot(k_values, silhouette_values, 
                marker=markers[method_name], 
                color=colors[method_name],
                label=method_name,
                linewidth=1.5,
                markersize=5,
                alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    # ax.set_title('Clustering Performance Comparison')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Set x-ticks to specific values
    ax.set_xticks([2, 5, 10, 15, 20, 25])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    plot_silhouette_comparison(save_path="results/report_figures/silhouette_comparison.png")

