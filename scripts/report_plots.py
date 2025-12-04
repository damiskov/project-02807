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
    #   "20": 0.2502384818544209,
    #   "25": 0.23527212132742312
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
    # "20": 0.09797404706478119,
    # "25": 0.09526319056749344
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
    # "20": 0.09252970665693283,
    # "25": 0.09114072471857071
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
    # "20": 0.07526630908250809,
    # "25": 0.06724324822425842
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
    "11": 0.332,
    "12": 0.334,
    "13": 0.321,
    "14": 0.304
}

ast_graph =  {'2': 0.6812263102451612,
  '3': 0.579870490238299,
  '4': 0.551317788589451,
  '5': 0.5261730667243093,
  '6': 0.4675932516618964,
  '7': 0.48403946344446935,
  '8': 0.4392262604920386,
  '9': 0.3806574550632817,
  '10': 0.36308145435945866,
  '11': 0.3510443850373366,
  '12': 0.31432470270116447,
  '13': 0.32365079657486634,
  '14': 0.3139509374601127,
  '15': 0.28722175755196067,
#   '16': 0.2867905543026262,
#   '17': 0.3002259030883523,
#   '18': 0.2918465736767373,
#   '19': 0.27827288308526227,
#   '20': 0.26529877863036555,
#   '21': 0.2668797228329107,
#   '22': 0.25373505726269496,
#   '23': 0.24558953716286505,
#   '24': 0.22545262438813823,
#   '25': 0.2335707264957377
}
wavlm_graph = {'2': 0.668867198057312,
  '3': 0.5861993770026649,
  '4': 0.5126876379491748,
  '5': 0.49310881567596826,
  '6': 0.4570888167416043,
  '7': 0.4082123309914841,
  '8': 0.39867050613644855,
  '9': 0.37553117652016405,
  '10': 0.3483842624891384,
  '11': 0.31551286554744096,
  '12': 0.2936670257487253,
  '13': 0.2725810002506747,
  '14': 0.27710399908295064,
  '15': 0.2619977601236882,
#   '16': 0.24023402005445196,
#   '17': 0.2408525935646164,
#   '18': 0.22105303021075595,
#   '19': 0.2194769103652485,
#   '20': 0.22707193337108317,
#   '21': 0.21438908675017843,
#   '22': 0.21381432962027186,
#   '23': 0.20189570605312987,
#   '24': 0.19645154059374467,
#   '25': 0.19655416452215568
}
clap_graph = {'2': 0.7574781029681097,
  '3': 0.6661869744616714,
  '4': 0.6000996501228222,
  '5': 0.44077495950340134,
  '6': 0.4141712813025827,
  '7': 0.48864552994655225,
  '8': 0.43165906624250855,
  '9': 0.410524623059109,
  '10': 0.34316696142706465,
  '11': 0.35265460135483,
  '12': 0.3457635103439385,
  '13': 0.3502860928540664,
  '14': 0.31690218070177134,
  '15': 0.32749673265492907,
#   '16': 0.2890933200977463,
#   '17': 0.2997617317087263,
#   '18': 0.2789038118342688,
#   '19': 0.2558227917750422,
#   '20': 0.2662306730202122,
#   '21': 0.24881718789204413,
#   '22': 0.2520695802020154,
#   '23': 0.2557398351219226,
# '24': 0.25458934298560837,
#   '25': 0.2473889978248277
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
    ax.set_xticks([2, 5, 10, 15])
    ax.set_xlim(1, 16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    plot_silhouette_comparison(save_path="results/report_figures/silhouette_comparison.png")

