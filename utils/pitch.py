from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pitch_class_normalization(sequence: List[int]) -> List[int]:
    """Convert a sequence of MIDI pitches to pitch classes (0-11), removing pauses (128)."""
    seq = np.array(sequence)
    return [p % 12 for p in seq if p != 128]
 

def pitch_class_trajectory_matrix(sequence: List[int]) -> np.ndarray:
    """Construct 12x12 pitch class trajectory matrix."""
    M = np.zeros((12, 12), dtype=np.int32)
    
    pitch_classes = pitch_class_normalization(sequence)
    
    for i in range(len(pitch_classes) - 1):
        a, b = pitch_classes[i], pitch_classes[i + 1]
        M[a, b] += 1
    
    return M

def normalize_ctm(M):
    M = M.astype(float)
    row_sums = M.sum(axis=1, keepdims=True)
    M = np.divide(M, row_sums, where=row_sums != 0)
    return np.log1p(M)



def interval_histogram(
    sequence: List[int],
    max_interval: int = 12
) -> np.ndarray:
    """
    Compute a normalized histogram of pitch intervals (in semitones).

    Args:
        sequence (list[int]): MIDI pitch sequence (0-127). Pauses coded as 128.
        max_interval (int): Maximum interval magnitude to include (symmetrical range [-k, +k]).

    Returns:
        np.ndarray: 1D array of length 2*max_interval + 1 representing
                    normalized interval frequencies from -max_interval to +max_interval.
                    (Bin order: [-max_interval, ..., -1, 0, +1, ..., +max_interval])
    """
    # Filter out pauses
    pitches = [p for p in sequence if p != 128]
    if len(pitches) < 2:
        return np.zeros(2 * max_interval + 1, dtype=float)

    diffs = np.diff(pitches)

    # Clip extreme jumps to the boundary bins
    diffs = np.clip(diffs, -max_interval, max_interval)

    # Map intervals to histogram bins
    bins = np.arange(-max_interval - 0.5, max_interval + 1.5)
    hist, _ = np.histogram(diffs, bins=bins)

    # Normalize to probability distribution
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return hist


def extract_pitch_based_features(sequence: List[int], max_interval: int = 12) -> np.ndarray:
    """
    Extract pitch-based features from a MIDI pitch sequence.

    Args:
        sequence (list[int]): MIDI pitch sequence (0-127). Pauses coded as 128.
        max_interval (int): Maximum interval magnitude for interval histogram.

    Returns:
        np.ndarray: 1D array of extracted features.
    """
    # Compute pitch class trajectory matrix
    pc_matrix = pitch_class_trajectory_matrix(sequence)

    # NOTE: maybe remove?
    pc_matrix = normalize_ctm(pc_matrix)

    # Compute interval histogram
    interval_hist = interval_histogram(sequence, max_interval=max_interval)

    # Combine features into a single feature vector
    features = np.concatenate([pc_matrix.flatten(), interval_hist])
    return features

def plot_pca_variance(pca: PCA):
    """Plot explained variance ratio from PCA object."""

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.show()


def plot_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Clusters Visualization"
):
    """Scatter plot of clustered data."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with colors based on cluster labels
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()