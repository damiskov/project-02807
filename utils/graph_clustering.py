import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA




def distance_to_similarity(D: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Convert distance matrix D to similarity matrix S using a Gaussian kernel.
    """
    # Heuristic for sigma: median of non-zero distances
    if sigma is None:
        sigma = np.median(dist_matrix[D > 0]) / 2  # or even /3
        print(f"Using sigma = {sigma:.4f}")
    
    S = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(S, 0)  # no self-similarity links
    return S

def to_knn_similarity(matrix: np.ndarray, k: int) -> np.ndarray:

    N = matrix.shape[0]
    result = np.zeros_like(matrix)                
    
    #  diagonal values are -inf so self never gets selected
    diag_backup = np.copy(np.diag(matrix))
    np.fill_diagonal(matrix, -np.inf)
    
    # compute indices of the k largest values per row
    topk_indices = np.argpartition(-matrix, k, axis=1)[:, :k]
    
    # put the original values into the result matrix at those positions
    rows = np.arange(N)[:, None]             
    result[rows, topk_indices] = matrix[rows, topk_indices]
    
    # Restore diagonal 
    np.fill_diagonal(matrix, diag_backup)
    np.fill_diagonal(result, 0)                 
    
    return result


def spectral_clustering (sim_matrix, k):


    # compute Laplacian 
    D = np.diag(sim_matrix.sum(axis=1))
    D_inv_sqrt = fractional_matrix_power(D, -0.5)
    L_sym = np.eye(sim_matrix.shape[0]) - D_inv_sqrt @ sim_matrix @ D_inv_sqrt

    # get the first k eigencevtors

    eigvals, eigvecs = eigh(L_sym)

    # Take the first k eigenvectors
    X = eigvecs[:, :k]

    X_norm = normalize(X, norm='l2', axis=1)


    
    # run k means on the eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_norm)  # X_norm from Laplacian step
    # print("Fitting KMeans with k =", k)

    # embeddings_df['cluster'] = labels
    # embeddings_df['cluster'] = labels



    # score = manual_silhouette_score(X_norm, labels)
    # print(f"Silhouette score for k = {k}: {score:.4f}")

    return labels, X_norm


def reduce_dim(X):
    pca = PCA().fit(X)

    k = 30

    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)

    return X_pca
