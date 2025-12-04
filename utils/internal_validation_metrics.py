import numpy as np

def manual_modularity(A, labels):
    """
    Compute modularity manually.
    Parameters
    ----------
    A : ndarray of shape (n_nodes, n_nodes)
        Adjacency matrix of the graph.
    labels : ndarray of shape (n_nodes,)
        Community labels for each node.
    Returns
    -------
    Q : float
        Modularity score.
    """
    m = A.sum() / 2
    k = A.sum(axis=1)  # degree of each node
    Q = 0.0

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        A_c = A[np.ix_(idx, idx)]            # edges inside community
        k_c = k[idx]                         # degrees inside community

        e_c = A_c.sum() / (2*m)
        a_c = k_c.sum() / (2*m)

        Q += e_c - a_c**2

    return Q

import numpy as np

def manual_dbi(X, labels):
    """
    Compute Davies-Bouldin Index manually.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.
    
    Returns
    -------
    dbi : float
        Davies-Bouldin Index (lower is better)
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    # Compute cluster centroids
    centroids = np.array([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
    
    # Compute cluster scatter Si (average distance to centroid)
    S = np.array([np.mean(np.linalg.norm(X[labels == lbl] - centroids[i], axis=1)) 
                  for i, lbl in enumerate(unique_labels)])
    
    # Compute pairwise centroid distances Mij
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                M[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Compute Rij = (Si + Sj) / Mij
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (S[i] + S[j]) / M[i, j]
    
    # For each cluster, take max Rij over j != i
    R_max = np.max(R, axis=1)
    
    # DBI = mean of R_max
    dbi = np.mean(R_max)
    return dbi

def manual_silhouette_score(X, labels):
    """
    Compute silhouette score manually.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    
    Returns
    -------
    silhouette_avg : float
        Mean silhouette score over all points.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    # Precompute distance matrix
    dist_matrix = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
    
    silhouettes = np.zeros(n_samples)
    
    for idx in range(n_samples):
        label = labels[idx]
        same_cluster = np.where(labels == label)[0]
        other_clusters = [np.where(labels == l)[0] for l in unique_labels if l != label]
        
        # a(i): mean distance to other points in same cluster
        if len(same_cluster) > 1:
            a_i = np.mean(dist_matrix[idx, same_cluster[same_cluster != idx]])
        else:
            a_i = 0.0  # cluster of size 1
            
        # b(i): smallest mean distance to points in any other cluster
        b_i = np.min([np.mean(dist_matrix[idx, oc]) for oc in other_clusters])
        
        # silhouette for this point
        silhouettes[idx] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
    
    return np.mean(silhouettes)