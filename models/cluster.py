# NOTE: Placeholder. Actual clustering models will go here.
# Will include: K-Means, DBSCAN, Hierarchical Clustering, etc. Potentially implemented from scratch.

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Literal
from abc import ABC, abstractmethod


# --- Cluster Model Base Class ---
@dataclass
class ClusterModel(ABC):
    """Abstract base class for clustering models."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the clustering model to the data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for the data."""
        pass


# --- In-house KMeans Cluster Model ---

@dataclass
class KMeansClusterModel(ClusterModel):
    """K-Means clustering model implemented from scratch."""

    n_clusters: int
    max_iters: int = 100
    tol: float = 1e-4
    random_state: Optional[int] = None
    centroids: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """Fit the K-Means model to the data."""
        n_samples, n_features = X.shape
        # Randomly initialize centroids
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

        
            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.labels_ = labels
                
            self.centroids = new_centroids
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for the data."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
# TODO: Irene - DBSCAN implementation

@dataclass
class DBSCANClusterModel(ClusterModel):
    ... # Implementation goes here


@dataclass
class HierarchicalCluster(ClusterModel):
    n_clusters: int
    method: Literal["agglomerative", "divisive"] = "agglomerative"
    labels_: Optional[np.ndarray] = None
    fitted: bool = False

    def _compute_centroid(self, X: np.ndarray) -> np.ndarray:
        return np.mean(X, axis=0)
    
    def _cluster_distance(self, X: np.ndarray, c1_idx: List[int], c2_idx: List[int]) -> float:
        """
        Distance between two clusters: d(centroid(C1), centroid(C2)).
        """
        c1 = self._compute_centroid(X[c1_idx])
        c2 = self._compute_centroid(X[c2_idx])
        return np.linalg.norm(c1 - c2)
    
    def _fit_agglomerative(self, X: np.ndarray) -> None:
        """
        Perform agglomerative hierarchical clustering until n_clusters is reached.
        """
        n_samples = X.shape[0]

        # Start with each point as its own cluster (slide 27)
        clusters = [[i] for i in range(n_samples)]

        # Agglomerative merging (slide 27)
        while len(clusters) > self.n_clusters:

            min_dist = float("inf")
            merge_pair = None

            # Compute distance betwenn all cluster pairs
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):

                    dist = self._cluster_distance(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)

            # Merge the two closest clusters
            i, j = merge_pair
            new_cluster = clusters[i] + clusters[j]

            # Remove old clusters and append merged cluster
            clusters.pop(max(i, j))
            clusters.pop(min(i, j))
            clusters.append(new_cluster)

        return clusters

    def _fit_divisive(self, X: np.ndarray):
        """
        Simple divisive hierarchical clustering:
        - Start with one cluster containing all points
        - Repeatedly split the cluster with the largest diameter
        """

        n_samples = X.shape[0]
        clusters = [list(range(n_samples))]

        def cluster_diameter(idx_list):
            """Compute diameter = max distance between any pair."""
            pts = X[idx_list]
            if len(idx_list) <= 1:
                return 0
            # pairwise distances
            dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
            return np.max(dists)

        while len(clusters) < self.n_clusters:
            # Pick cluster with largest diameter
            diameters = [cluster_diameter(c) for c in clusters]
            split_idx = int(np.argmax(diameters))
            cluster_to_split = clusters.pop(split_idx)

            if len(cluster_to_split) <= 1:
                # Can't split further
                clusters.append(cluster_to_split)
                continue

            # Perform a 2-means (k=2) on this cluster for splitting
            pts = X[cluster_to_split]

            # Initialize two farthest points as centroids
            dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
            a, b = np.unravel_index(np.argmax(dists), dists.shape)
            c1, c2 = pts[a], pts[b]

            old_assignments = None
            for _ in range(10):
                # Assign points
                d1 = np.linalg.norm(pts - c1, axis=1)
                d2 = np.linalg.norm(pts - c2, axis=1)
                assignments = (d1 > d2).astype(int)

                # Check convergence
                if old_assignments is not None and np.all(assignments == old_assignments):
                    break
                old_assignments = assignments.copy()

                # Update centroids
                if np.any(assignments == 0):
                    c1 = pts[assignments == 0].mean(axis=0)
                if np.any(assignments == 1):
                    c2 = pts[assignments == 1].mean(axis=0)

            # Create two new clusters
            cluster_A = [cluster_to_split[i] for i in range(len(cluster_to_split)) if assignments[i] == 0]
            cluster_B = [cluster_to_split[i] for i in range(len(cluster_to_split)) if assignments[i] == 1]

            clusters.append(cluster_A)
            clusters.append(cluster_B)

        return clusters

    def fit(self, X: np.ndarray) -> None:
        if self.method == "agglomerative":
            clusters = self._fit_agglomerative(X)
        elif self.method == "divisive":
            clusters = self._fit_divisive(X)
        else:
            raise ValueError("Unknown method: choose 'agglomerative' or 'divisive'")

        # Assign labels
        labels = np.zeros(X.shape[0], dtype=int)
        for cluster_id, idxs in enumerate(clusters):
            labels[idxs] = cluster_id

        self.labels_ = labels
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.labels_
