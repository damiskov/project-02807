# NOTE: Placeholder. Actual clustering models will go here.
# Will include: K-Means, DBSCAN, Hierarchical Clustering, etc. Potentially implemented from scratch.

import numpy as np
import pandas as pd
from tqdm import tqdm
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


@dataclass
class KMeansClusterModel(ClusterModel):
    """K-Means clustering model implemented from scratch."""

    n_clusters: int
    name: str = "kmeans"
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

        for i in tqdm(range(self.max_iters), desc="K-Means fitting"):
            # Assign clusters
            # distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            distances = self.compute_distances_chunked(X, self.centroids)
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
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels_
    
    def compute_distances_chunked(self, X, centroids, chunk_size=256):
        """Chunking distance computation to save memory."""
        n = X.shape[0]
        k = centroids.shape[0]
        distances = np.empty((n, k))
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = X[start:end]
            distances[start:end] = np.linalg.norm(
                chunk[:, None, :] - centroids[None, :, :],
                axis=2
            )
        return distances

    

# --- DBSCAN Cluster Model ---

@dataclass
class DBSCANClusterModel(ClusterModel):
    """DBSCAN clustering model implemented from scratch."""

    name: str = "dbscan"
    eps: float = 0.6
    min_samples: int = 5
    labels_: Optional[np.ndarray] = None
    fitted: bool = False

    # Internal states
    UNVISITED: int = -99
    NOISE: int = -1

    def fit(self, X: np.ndarray) -> None:
        """Assign cluster labels using the DBSCAN algorithm."""
        n = X.shape[0]
        self.labels_ = np.full(n, self.UNVISITED, dtype=int)
        cluster_id = 0

        for i in range(n):
            if self.labels_[i] != self.UNVISITED:
                continue

            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = self.NOISE
                continue

            self._expand_cluster(X, i, neighbors, cluster_id)
            cluster_id += 1

        self.fitted = True

    def _region_query(self, X: np.ndarray, idx: int) -> List[int]:
        """Return all points within eps distance of point idx."""
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def _expand_cluster(
        self,
        X: np.ndarray,
        seed_idx: int,
        neighbors: List[int],
        cluster_id: int,
    ) -> None:
        """Grow a new cluster starting from the seed point."""
        self.labels_[seed_idx] = cluster_id
        queue = list(neighbors)
        q = 0

        while q < len(queue):
            j = queue[q]

            if self.labels_[j] == self.UNVISITED:
                self.labels_[j] = cluster_id
                new_neighbors = self._region_query(X, j)
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)

            elif self.labels_[j] == self.NOISE:
                self.labels_[j] = cluster_id

            q += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return cluster labels assigned during fit()."""
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.labels_
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels_


# --- In-house KMeans Cluster Model ---

# updated class - previous one was way too slow



@dataclass
class HierarchicalClusterModel:
    n_clusters: int
    linkage: Literal["single", "complete", "average", "centroid"] = "average"

    labels_: Optional[np.ndarray] = None
    fitted: bool = False

    def _update_distance_matrix(self, D, idx_a, idx_b, cluster_sizes, method):
        """
        Update condensed distance matrix D after merging clusters a and b.
        New cluster index becomes idx_a; idx_b is removed.
        """
        n = D.shape[0]
        mask = np.ones(n, dtype=bool)
        mask[idx_b] = False

        # Distances to new merged cluster
        d_a = D[idx_a]
        d_b = D[idx_b]

        match method:
            case "single":
                new_dist = np.minimum(d_a, d_b)
            case "complete":
                new_dist = np.maximum(d_a, d_b)
            case "average":
                na = cluster_sizes[idx_a]
                nb = cluster_sizes[idx_b]
                new_dist = (na * d_a + nb * d_b) / (na + nb)
            case "centroid":
                na = cluster_sizes[idx_a]
                nb = cluster_sizes[idx_b]
                new_dist = np.sqrt(
                    (na * d_a**2 + nb * d_b**2) / (na + nb)
                )
            case _:
                raise ValueError(f"Unknown linkage: {method}")

        # Remove self-distance
        new_dist[idx_a] = 0.0
        new_dist[idx_b] = 0.0

        # Update row idx_a
        D[idx_a] = D[:, idx_a] = new_dist

        # Remove row/column idx_b
        D = D[mask][:, mask]

        return D, mask


    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]

        # Initial clusters: each point is its own cluster
        cluster_ids = np.arange(n_samples)
        cluster_sizes = np.ones(n_samples, dtype=int)

        # Precompute full distance matrix
        diff = X[:, None, :] - X[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(D, np.inf)    # prevent zero-distance merges

        active = np.ones(n_samples, dtype=bool)

        # Agglomeration loop
        num_active = n_samples
        while num_active > self.n_clusters:
            # Find closest pair
            D_active = np.where(active[:, None] & active[None, :], D, np.inf)
            i, j = np.unravel_index(np.argmin(D_active), D_active.shape)
            if i > j:
                i, j = j, i

            # Merge j into i
            cluster_sizes[i] += cluster_sizes[j]

            # Lance–Williams update
            for k in range(n_samples):
                if not active[k] or k == i:
                    continue

                match self.linkage:
                    case "single":
                        D[i, k] = D[k, i] = min(D[i, k], D[j, k])
                    case "complete":
                        D[i, k] = D[k, i] = max(D[i, k], D[j, k])
                    case "average":
                        na = cluster_sizes[i]
                        nb = cluster_sizes[j]
                        D[i, k] = D[k, i] = (
                            (na - nb) * D[i, k] + nb * D[j, k]
                        ) / na
                    case "centroid":
                        na = cluster_sizes[i]
                        nb = cluster_sizes[j]
                        D[i, k] = D[k, i] = np.sqrt(
                            (na - nb) / na * D[i, k]**2
                            + (nb / na) * D[j, k]**2
                        )

            # Disable j
            active[j] = False
            D[j, :] = np.inf
            D[:, j] = np.inf

            num_active -= 1

        # Assign labels
        labels = np.zeros(n_samples, dtype=int)
        cluster_map = {idx: cid for cid, idx in enumerate(np.where(active)[0])}
        for i in range(n_samples):
            # walk up to active cluster
            k = i
            while not active[k]:
                # belongs to the cluster it merged into
                # choose nearest surviving cluster for simplicity
                k = np.argmin(D[i])
            labels[i] = cluster_map[k]

        self.labels_ = labels
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        return self.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels_



# @dataclass
# class HierarchicalClusterModel(ClusterModel):

#     n_clusters: int
#     name: str = "hierarchical"
#     method: Literal["agglomerative", "divisive"] = "agglomerative"
#     labels_: Optional[np.ndarray] = None
#     fitted: bool = False

#     def _compute_centroid(self, X: np.ndarray) -> np.ndarray:
#         return np.mean(X, axis=0)
    
#     def _cluster_distance(self, X: np.ndarray, c1_idx: List[int], c2_idx: List[int]) -> float:
#         """
#         Distance between two clusters: d(centroid(C1), centroid(C2)).
#         """
#         c1 = self._compute_centroid(X[c1_idx])
#         c2 = self._compute_centroid(X[c2_idx])
#         return np.linalg.norm(c1 - c2)
    
#     def _fit_agglomerative(self, X: np.ndarray) -> None:
#         """
#         Perform agglomerative hierarchical clustering until n_clusters is reached.
#         """
#         n_samples = X.shape[0]

#         # Start with each point as its own cluster (slide 27)
#         clusters = [[i] for i in range(n_samples)]

#         # Agglomerative merging (slide 27)
#         while len(clusters) > self.n_clusters:

#             min_dist = float("inf")
#             merge_pair = None

#             # Compute distance betwenn all cluster pairs
#             for i in range(len(clusters)):
#                 for j in range(i + 1, len(clusters)):

#                     dist = self._cluster_distance(X, clusters[i], clusters[j])
#                     if dist < min_dist:
#                         min_dist = dist
#                         merge_pair = (i, j)

#             # Merge the two closest clusters
#             i, j = merge_pair
#             new_cluster = clusters[i] + clusters[j]

#             # Remove old clusters and append merged cluster
#             clusters.pop(max(i, j))
#             clusters.pop(min(i, j))
#             clusters.append(new_cluster)

#         return clusters

#     def _fit_divisive(self, X: np.ndarray):
#         """
#         Simple divisive hierarchical clustering:
#         - Start with one cluster containing all points
#         - Repeatedly split the cluster with the largest diameter
#         """

#         n_samples = X.shape[0]
#         clusters = [list(range(n_samples))]

#         def cluster_diameter(idx_list):
#             """Compute diameter = max distance between any pair."""
#             pts = X[idx_list]
#             if len(idx_list) <= 1:
#                 return 0
#             # pairwise distances
#             dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
#             return np.max(dists)

#         while len(clusters) < self.n_clusters:
#             # Pick cluster with largest diameter
#             diameters = [cluster_diameter(c) for c in clusters]
#             split_idx = int(np.argmax(diameters))
#             cluster_to_split = clusters.pop(split_idx)

#             if len(cluster_to_split) <= 1:
#                 # Can't split further
#                 clusters.append(cluster_to_split)
#                 continue

#             # Perform a 2-means (k=2) on this cluster for splitting
#             pts = X[cluster_to_split]

#             # Initialize two farthest points as centroids
#             dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
#             a, b = np.unravel_index(np.argmax(dists), dists.shape)
#             c1, c2 = pts[a], pts[b]

#             old_assignments = None
#             for _ in range(10):
#                 # Assign points
#                 d1 = np.linalg.norm(pts - c1, axis=1)
#                 d2 = np.linalg.norm(pts - c2, axis=1)
#                 assignments = (d1 > d2).astype(int)

#                 # Check convergence
#                 if old_assignments is not None and np.all(assignments == old_assignments):
#                     break
#                 old_assignments = assignments.copy()

#                 # Update centroids
#                 if np.any(assignments == 0):
#                     c1 = pts[assignments == 0].mean(axis=0)
#                 if np.any(assignments == 1):
#                     c2 = pts[assignments == 1].mean(axis=0)

#             # Create two new clusters
#             cluster_A = [cluster_to_split[i] for i in range(len(cluster_to_split)) if assignments[i] == 0]
#             cluster_B = [cluster_to_split[i] for i in range(len(cluster_to_split)) if assignments[i] == 1]

#             clusters.append(cluster_A)
#             clusters.append(cluster_B)

#         return clusters

#     def fit(self, X: np.ndarray) -> None:
#         if self.method == "agglomerative":
#             clusters = self._fit_agglomerative(X)
#         elif self.method == "divisive":
#             clusters = self._fit_divisive(X)
#         else:
#             raise ValueError("Unknown method: choose 'agglomerative' or 'divisive'")

#         # Assign labels
#         labels = np.zeros(X.shape[0], dtype=int)
#         for cluster_id, idxs in enumerate(clusters):
#             labels[idxs] = cluster_id

#         self.labels_ = labels
#         self.fitted = True

#     def predict(self, X: np.ndarray) -> np.ndarray:
#         if not self.fitted:
#             raise RuntimeError("Model is not fitted yet.")
#         return self.labels_
