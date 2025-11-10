# NOTE: Placeholder. Actual clustering models will go here.
# Will include: K-Means, DBSCAN, Hierarchical Clustering, etc. Potentially implemented from scratch.

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
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