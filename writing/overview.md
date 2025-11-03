# Project Title: Graph-Enhanced Clustering of Classical Music Using Note-Sequence and Structural Similarity

## Project Statement:
- This project investigates how graph-based clustering can reveal stylistic similarities among classical music pieces when compared to traditional clustering algorithms.
- Using the MusicNet dataset, each composition will be represented by a Chord Trajectory Matrix and a Shingle/MinHash similarity graph.

We will apply both:
- General clustering algorithms (K-means, DBSCAN, Hierarchical, CURE) — topics from Week 6
- Graph clustering algorithms (Spectral, Louvain) — topics from Week 7

and extend these with a novel technique not covered in the lectures:
- Learning low-dimensional graph embeddings using node2vec and clustering in the embedding space.


This approach allows us to compare classical clustering in feature space with community detection in a learned music-similarity network.

| **Category**          | **Algorithms**                                                         | **Source**                 |
| --------------------- | ---------------------------------------------------------------------- | -------------------------- |
| From course           | K-means, DBSCAN, Hierarchical, CURE                                    | Week 6 lectures            |
| From course           | Spectral clustering, Louvain community detection, Modularity           | Week 7 lectures            |
| Beyond course (novel) | Graph embedding (*node2vec*) for low-dimensional music representations | Extension; not in syllabus |


## Dataset: MusicNet
- 330 classical compositions with annotated notes, instruments, and composer metadata.

## Feature Extraction:
- Chord Trajectory Matrix (CTM) – 128×128 transition matrix of MIDI notes + pauses (as in Wang & Haque).
- Shingle/MinHash similarity – k-shingles over note sequences -> LSH to compute approximate pairwise similarities.

## Graph representation:
- Nodes = music pieces
- Edges = similarity weights from LSH or Frobenius norm of CTMs.

## Methods Overview
| Step | Method                                     | Purpose                                                       |
| ---- | ------------------------------------------ | ------------------------------------------------------------- |
| 1    | **K-means, Hierarchical, DBSCAN, CURE**    | Classical vector-space clustering baseline                    |
| 2    | **Spectral Clustering, Louvain Algorithm** | Graph-based community detection                               |
| 3    | **node2vec Embedding (NEW)**               | Learn structural representations of compositions              |
| 4    | **Cluster Validation**                     | Silhouette, Davies–Bouldin, Modularity, and cluster stability |
| 5    | **Visualization**                          | UMAP or MDS to plot clusters in 2-D embedding space           |
| 6    | **Interpretation**                         | Relate clusters to composer, era, or instrumentation metadata |


## Roadmap & Deliverables

### Week 1 – Data Preparation & Feature Engineering
- Download MusicNet, parse interval trees to note sequences.
- Implement Chord Trajectory Matrix and Shingling feature extraction.
- Construct pairwise similarity matrix → affinity graph.

### Week 2 – Baseline Clustering (Topics from Week 6)

- Apply K-means, DBSCAN, Hierarchical, and CURE to feature vectors.
- Evaluate with Silhouette and Davies-Bouldin indices.
- Discuss how distance definitions (Euclidean vs Frobenius) affect results.

### Week 3 – Graph Clustering (Topics from Week 7)

- Build affinity graph and apply Spectral Clustering and Louvain.
- Compute modularity, betweenness centrality, and compare with Week 2 results.
- Visualize clusters using force-directed layouts.

### Week 4 – Novel Extension: Graph Embedding (node2vec)
- Learn low-dimensional embeddings from the graph.
- Cluster embeddings with K-means and DBSCAN.
- Evaluate with the same metrics → determine if embedding improves cohesion.

### Week 5 – Analysis & Reporting
- Summarize comparative results (table of scores).
- Produce visuals (MDS/UMAP plots, composer color-coded clusters).
- Write report + Jupyter Notebook appendix.


## Evaluation Plan
| Metric                    | Type            | Interpretation                          |
| ------------------------- | --------------- | --------------------------------------- |
| **Silhouette Score**      | Internal        | Cohesion vs separation in feature space |
| **Davies–Bouldin Index**  | Internal        | Compactness vs distance (Week 6 topic)  |
| **Modularity**            | Graph-theoretic | Community structure strength            |
| **Stability (Bootstrap)** | Novel optional  | Robustness under noise or sample change |

## Possible Stretch Goals (if time allows)
- Compare CTM + Shingle fusion (multi-view spectral clustering).
- Add temporal motifs (2- or 3-note patterns) to extend the CTM.
- Perform cluster significance testing via randomized graphs.

## Deliverables Summary
| Deliverable                 | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| **Report (≤ 11 000 chars)** | Problem motivation, data prep, methods, results, discussion |
| **Jupyter Notebook**        | Full pipeline with clear section headers and docstrings     |
| **Appendix**                | Contribution outline + instructions to run code             |
| **Figures / Tables**        | Cluster visualizations and metric comparison table          |
