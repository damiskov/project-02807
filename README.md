# project-02807

## Computational Tools for Data Science — Final Project (Group 44)

This repository contains the full codebase for our final project in **02807 — Computational Tools for Data Science**, DTU (Fall 2025).

The project integrates multiple topics from the course curriculum, including:

- General clustering algorithms
- Graph-based clustering/evaluation methods
- Frequent-items / TF–IDF analysis for metadata interpretation
- Similarity search using high-dimensional embedding representations

Additionally, the project implements techniques **beyond the curriculum**, satisfying the requirement for self-directed exploration, including:

- Neural audio embeddings (CLAP, AST, WavLM)
- MIDI feature extraction and Chord Trajectory Matrices (CTMs)
- Dimensionality reduction and outlier-aware clustering pipelines
- Custom visualisation and evaluation tooling

NOTE: Custom implementations of various clustering algorithms (Hierarchical, DBSCAN, KMeans) can be found in `models/`. These are not used when running the `main_notebook.ipynb` due to memory and run-time issues.

> If you have any questions about setting up the environment or issues regarding running the notebook, please feel free to contact s204755@student.dtu.dk

## Environment Setup

This project uses **Python 3.12** and the **uv** package manager.

1. Install Python 3.12  
   Download from: https://www.python.org/downloads/

2. Install uv  
   Instructions: https://docs.astral.sh/uv/getting-started/installation/

3. Create & activate the virtual environment

From the project root:

```bash
uv venv
```

4. Activate it:

- Windows

```bash
.venv/Scripts/activate
```

- Mac/Linux

```bash
source .venv/bin/activate
```

4. Install dependencies

```
uv sync
```

---

## Project Summary

The project investigates how **symbolic music representations** (e.g., CTMs) compare to **neural audio embeddings** (CLAP, AST, WavLM) for **unsupervised clustering of video-game soundtracks**.

We:

- Extracted structured symbolic features from MIDI files
- Generated neural embeddings from WAV audio
- Applied K-Means, DBSCAN, and hierarchical clustering
- Used PCA for dimensionality reduction
- Performed internal validation (silhouette coefficient, DB index, dispersion)
- Used TF–IDF to interpret clusters via metadata (themes, companies, keywords)

All pipeline components are implemented under either `utils/` or `scripts/` and reproduced via notebooks in `notebooks/` or `scripts/`.

Some auxiliary functions (mainly related to plotting) can be found in `scripts/`.

## Reproducibility Notes

- All preprocessing steps are deterministic.
- PCA and clustering use fixed random seeds where supported.
- Figures and validations can be regenerated using the notebooks.
