# Notebooks

This folder contains notebooks that you can use to test the implementations. Some conventions:
- 1-to-1 mapping of notebook to any type of implementation. E.g., if you are working on the DBSCAN algorithm, you MUST implement this in the `utils` folder (E.g., `utils/dbscan.py`) and test this in a notebook called `notebooks/DBSCAN.ipynb`. This is simply to minimize the possibility of any merge conflicts.
- The dataset is READ ONLY. Everything you do must be relative to the data located in `data/features/*`. If you perform a bunch of transformations and don't want to redo these, simply save you transformed data locally and ADD IT TO THE GITIGNORE.

I have created some placeholder notebooks for demonstration purposes.

David 02-11-2025