import nbformat as nbf

paths = [
    "eda.ipynb",
    "ctms_general.ipynb",
    "embeddings_general.ipynb",
    "part2.ipynb",
]

output = "main_notebook.ipynb"

merged = nbf.v4.new_notebook()
merged_cells = []

for path in paths:
    print(f"Adding: {path}")
    nb = nbf.read(path, as_version=4)
    merged_cells.extend(nb.cells)

merged["cells"] = merged_cells
nbf.write(merged, output)

print(f"\nMerged notebook written to: {output}")
