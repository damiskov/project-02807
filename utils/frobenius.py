from utils.load import load_dataset

def frobenius_norm(matrix):
    """Calculate the Frobenius norm of a matrix."""
    return sum(sum(cell**2 for cell in row) for row in matrix) ** 0.5

if __name__ == "__main__":
    matrices, meta = load_dataset()
    # apply the frobenius norm to each matrix and add as a new column
    meta['frobenius_norm'] = matrices['matrix'].apply(frobenius_norm)
    print(meta[['composer', 'frobenius_norm']].head())