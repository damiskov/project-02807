import pandas as pd

def inspect_seq(
    path: str = "dataset_construction/data/sequences/sequence_dataset.parquet"
):
    df = pd.read_parquet(path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    print("Columns:", df.columns.tolist())
    print("\nSample rows:")
    print(df.sample(5))

def inspect_emb(
    path: str = "dataset_construction/data/embeddings_batched/embedding_dataset.parquet"
):
    df = pd.read_parquet(path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head(5))
    # shape of embedding columns
    print("\nEmbedding shapes:")
    print("AST embedding shape:", df.iloc[0]["ast"].shape)
    print("WavLM embedding shape:", df.iloc[0]["wavlm"].shape)
    print("CLAP embedding shape:", df.iloc[0]["clap"].shape)

if __name__ == "__main__":
    inspect_emb(path = "dataset_construction/data/embeddings_batched/embedding_dataset.parquet")
    inspect_emb(path = "dataset_construction/data/embeddings/embedding_dataset.parquet")