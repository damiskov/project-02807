import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def inspect_seq(
    path: str = "dataset_construction/data/sequences/sequence_dataset.parquet"
):
    df = pd.read_parquet(path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    print("Columns:", df.columns.tolist())
    print("\nSample rows:")
    df["ctm"] = df["ctm"].apply(lambda x: np.stack(x, axis=0).astype(np.int32))
    print(df.sample(5))
    # print ctm sample
    print("\nCTM sample shape:")
    print(df.iloc[0]["ctm"].shape)
    ctm_0 = df.iloc[0]["ctm"]
    print(type(ctm_0))
    print(ctm_0)

    # heatmap of ctm
    plt.imshow(ctm_0, cmap="hot", interpolation="nearest")
    plt.title("CTM Heatmap Sample")
    plt.colorbar()
    plt.show()


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
    inspect_seq(path = "dataset_construction/data/sequences/sequence_dataset.parquet")
    # inspect_emb(path = "dataset_construction/data/embeddings/embedding_dataset.parquet")