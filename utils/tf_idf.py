import json
import numpy as np
import pandas as pd

from typing import Dict, Union, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_cluster_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    k: int = 20,
    ngram_range=(1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
) -> Dict[Union[int, str], List[str]]:
    """
    Compute top-k TF-IDF words for each cluster using the metadata column.

    Args:
        df : pd.DataFrame
            Must contain ['cluster_label', 'metadata_combined'].
        k : int
            Number of top words to return per cluster.
        ngram_range : tuple
            Size of n-grams used by TF-IDF.
        min_df : int
            Minimum number of documents a word must appear in.
        max_df : float
            Ignore words appearing in more than max_df fraction of docs.

    Returns
        dict : {cluster_label: [top_words]}
    """

    if "cluster_label" not in df or "metadata" not in df:
        raise ValueError(
            "DataFrame must contain 'cluster_label' and 'metadata'."
        )

    results = {}
    clusters = df["cluster_label"].unique()

    for cluster in clusters:
        df_cluster = df[df["cluster_label"] == cluster]
        texts = df_cluster["metadata"].fillna("").astype(str).tolist()

        # ignore empty texts
        if len(texts) == 0 or all(t.strip() == "" for t in texts):
            results[cluster] = []
            continue

        # Build TF-IDF model for this cluster
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

        tfidf_matrix = vectorizer.fit_transform(texts)

        # Aggregate word importance across docs
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Select top k
        if len(scores) == 0:
            results[cluster] = []
        else:
            top_idx = scores.argsort()[::-1][:k]
            results[cluster] = feature_names[top_idx].tolist()

    # json can't handle np.int64 keys
    results = {int(k): v for k, v in results.items()}

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

    return results


def tfidf_cluter_per_column(
    df: pd.DataFrame,
    text_columns: List[str],
    save_path: Optional[str] = None,
    k: int = 15,
    ngram_range=(1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
) -> Dict[Union[int, str], Dict[str, List[str]]]:
    """
    Compute top-k TF-IDF keywords for each cluster and each specified text column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column named 'cluster_label'.
    text_columns : list of str
        Column names containing text.
    k : int
        Number of top words to return per column per cluster.
    ngram_range : tuple
        N-grams for TF-IDF vectorizer (default (1,2)).
    min_df : int
        Minimum document frequency.
    max_df : float
        Maximum document frequency ratio.

    Returns
    -------
    dict
        Structure:
        {
            cluster_label_1: {
                "name": [...top_k_words...],
                "themes": [...],
                ...
            },
            cluster_label_2: {
                ...
            }
        }
    """

    if "cluster_label" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster_label' column.")

    results = {}
    clusters = df["cluster_label"].unique()

    for cluster in clusters:
        df_cluster = df[df["cluster_label"] == cluster]
        results[cluster] = {}

        for col in text_columns:
            # Clean & combine all documents in this cluster
            texts = (
                df_cluster[col]
                .fillna("")
                .astype(str)
                .tolist()
            )

            if len(texts) == 0 or all(t.strip() == "" for t in texts):
                results[cluster][col] = []
                continue

            # Fit TF-IDF for the cluster
            vectorizer = TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Sum TF-IDF values across all docs in the cluster
            word_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            feature_names = np.array(vectorizer.get_feature_names_out())

            if len(word_scores) == 0:
                results[cluster][col] = []
                continue

            # Top-k words
            top_idx = word_scores.argsort()[::-1][:k]
            top_words = feature_names[top_idx].tolist()

            results[cluster][col] = top_words

    # json can't handle np.int64 keys
    results = {int(k): v for k, v in results.items()}

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

    return results
