import json
import numpy as np
import pandas as pd
from loguru import logger

from typing import Dict, Union, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_cluster_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    k: int = 20,
    ngram_range=(1, 2),
    min_df: int = 1,
    max_df: float = 0.8,
    scoring: str = "distinctiveness",  # "distinctiveness" or "tfidf"
) -> Dict[Union[int, str], List[str]]:
    """
    Compute top-k keywords per cluster from the 'metadata' column.

    uses a global TF-IDF vectorizer fitted on all documents, then computes
    mean TF-IDF per cluster to find distinctive terms.

    Args:
        df: DataFrame containing 'cluster_label' and 'metadata' (or 'metadata_combined').
        k: Top-k terms to return per cluster.
        ngram_range: N-gram range for TF-IDF.
        min_df: Minimum doc frequency (absolute).
        max_df: Max document frequency ratio.
        scoring: 'distinctiveness' (default) or 'tfidf' (cluster mean TF-IDF).
    """

    if "cluster_label" not in df.columns:
        raise ValueError("DataFrame must contain 'cluster_label'.")

    if "metadata" in df.columns:
        text_col = "metadata"
    elif "metadata_combined" in df.columns:
        text_col = "metadata_combined"
    else:
        raise ValueError("DataFrame must contain 'metadata' or 'metadata_combined'.")

    all_texts = df[text_col].fillna("").astype(str).tolist()
    if len(all_texts) == 0 or all(t.strip() == "" for t in all_texts):
        return {}

    n_docs = len(all_texts)

    # Ensure max_df is consistent with min_df for small n_docs
    if isinstance(max_df, float) and max_df < 1.0:
        min_allowed_max_df = min_df / n_docs
        if max_df < min_allowed_max_df:
            max_df = min_allowed_max_df

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    try:
        tfidf_all = vectorizer.fit_transform(all_texts)
    except ValueError as e:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=ngram_range,
            min_df=1,
            max_df=1.0,
        )
        tfidf_all = vectorizer.fit_transform(all_texts)

    if tfidf_all.shape[1] == 0:
        return {}

    feature_names = np.array(vectorizer.get_feature_names_out())
    global_mean = np.asarray(tfidf_all.mean(axis=0)).ravel()

    results: Dict[Union[int, str], List[str]] = {}
    clusters = df["cluster_label"].unique()

    for cluster in clusters:
        idx = np.where(df["cluster_label"].values == cluster)[0]
        if len(idx) == 0:
            results[cluster] = []
            continue

        tfidf_cluster = tfidf_all[idx, :]
        if tfidf_cluster.shape[0] == 0:
            results[cluster] = []
            continue

        cluster_mean = np.asarray(tfidf_cluster.mean(axis=0)).ravel()

        if scoring == "tfidf":
            scores = cluster_mean
        else:
            scores = cluster_mean - global_mean

        if scores.size == 0:
            results[cluster] = []
            continue

        top_idx = np.argsort(scores)[::-1][:k]
        results[cluster] = feature_names[top_idx].tolist()

    # JSON can't handle np.int64 keys
    results = {int(c): v for c, v in results.items()}

    if save_path:
        logger.info(f"Saving TF-IDF cluster summary to {save_path}")
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
            # clean and combine
            texts = (
                df_cluster[col]
                .fillna("")
                .astype(str)
                .tolist()
            )

            if len(texts) == 0 or all(t.strip() == "" for t in texts):
                results[cluster][col] = []
                continue

            # Fit TF-IDF on cluster texts
            vectorizer = TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
            except ValueError:
                logger.error(f"TF-IDF fitting failed for cluster {cluster} with {len(texts)} documents.")
                results[cluster][col] = []
                continue

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
