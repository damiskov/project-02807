import numpy as np
from scipy.stats import chi2

def mahalanobis_mask(X_pca, alpha=0.001):
    """
    Remove multivariate outliers using Mahalanobis distance thresholding.
    alpha: significance level (0.001 ~ 3.3std equivalent)
    """
    cov = np.cov(X_pca, rowvar=False)
    cov_inv = np.linalg.inv(cov)
    mean = X_pca.mean(axis=0)

    d2 = np.array([
        (x - mean).T @ cov_inv @ (x - mean)
        for x in X_pca
    ])

    threshold = chi2.ppf(1 - alpha, df=X_pca.shape[1])
    mask = d2 < threshold
    return mask, d2, threshold