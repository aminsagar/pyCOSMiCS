"""
PCA via truncated SVD — equivalent to pcarep() from the MCR-ALS MATLAB toolbox.
"""

import numpy as np


def pcarep(data: np.ndarray, n_components: int):
    """
    Reproduce a data matrix from its first n_components principal components.

    Parameters
    ----------
    data : ndarray, shape (n_rows, n_cols)
    n_components : int

    Returns
    -------
    U : ndarray, shape (n_rows, n_components)  — scores (left singular vectors)
    S : ndarray, shape (n_components, n_components)  — diagonal singular-value matrix
    V : ndarray, shape (n_cols, n_components)  — loadings (right singular vectors)
    d_reprod : ndarray, shape (n_rows, n_cols)  — PCA-reconstructed data
    sd : float  — fitting error as a percentage of the total variance
    """
    U_full, s_full, Vt_full = np.linalg.svd(data, full_matrices=False)

    k = min(n_components, len(s_full))
    U = U_full[:, :k]
    s = s_full[:k]
    S = np.diag(s)
    V = Vt_full[:k, :].T  # (n_cols, k)

    d_reprod = U @ S @ V.T

    sst = np.sum(data ** 2)
    res_ss = np.sum((data - d_reprod) ** 2)
    sd = np.sqrt(res_ss / max(sst, 1e-300)) * 100.0

    return U, S, V, d_reprod, sd
