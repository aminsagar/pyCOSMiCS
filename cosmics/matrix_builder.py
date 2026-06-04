"""
Build the augmented input matrix and initial-estimate matrix sent to ALS.
Translated from crearMat_manualpure_rank.m.

The four SAXS representations (Absolute, Holtzer, Kratky, Porod) are
concatenated column-wise, scaled so that all representations contribute
equally (scaling by the ratio of first eigenvalues).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .pca import pcarep


COMBINATIONS = [
    [True,  False, False, False],  # A
    [True,  True,  False, False],  # A+H
    [True,  False, True,  False],  # A+K
    [True,  False, False, True ],  # A+P
    [True,  True,  True,  False],  # A+H+K
    [True,  True,  False, True ],  # A+H+P
    [True,  False, True,  True ],  # A+K+P
    [True,  True,  True,  True ],  # A+H+K+P
]

COMBINATION_NAMES = ["A", "A+H", "A+K", "A+P", "A+H+K", "A+H+P", "A+K+P", "A+H+K+P"]


def _first_eigenvalue(matrix: np.ndarray) -> float:
    """Return the largest singular value of *matrix*."""
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return float(s[0])


def build_input_matrix(
    mat_abs: np.ndarray,
    mat_holtzer: np.ndarray,
    mat_kratky: np.ndarray,
    mat_porod: np.ndarray,
    combination: List[bool],
    ev_abs: float,
) -> Tuple[np.ndarray, List[int], float, float, float]:
    """
    Concatenate the selected representations into one augmented matrix.

    All non-absolute representations are scaled so their first eigenvalue
    matches *ev_abs* (the first eigenvalue of the absolute matrix).

    Parameters
    ----------
    mat_abs, mat_holtzer, mat_kratky, mat_porod
        Shape (n_curves, n_points_*) — rows are curves, columns are q-points.
    combination : [use_abs, use_holtzer, use_kratky, use_porod]
    ev_abs : first eigenvalue of mat_abs (pre-computed to avoid recomputing).

    Returns
    -------
    mat_input   : augmented matrix (n_curves, total_points)
    puntos      : list of n_points for each included block
    scale_h, scale_k, scale_p : scaling factors applied (1.0 if not used)
    """
    scale_h = scale_k = scale_p = 1.0

    mat_input = None
    puntos: List[int] = []

    if combination[0]:
        mat_input = mat_abs.copy()
        puntos.append(mat_abs.shape[1])

    if combination[1]:
        ev_h = _first_eigenvalue(mat_holtzer)
        scale_h = ev_abs / ev_h if ev_h > 0 else 1.0
        mh = mat_holtzer * scale_h
        mat_input = mh if mat_input is None else np.hstack([mat_input, mh])
        puntos.append(mat_holtzer.shape[1])

    if combination[2]:
        ev_k = _first_eigenvalue(mat_kratky)
        scale_k = ev_abs / ev_k if ev_k > 0 else 1.0
        mk = mat_kratky * scale_k
        mat_input = mk if mat_input is None else np.hstack([mat_input, mk])
        puntos.append(mat_kratky.shape[1])

    if combination[3]:
        ev_p = _first_eigenvalue(mat_porod)
        scale_p = ev_abs / ev_p if ev_p > 0 else 1.0
        mp = mat_porod * scale_p
        mat_input = mp if mat_input is None else np.hstack([mat_input, mp])
        puntos.append(mat_porod.shape[1])

    return mat_input, puntos, scale_h, scale_k, scale_p


def build_initial_estimates(
    mat_abs: np.ndarray,
    mat_holtzer: np.ndarray,
    mat_kratky: np.ndarray,
    mat_porod: np.ndarray,
    combination: List[bool],
    pure_components: List[int],
    scale_h: float = 1.0,
    scale_k: float = 1.0,
    scale_p: float = 1.0,
) -> np.ndarray:
    """
    Build the initial-estimate matrix by extracting rows (curves) from each
    representation block, applying the same scale factors used in build_input_matrix.

    Returns ndarray of shape (n_species, total_points).
    """
    n_species = len(pure_components)
    rows: List[np.ndarray] = []

    for sp in range(n_species):
        idx = pure_components[sp]
        row: Optional[np.ndarray] = None

        if combination[0]:
            row = mat_abs[idx : idx + 1, :]
        if combination[1]:
            h_row = mat_holtzer[idx : idx + 1, :] * scale_h
            row = h_row if row is None else np.hstack([row, h_row])
        if combination[2]:
            k_row = mat_kratky[idx : idx + 1, :] * scale_k
            row = np.hstack([row, k_row])
        if combination[3]:
            p_row = mat_porod[idx : idx + 1, :] * scale_p
            row = np.hstack([row, p_row])

        rows.append(row)

    return np.vstack(rows)  # (n_species, total_points)


def build_ssel_fixed(
    mat_input: np.ndarray,
    n_species: int,
    pure_components: List[int],
    fixed_species: List[bool],
) -> np.ndarray:
    """
    Build the spectra equality-constraint matrix ssel.
    Fixed species rows are set to the corresponding row of mat_input;
    free species rows are filled with NaN.
    """
    ssel = np.full((n_species, mat_input.shape[1]), np.nan)
    for sp in range(n_species):
        if fixed_species[sp]:
            idx = pure_components[sp]
            ssel[sp, :] = mat_input[idx, :]
    return ssel


def zero_eliminated_curves(
    mat_absolute: np.ndarray,
    eliminated: List[int],
) -> np.ndarray:
    """Zero-out rows corresponding to eliminated curve indices (0-based)."""
    mat = mat_absolute.copy()
    for idx in eliminated:
        mat[idx, :] = 0.0
    return mat
