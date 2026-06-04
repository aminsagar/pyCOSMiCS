"""
Curve normalisation, scaling, and representation transforms.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def normalize_by_i0(
    intensities: np.ndarray,
    errors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scale all curves so that the minimum I(0) equals 100.
    Mirrors COSMiCS_multi.m lines 217–223.
    """
    i0_min = np.min(intensities[0, :])
    scale = 100.0 / i0_min
    intensities = intensities * scale
    if errors is not None:
        errors = errors * scale
    return intensities, errors


def scale_at_reference(
    intensities: np.ndarray,
    errors: Optional[np.ndarray] = None,
    ref_idx: int = 19,          # 0-based (MATLAB row 20 → index 19)
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Scale every curve by the ratio of its intensity at *ref_idx* to the
    minimum value among all curves at that index.

    Returns the scaled arrays and the per-curve scale factors.
    """
    ref_row = intensities[ref_idx, :]
    ref_min = np.min(ref_row)
    scale_factors = ref_row / ref_min          # shape (n_curves,)
    intensities = intensities / scale_factors[np.newaxis, :]
    if errors is not None:
        errors = errors / scale_factors[np.newaxis, :]
    return intensities, errors, scale_factors


def find_cut_index(q_values: np.ndarray, q_max: float) -> int:
    """Return the first index where q >= q_max (clamped to [1, n])."""
    idx = int(np.searchsorted(q_values, q_max))
    return max(1, min(idx, len(q_values)))


def default_cuts(units: str) -> Tuple[float, float, float, float]:
    """Return default q-cutoffs for each representation."""
    if units == 'N':   # 1/nm
        return 3.0, 1.6, 1.6, 0.7
    else:              # 1/Å  (units == 'A')
        return 0.5, 0.16, 0.16, 0.07


def apply_transforms(
    intensities: np.ndarray,
    q_values: np.ndarray,
    points_abs: int,
    points_holtzer: int,
    points_kratky: int,
    points_porod: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the four SAXS representations (absolute, Holtzer, Kratky, Porod).

    The *intensities* matrix has shape (n_points, n_curves); q_values is
    shape (n_points,). The functions return the representation matrix with
    shape (n_curves, n_points_cut) matching MATLAB convention used by ALS.

    Returns
    -------
    mat_abs, mat_holtzer, mat_kratky, mat_porod  — shape (n_curves, n_pts)
    q_abs, q_holtzer, q_kratky, q_porod           — corresponding q vectors
    """
    q_a = q_values[:points_abs]
    q_h = q_values[:points_holtzer]
    q_k = q_values[:points_kratky]
    q_p = q_values[:points_porod]

    # MATLAB matrix is (n_curves × n_points), i.e. transposed relative to
    # intensities (n_points × n_curves).
    I_T = intensities.T   # shape (n_curves, n_points)

    mat_abs = I_T[:, :points_abs].copy()
    mat_holtzer = I_T[:, :points_holtzer] * q_h[np.newaxis, :]
    mat_kratky = I_T[:, :points_kratky] * q_k[np.newaxis, :] ** 2
    mat_porod = I_T[:, :points_porod] * q_p[np.newaxis, :] ** 4

    return mat_abs, mat_holtzer, mat_kratky, mat_porod, q_a, q_h, q_k, q_p
