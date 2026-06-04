"""
File I/O for SAXS curves.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_curves(
    folder: str | Path,
    pattern: str,
    header_lines: int = 3,
    n_columns: int = 3,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all SAXS curve files matching *pattern* inside *folder*.

    Each file is expected to have columns [q, I(q), ...] with *header_lines*
    header rows and exactly *n_columns* data columns.

    Returns
    -------
    curves : list of ndarray, each shape (n_points, n_columns)
    names  : list of str — file basenames in sorted order
    """
    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {folder}"
        )

    curves, names = [], []
    for p in paths:
        data = np.loadtxt(p, skiprows=header_lines, usecols=range(n_columns))
        curves.append(data)
        names.append(p.name)

    return curves, names


def align_curves(curves: List[np.ndarray]) -> List[np.ndarray]:
    """
    Trim all curves to a common length and q-range.

    Strategy (mirrors COSMiCS_multi.m):
    1. If all curves already have the same length, return as-is.
    2. If q-vectors agree on the minimum length, trim to that length.
    3. Otherwise find the largest first q-value, align start points,
       then trim to the resulting minimum length.
    """
    lengths = [c.shape[0] for c in curves]
    if len(set(lengths)) == 1:
        return curves  # already aligned

    min_len = min(lengths)
    q_ref = curves[0][:min_len, 0]

    if all(np.allclose(c[:min_len, 0], q_ref, rtol=1e-6, atol=0) for c in curves):
        return [c[:min_len] for c in curves]

    # Find common starting q
    q_start = max(c[0, 0] for c in curves)
    aligned = []
    for c in curves:
        idx = int(np.searchsorted(c[:, 0], q_start))
        aligned.append(c[idx:])

    min_len2 = min(a.shape[0] for a in aligned)
    return [a[:min_len2] for a in aligned]


def curves_to_matrices(
    curves: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack curves into intensity and error matrices.

    Returns
    -------
    q_values    : ndarray, shape (n_points,)
    intensities : ndarray, shape (n_points, n_curves)
    errors      : ndarray, shape (n_points, n_curves)  — zeros if no 3rd column
    """
    n_col = curves[0].shape[1]
    q_values = curves[0][:, 0]
    intensities = np.column_stack([c[:, 1] for c in curves])
    if n_col >= 3:
        errors = np.column_stack([c[:, 2] for c in curves])
    else:
        errors = np.zeros_like(intensities)
    return q_values, intensities, errors
