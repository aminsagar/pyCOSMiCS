"""
Methods for selecting initial estimates (pure-component curves).

Method 1 — dissimilarity_based_selection():
    Greedy farthest-point selection in normalised intensity space.
    Replaces the MATLAB 'pure' (SIMPLISMA) function from MCR-ALS which is
    not included in the COSMiCS repository.

Method 2 — chi_based_selection():
    Exact port of ChiSelection.m / crearMat_manualpure_rank.m (Method 2 path).
    Given two user-supplied curves, finds the third as the one with the
    highest average rank in chi-square distance from both.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .metrics import compare2curves


# ---------------------------------------------------------------------------
# Method 1 — greedy dissimilarity
# ---------------------------------------------------------------------------

def dissimilarity_based_selection(
    intensities: np.ndarray,
    n_species: int,
) -> List[int]:
    """
    Select *n_species* curves that are maximally dissimilar from each other.

    Algorithm:
      1. Normalise each curve to unit L2 norm.
      2. Seed with the curve that has the highest raw norm (largest signal).
      3. Iteratively add the curve farthest from the current selected set.

    Returns 0-based curve indices, sorted.
    """
    n_curves = intensities.shape[1]
    norms = np.linalg.norm(intensities, axis=0)
    norms_safe = np.where(norms == 0, 1.0, norms)
    normed = intensities / norms_safe[np.newaxis, :]

    selected = [int(np.argmax(norms))]

    for _ in range(n_species - 1):
        # Minimum distance of each curve to the selected set
        min_dists = np.full(n_curves, np.inf)
        for s in selected:
            d = np.linalg.norm(normed - normed[:, s : s + 1], axis=0)
            min_dists = np.minimum(min_dists, d)

        for s in selected:
            min_dists[s] = -1.0  # exclude already selected

        selected.append(int(np.argmax(min_dists)))

    return sorted(selected)


# ---------------------------------------------------------------------------
# Method 2 — chi-rank selection  (ChiSelection.m)
# ---------------------------------------------------------------------------

def chi_based_selection(
    intensities: np.ndarray,
    errors: np.ndarray,
    q_values: np.ndarray,
    two_estimates: List[int],
) -> List[int]:
    """
    Given two seed curves *two_estimates* (0-based indices), find a third
    curve that is jointly most dissimilar from both (by chi-square rank).

    Returns a list of three 0-based curve indices.
    """
    n_curves = intensities.shape[1]
    e1, e2 = two_estimates[0], two_estimates[1]

    def _chi_vs(ref_idx: int) -> np.ndarray:
        ref = np.column_stack([
            q_values,
            intensities[:, ref_idx],
            errors[:, ref_idx],
        ])
        chi = np.zeros(n_curves)
        for i in range(n_curves):
            syn = np.column_stack([q_values, intensities[:, i]])
            chi[i], _ = compare2curves(ref, syn)
        return chi

    chi1 = _chi_vs(e1)
    chi2 = _chi_vs(e2)

    rank1 = np.argsort(np.argsort(chi1))
    rank2 = np.argsort(np.argsort(chi2))

    # Build average-rank list aligned by curve number (matches MATLAB)
    avg_rank = np.zeros(n_curves)
    for i in range(n_curves):
        avg_rank[i] = (rank1[i] + rank2[i]) / 2.0

    third = int(np.argmax(avg_rank))
    return [e1, e2, third]
