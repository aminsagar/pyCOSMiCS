"""
Chi-square and curve-comparison utilities.
Translated from compare2curves.m, reconstCurvas.m, and the checkTest_scaled.m
chi-computation block.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

EPS = 1e-50


def compare2curves(
    curve_exp: np.ndarray,
    curve_synth: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute the reduced chi-square between an experimental and a synthetic curve.

    Parameters
    ----------
    curve_exp   : ndarray, columns [q, I, E]  (error in column 2)
    curve_synth : ndarray, any number of columns — the *last* column is I

    Returns
    -------
    chi : reduced chi-square (float)
    fit : scaled synthetic intensity, shape (n_points,)
    """
    n = len(curve_exp)
    weights = 1.0 / np.maximum(curve_exp[:, 2] ** 2, EPS)

    I_exp = curve_exp[:, 1]
    I_syn = curve_synth[:, -1]

    sum1 = np.sum(I_exp * I_syn * weights)
    sum2 = np.sum(I_syn ** 2 * weights)
    ratio = sum1 / max(sum2, EPS)

    fit = I_syn * ratio
    chi = np.sum((I_exp - fit) ** 2 * weights) / max(n - 1, 1)

    return chi, fit


def reconstruct_curves(
    conc: np.ndarray,
    spectra: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct data from concentration profiles and species spectra.

    Parameters
    ----------
    conc    : ndarray, shape (n_curves, n_species)
    spectra : ndarray, shape (n_species, n_points)

    Returns
    -------
    ndarray, shape (n_points, n_curves)
    """
    return (conc @ spectra).T


def compute_chi_all(
    intensities: np.ndarray,
    errors: np.ndarray,
    q_values: np.ndarray,
    reconstruction: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reduced chi-square for every curve.

    Parameters
    ----------
    intensities   : (n_points_full, n_curves)
    errors        : (n_points_full, n_curves)
    q_values      : (n_points_full,)
    reconstruction: (n_points_full, n_curves)   — output of reconstruct_curves()
    n_points      : number of points to include (pointsAbs)

    Returns
    -------
    chi_all : (n_curves,)
    fits    : (n_points, n_curves)
    """
    n_curves = intensities.shape[1]
    chi_all = np.zeros(n_curves)
    fits = np.zeros((n_points, n_curves))

    for i in range(n_curves):
        exp = np.column_stack([
            q_values[:n_points],
            intensities[:n_points, i],
            errors[:n_points, i],
        ])
        syn = np.column_stack([
            q_values[:n_points],
            reconstruction[:n_points, i],
        ])
        chi, fit = compare2curves(exp, syn)
        chi_all[i] = chi
        fits[:, i] = fit

    return chi_all, fits
