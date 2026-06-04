"""
Matplotlib-based plotting utilities.
Translated from plots.m, publishReport_two.m, and the inline figure calls
scattered throughout COSMiCS_multi.m / checkTest_scaled.m.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _color_cycle(n: int):
    return cm.jet(np.linspace(0, 1, max(n, 1)))


# ---------------------------------------------------------------------------
# Raw data views
# ---------------------------------------------------------------------------

def plot_semilog(
    q_values: np.ndarray,
    intensities: np.ndarray,
    title: str = "Data set (semilogarithmic scale)",
    show: bool = True,
) -> plt.Figure:
    """Semilogarithmic I(q) plot — seleccion=2 in MATLAB."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = _color_cycle(intensities.shape[1])
    for i in range(intensities.shape[1]):
        ax.semilogy(q_values, intensities[:, i], color=colors[i], linewidth=1.5)
    ax.set_xlabel("q", fontsize=13)
    ax.set_ylabel("I(q)", fontsize=13)
    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_all_representations(
    q_values: np.ndarray,
    intensities: np.ndarray,
    points_abs: int,
    points_holtzer: int,
    points_kratky: int,
    points_porod: int,
    show: bool = True,
) -> plt.Figure:
    """Four-panel plot of Absolute, Holtzer, Kratky, Porod — seleccion=1 in MATLAB."""
    n = intensities.shape[1]
    colors = _color_cycle(n)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    qa = q_values[:points_abs]
    for i in range(n):
        axes[0, 0].plot(qa, intensities[:points_abs, i], color=colors[i], linewidth=1.5)
    axes[0, 0].set_title("Absolute scale")
    axes[0, 0].set_xlabel("q"); axes[0, 0].set_ylabel("I(q)")

    qh = q_values[:points_holtzer]
    for i in range(n):
        axes[0, 1].plot(qh, intensities[:points_holtzer, i] * qh, color=colors[i], linewidth=1.5)
    axes[0, 1].set_title("Holtzer plot")
    axes[0, 1].set_xlabel("q"); axes[0, 1].set_ylabel("I(q)·q")

    qk = q_values[:points_kratky]
    for i in range(n):
        axes[1, 0].plot(qk, intensities[:points_kratky, i] * qk ** 2, color=colors[i], linewidth=1.5)
    axes[1, 0].set_title("Kratky plot")
    axes[1, 0].set_xlabel("q"); axes[1, 0].set_ylabel("I(q)·q²")

    qp = q_values[:points_porod]
    for i in range(n):
        axes[1, 1].plot(qp, intensities[:points_porod, i] * qp ** 4, color=colors[i], linewidth=1.5)
    axes[1, 1].set_title("Porod plot")
    axes[1, 1].set_xlabel("q"); axes[1, 1].set_ylabel("I(q)·q⁴")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def plot_pca(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_show: int = 5,
    show: bool = True,
) -> plt.Figure:
    """Bar chart of eigenvalues + first eigenvectors."""
    n_ev = min(len(eigenvalues), 10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(range(1, n_ev + 1), eigenvalues[:n_ev])
    for i, v in enumerate(eigenvalues[:n_ev]):
        axes[0].text(i + 1, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Eigenvalues")
    axes[0].set_xlabel("Component")

    n_show = min(n_show, eigenvectors.shape[1])
    for i in range(n_show):
        axes[1].plot(eigenvectors[:, i], label=f"EV {i+1}")
    axes[1].set_title("Eigenvectors")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# ALS workflow results
# ---------------------------------------------------------------------------

def plot_als_result(
    q_values: np.ndarray,
    species: np.ndarray,
    conc: np.ndarray,
    chi2: float,
    test_name: str,
    solution: int,
    chi_per_curve: Optional[np.ndarray] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Species spectra (semilog) + concentration profiles + optional per-curve χ².
    Mirrors the subplot block inside the workflow loop of COSMiCS_multi.m.
    """
    n_panels = 3 if chi_per_curve is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels))

    axes[0].semilogy(q_values, species)
    axes[0].set_title(f"{test_name}  χ²={chi2:.2f}")
    axes[0].set_xlabel("q"); axes[0].set_ylabel("I(q)")
    if solution != 1:
        axes[0].text(0.78, 0.85, "DIVERGENCE!", transform=axes[0].transAxes,
                     color="red", fontsize=11, fontweight="bold")

    axes[1].plot(conc)
    axes[1].set_title("Concentrations")
    axes[1].set_xlabel("Curve index"); axes[1].set_ylabel("Concentration")

    if chi_per_curve is not None:
        axes[2].bar(range(len(chi_per_curve)), chi_per_curve)
        axes[2].set_title("χ² per curve")
        axes[2].set_xlabel("Curve index"); axes[2].set_ylabel("χ²")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_chi_selection(
    avg_rank: np.ndarray,
    best_idx: int,
    pure_components: list,
    intensities: np.ndarray,
    show: bool = True,
) -> plt.Figure:
    """Scatter plot of chi-rank selection + selected curves — ChiSelection.m."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    x = np.arange(len(avg_rank))
    axes[0].scatter(x, avg_rank, c=avg_rank, cmap="jet", s=avg_rank + 1)
    axes[0].scatter(best_idx, avg_rank[best_idx], color="black", linewidths=2, zorder=5)
    axes[0].set_xlabel("Curve number", fontsize=14)
    axes[0].set_ylabel("Average rank", fontsize=14)
    axes[0].set_title("Average chi rank", fontsize=15)

    for idx in pure_components:
        axes[1].semilogy(intensities[:, idx], linewidth=2, label=f"Curve {idx+1}")
    axes[1].set_xlabel("q index", fontsize=14)
    axes[1].set_ylabel("Intensity", fontsize=14)
    axes[1].set_title("Selected estimates", fontsize=15)
    axes[1].legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig
