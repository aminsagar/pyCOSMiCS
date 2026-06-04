"""
HTML report generator.
Translated from publishReport_two.m.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive; must be set before importing pyplot
import matplotlib.pyplot as plt

from .plots import (
    plot_semilog,
    plot_all_representations,
    plot_pca,
    plot_als_result,
)


_CSS_FALLBACK = """
body { font-family: 'Roboto', sans-serif; margin: 2em; background: #fafafa; }
h1 { color: #2c3e50; }
h3 { color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h4 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
th { background-color: #2c7873; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
img.small  { max-width: 48%; margin: 1%; vertical-align: top; }
img.grande { max-width: 90%; margin: 1em auto; display: block; }
footer { margin-top: 3em; color: #888; font-size: 0.9em; border-top: 1px solid #ccc; padding-top: 1em; }
"""

_SOLUTION_LABELS = {1: "Convergence", 2: "Divergence", 3: "Iter. exceeded"}


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    output_folder: str | Path,
    experiment_name: str,
    out_file_names: List[str],
    q_values: np.ndarray,
    intensities: np.ndarray,
    mat_holtzer: np.ndarray,
    mat_kratky: np.ndarray,
    mat_porod: np.ndarray,
    points_abs: int,
    points_holtzer: int,
    points_kratky: int,
    points_porod: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    matrices_used: List[List[bool]],
    combination_names: List[str],
    statistics_workflow: List[dict],
    chi_average_workflow: np.ndarray,
    species_workflow: List[np.ndarray],
    copt_elim_workflow: List[np.ndarray],
    chi_square_all: np.ndarray,
    curvas_estim_inic_wf: List[List[int]],
    curvas_eliminadas_wf: List[List[int]],
    elim_points: int,
    units: str,
    corte_abs: float,
    corte_holtzer: float,
    corte_kratky: float,
    corte_porod: float,
    n_species: int,
    tipo_als: int,
    tol_sigma: float,
    num_iterations: int,
    style_css_src: Optional[str] = None,
) -> Path:
    """
    Write SVG figures and an HTML report to <output_folder>/Report/.

    Returns the path to report.html.
    """
    report_dir = Path(output_folder) / "Report"
    report_dir.mkdir(parents=True, exist_ok=True)

    n_curves = intensities.shape[1]
    colors = plt.cm.jet(np.linspace(0, 1, n_curves))
    units_str = "1/Å" if units == "A" else "1/nm"

    # ------------------------------------------------------------------ #
    # Dataset figures                                                      #
    # ------------------------------------------------------------------ #
    fig = plot_semilog(q_values[:points_abs], intensities[:points_abs], show=False)
    _savefig(fig, report_dir / "datasetSemilog.svg")

    for tag, pts, mat in [
        ("datasetHoltzer", points_holtzer, mat_holtzer),
        ("datasetKratky",  points_kratky,  mat_kratky),
        ("datasetPorod",   points_porod,   mat_porod),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(n_curves):
            ax.plot(q_values[:pts], mat[i, :pts], color=colors[i])
        ax.set_xlabel("q"); ax.set_ylabel("I(q)·q^n")
        ax.set_title(tag.replace("dataset", ""))
        _savefig(fig, report_dir / f"{tag}.svg")

    # ------------------------------------------------------------------ #
    # PCA figure                                                           #
    # ------------------------------------------------------------------ #
    fig_pca = plot_pca(eigenvalues, eigenvectors, show=False)
    _savefig(fig_pca, report_dir / "eigenvalues.svg")

    # ------------------------------------------------------------------ #
    # Per-test solution figures                                            #
    # ------------------------------------------------------------------ #
    for il in range(len(matrices_used)):
        stats = statistics_workflow[il]
        sp = species_workflow[il]
        fig_sol = plot_als_result(
            q_values[:points_abs],
            sp[:points_abs] if sp.ndim == 2 else sp[:points_abs, :],
            copt_elim_workflow[il],
            chi_average_workflow[il],
            f"Test {il+1}: {combination_names[il]}",
            stats["solution"],
            chi_square_all[:, il] if chi_square_all is not None else None,
            show=False,
        )
        _savefig(fig_sol, report_dir / f"solution{il+1}.svg")

        fig_chi, ax = plt.subplots(figsize=(8, 4))
        ax.plot(chi_square_all[:, il], ":*", markersize=5)
        ax.set_title(f"χ² per curve — Test {il+1}")
        ax.set_xlabel("Curve index"); ax.set_ylabel("χ²")
        _savefig(fig_chi, report_dir / f"chiTest{il+1}.svg")

    # ------------------------------------------------------------------ #
    # CSS                                                                  #
    # ------------------------------------------------------------------ #
    css_path = report_dir / "style.css"
    if style_css_src and os.path.exists(style_css_src):
        shutil.copy(style_css_src, css_path)
    else:
        css_path.write_text(_CSS_FALLBACK)

    # ------------------------------------------------------------------ #
    # HTML                                                                 #
    # ------------------------------------------------------------------ #
    html_path = report_dir / "report.html"
    lines: List[str] = []
    W = lines.append

    W("<!DOCTYPE html>")
    W("<html><head>")
    W('<meta charset="UTF-8">')
    W('<link rel="stylesheet" href="style.css">')
    W("</head><body>")
    W(f"<h1>{experiment_name}</h1>")

    # -- Data info --------------------------------------------------------
    W('<h3 id="dataInfo">Data information</h3>')
    W(f"<p><b>Output folder:</b> {output_folder}</p>")
    W(f"<p><b>Total curves:</b> {n_curves}</p>")
    W("<p><b>List of files:</b></p>")
    for i, name in enumerate(out_file_names):
        W(f"Curve {i+1} — {name}<br>")
    W('<img class="small" src="datasetSemilog.svg" alt="Semilog">')
    W('<img class="small" src="datasetHoltzer.svg" alt="Holtzer">')
    W('<img class="small" src="datasetKratky.svg" alt="Kratky">')
    W('<img class="small" src="datasetPorod.svg" alt="Porod">')

    # -- Analysis options -------------------------------------------------
    W('<h3 id="analysisOptions">Analysis options</h3>')
    W(f"<p><b>Points removed at beginning:</b> {elim_points}</p>")
    W(f"<p><b>Momentum transfer range ({units_str}):</b></p>")
    W("<ul>")
    W(f"  <li>Absolute: {corte_abs:.2f}</li>")
    W(f"  <li>Holtzer:  {corte_holtzer:.2f}</li>")
    W(f"  <li>Kratky:   {corte_kratky:.2f}</li>")
    W(f"  <li>Porod:    {corte_porod:.2f}</li>")
    W("</ul>")
    W(f"<p><b>Number of species:</b> {n_species}</p>")

    tipo_labels = {
        1: "Fibrillation / folding — non-negativity + closure",
        2: "Titration — non-negativity + selectivity + closure",
        3: "SEC-SAXS — non-negativity only",
        4: "User-defined",
    }
    W(f"<p><b>Experiment type:</b> {tipo_labels.get(tipo_als, str(tipo_als))}</p>")
    W(f"<p><b>Convergence criterion:</b> {tol_sigma:.3f}%</p>")
    W(f"<p><b>Max iterations:</b> {num_iterations}</p>")

    # -- PCA --------------------------------------------------------------
    W('<h3 id="PCA">Principal Component Analysis</h3>')
    W('<img class="small" src="eigenvalues.svg" alt="PCA">')

    # -- Results table ----------------------------------------------------
    W('<h3 id="Results">Analysis results</h3>')
    W("<table>")
    W("<tr>")
    for hdr in ["Test", "Combination", "Rem. curves", "Init. estim.",
                "lof (PCA)", "lof (exp)", "R²", "χ²", "Convergence"]:
        W(f"  <th>{hdr}</th>")
    W("</tr>")

    for il in range(len(matrices_used)):
        stats = statistics_workflow[il]
        sol   = _SOLUTION_LABELS.get(stats["solution"], "?")
        elim  = ", ".join(str(e + 1) for e in curvas_eliminadas_wf[il]) if curvas_eliminadas_wf[il] else "—"
        estim = ", ".join(str(e + 1) for e in curvas_estim_inic_wf[il])  if curvas_estim_inic_wf[il]  else "—"
        W("<tr>")
        W(f"  <td><b>{il+1}</b></td>")
        W(f"  <td>{combination_names[il]}</td>")
        W(f"  <td>{elim}</td>")
        W(f"  <td>{estim}</td>")
        W(f"  <td>{stats['lof_pca']:.4f}</td>")
        W(f"  <td>{stats['lof_exp']:.4f}</td>")
        W(f"  <td>{stats['r2']*100:.4f}%</td>")
        W(f"  <td>{chi_average_workflow[il]:.2f}</td>")
        W(f"  <td>{sol}</td>")
        W("</tr>")
    W("</table>")

    # -- Per-test details -------------------------------------------------
    for il in range(len(matrices_used)):
        W(f"<h4>Test {il+1} — {combination_names[il]}</h4>")
        W(f'<img class="grande" src="solution{il+1}.svg" alt="Solution {il+1}">')
        W(f'<img class="grande" src="chiTest{il+1}.svg" alt="Chi {il+1}">')

    W("<footer>")
    W("<p>COSMiCS — Chemometric decomposition Of SAXS data</p>")
    W("<p>Authors: Fatima Herranz-Trillo, Amin Sagar | Pau Bernado Group, CBS CNRS</p>")
    W("</footer>")
    W("</body></html>")

    html_path.write_text("\n".join(lines))
    return html_path
