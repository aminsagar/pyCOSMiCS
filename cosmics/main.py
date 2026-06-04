"""
COSMiCS — Chemometric decomposition Of SAXS data
Python port of COSMiCS_multi.m

Usage
-----
python -m cosmics.main  \\
    --input /path/to/exp1 /path/to/exp2  \\
    --output /path/to/results            \\
    --pattern "curve*.dat"               \\
    [options]

Run  python -m cosmics.main --help  for all options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .als import als_closure2
from .initial_estimates import chi_based_selection, dissimilarity_based_selection
from .io import align_curves, curves_to_matrices, load_curves
from .matrix_builder import (
    COMBINATION_NAMES,
    COMBINATIONS,
    build_initial_estimates,
    build_input_matrix,
    zero_eliminated_curves,
)
from .metrics import compute_chi_all, reconstruct_curves
from .pca import pcarep
from .plots import (
    plot_all_representations,
    plot_als_result,
    plot_chi_selection,
    plot_pca,
    plot_semilog,
)
from .preprocessing import (
    apply_transforms,
    default_cuts,
    find_cut_index,
    normalize_by_i0,
    scale_at_reference,
)
from .report import generate_report


# =========================================================================
# Helpers
# =========================================================================

def _prompt(msg: str, default=None, cast=str):
    """Interactive prompt with an optional default value."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{msg}{suffix}: ").strip()
    if raw == "" and default is not None:
        return default
    return cast(raw)


def _prompt_yn(msg: str, default: str = "y") -> bool:
    ans = _prompt(msg, default=default).lower()
    return ans.startswith("y")


def _prompt_float(msg: str, default: float) -> float:
    return _prompt(msg, default=default, cast=float)


def _prompt_int(msg: str, default: int) -> int:
    return _prompt(msg, default=default, cast=int)


# =========================================================================
# Core workflow
# =========================================================================

def run(args: argparse.Namespace) -> None:   # noqa: C901 (long but mirrors MATLAB)
    """
    Full COSMiCS workflow — direct translation of COSMiCS_multi.m.

    All interactive MATLAB prompts are replaced by CLI arguments; interactive
    mode (--interactive) re-enables the questions for exploratory sessions.
    """
    interactive = args.interactive
    show_plots  = not args.no_plots

    print()
    print("*" * 100)
    print("*  COSMiCS — SAXS DATA DECOMPOSITION BY CHEMOMETRIC ANALYSIS  *")
    print("*" * 100)
    print()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    input_dirs: List[str] = args.input
    num_exp_v = len(input_dirs)

    pattern       = args.pattern
    header_lines  = args.header
    n_columns     = args.columns

    all_curves:   List[np.ndarray] = []
    all_names:    List[str]        = []
    experiment_ranges = [0]

    for k, folder in enumerate(input_dirs):
        print(f"\nLoading experiment {k+1}: {folder}")
        curves, names = load_curves(folder, pattern, header_lines, n_columns)
        all_curves.extend(curves)
        all_names.extend(names)
        experiment_ranges.append(experiment_ranges[-1] + len(curves))
        print(f"  {len(curves)} files matched '{pattern}'")

    experiment_ranges = np.array(experiment_ranges, dtype=int)

    # Align and stack
    aligned = align_curves(all_curves)
    q_values, intensities, errors = curves_to_matrices(aligned)

    n_total = intensities.shape[1]
    print(f"\nTotal curves loaded: {n_total}")

    # ------------------------------------------------------------------
    # 2. NORMALISE
    # ------------------------------------------------------------------
    intensities, errors = normalize_by_i0(intensities, errors)
    intensities, errors, scale_factors = scale_at_reference(intensities, errors)

    np.savetxt(output_folder / "scale.txt", scale_factors)

    # ------------------------------------------------------------------
    # 3. SHOW RAW DATA
    # ------------------------------------------------------------------
    if show_plots:
        plot_semilog(q_values, intensities, show=True)

    # ------------------------------------------------------------------
    # 4. UNITS
    # ------------------------------------------------------------------
    units = args.units.upper()
    if interactive:
        units = _prompt("Units (A=1/Angstrom, N=1/nm)", default=units).upper()

    # ------------------------------------------------------------------
    # 5. REMOVE LEADING POINTS
    # ------------------------------------------------------------------
    elim_points = args.elim_points
    if interactive and _prompt_yn("Remove data points from beginning?", "y"):
        while True:
            elim_points = _prompt_int("Number of points to remove", 0)
            if elim_points > 0:
                intensities = intensities[elim_points:, :]
                errors      = errors[elim_points:, :]
                q_values    = q_values[elim_points:]
            if show_plots:
                plot_semilog(q_values, intensities, show=True)
            if _prompt_yn("Keep this?", "y"):
                break
            # restore
            aligned2 = align_curves(all_curves)
            q_values, intensities, errors = curves_to_matrices(aligned2)
            intensities, errors = normalize_by_i0(intensities, errors)
            intensities, errors, scale_factors = scale_at_reference(intensities, errors)
            elim_points = 0
    elif elim_points > 0:
        intensities = intensities[elim_points:, :]
        errors      = errors[elim_points:, :]
        q_values    = q_values[elim_points:]

    # ------------------------------------------------------------------
    # 6. Q-RANGE CUTS
    # ------------------------------------------------------------------
    def_abs, def_h, def_k, def_p = default_cuts(units)
    corte_abs     = args.cut_abs     if args.cut_abs     is not None else def_abs
    corte_holtzer = args.cut_holtzer if args.cut_holtzer is not None else def_h
    corte_kratky  = args.cut_kratky  if args.cut_kratky  is not None else def_k
    corte_porod   = args.cut_porod   if args.cut_porod   is not None else def_p

    if interactive:
        print(f"\nDefault cuts ({units_str(units)}): Abs={def_abs}  H={def_h}  K={def_k}  P={def_p}")
        if _prompt_yn("Use default cuts?", "y"):
            corte_abs, corte_holtzer, corte_kratky, corte_porod = def_abs, def_h, def_k, def_p
        else:
            corte_abs     = _prompt_float("Cut for Absolute scale", def_abs)
            corte_holtzer = _prompt_float("Cut for Holtzer scale",  def_h)
            corte_kratky  = _prompt_float("Cut for Kratky scale",   def_k)
            corte_porod   = _prompt_float("Cut for Porod scale",    def_p)

    points_abs     = find_cut_index(q_values, corte_abs)
    points_holtzer = find_cut_index(q_values, corte_holtzer)
    points_kratky  = find_cut_index(q_values, corte_kratky)
    points_porod   = find_cut_index(q_values, corte_porod)

    print(f"\nQ-cuts ({units_str(units)}):  Abs={corte_abs}({points_abs}pts)  "
          f"H={corte_holtzer}({points_holtzer}pts)  "
          f"K={corte_kratky}({points_kratky}pts)  "
          f"P={corte_porod}({points_porod}pts)")

    # ------------------------------------------------------------------
    # 7. REPRESENTATION MATRICES
    # ------------------------------------------------------------------
    mat_abs, mat_holtzer, mat_kratky, mat_porod, \
        q_abs, q_h, q_k, q_p = apply_transforms(
            intensities, q_values,
            points_abs, points_holtzer, points_kratky, points_porod,
        )

    # ------------------------------------------------------------------
    # 8. PCA
    # ------------------------------------------------------------------
    n_pca = min(n_total, 10)
    U_abs, S_abs, V_abs, _, _ = pcarep(intensities, n_total)
    eigenvalues = np.diag(S_abs)[:n_pca]
    eigenvectors = U_abs[:, :n_pca]

    ev_abs_full = np.diag(S_abs)   # full eigenvalue vector for scaling

    np.savetxt(output_folder / "eigenvalues.txt",  eigenvalues.reshape(-1, 1))
    np.savetxt(output_folder / "eigenvectors.txt", eigenvectors)

    if show_plots:
        plot_pca(eigenvalues, eigenvectors, show=True)

    # ------------------------------------------------------------------
    # 9. NUMBER OF SPECIES
    # ------------------------------------------------------------------
    n_species = args.n_species
    if interactive:
        n_species = _prompt_int("Number of species", n_species)

    # ------------------------------------------------------------------
    # 10. INITIAL ESTIMATES
    # ------------------------------------------------------------------
    init_method = args.init_method
    if interactive:
        print("\nInitial estimate method:")
        print("  1 — dissimilarity-based (replaces MATLAB pure/SIMPLISMA)")
        print("  2 — chi-rank based (ChiSelection.m)")
        init_method = _prompt_int("Method", init_method)

    if init_method == 2:
        two_est = args.two_estimates
        if two_est is None:
            if interactive:
                raw = input("Enter two seed curve numbers (1-based, e.g. 1 51): ")
                two_est = [int(x) - 1 for x in raw.split()]
            else:
                raise ValueError("--two-estimates is required for --init-method 2")
        else:
            two_est = [e - 1 for e in two_est]   # convert to 0-based

        pure_components = chi_based_selection(intensities, errors, q_values, two_est)
        print(f"\nInitial estimates (chi-based): {[p+1 for p in pure_components]}")

        if show_plots:
            from .metrics import compare2curves
            n_c = n_total
            avg_rank = np.zeros(n_c)
            # Reconstruct avg_rank for plotting
            ref1 = np.column_stack([q_values, intensities[:, two_est[0]], errors[:, two_est[0]]])
            ref2 = np.column_stack([q_values, intensities[:, two_est[1]], errors[:, two_est[1]]])
            chi1 = np.array([compare2curves(ref1, np.column_stack([q_values, intensities[:, i]]))[0]
                             for i in range(n_c)])
            chi2 = np.array([compare2curves(ref2, np.column_stack([q_values, intensities[:, i]]))[0]
                             for i in range(n_c)])
            r1 = np.argsort(np.argsort(chi1))
            r2 = np.argsort(np.argsort(chi2))
            avg_rank = (r1 + r2) / 2.0
            plot_chi_selection(avg_rank, pure_components[2], pure_components, intensities, show=True)

    else:
        pure_components = dissimilarity_based_selection(intensities, n_species)
        print(f"\nInitial estimates (dissimilarity): {[p+1 for p in pure_components]}")

    if interactive and _prompt_yn("Sort initial estimates by index?", "y"):
        pure_components = sorted(pure_components)

    # ------------------------------------------------------------------
    # 11. CONSTRAINT TYPE
    # ------------------------------------------------------------------
    tipo_als = args.experiment_type
    if interactive:
        print("\n1 — Fibrillation/folding (non-neg + closure)")
        print("2 — Titration (non-neg + selectivity + closure)")
        print("3 — SEC-SAXS (non-neg only)")
        print("4 — User-defined")
        tipo_als = _prompt_int("Experiment type", tipo_als)

    if tipo_als == 1:
        constraints = [1, 3]
        print("  Constraints: non-negativity + closure")
    elif tipo_als == 2:
        constraints = [1, 3, 4]
        print("  Constraints: non-negativity + selectivity + closure")
    elif tipo_als == 3:
        constraints = [1]
        print("  Constraints: non-negativity only")
    else:
        raw_c = getattr(args, "user_constraints", None) or [1, 3]
        if interactive:
            raw = input("Enter constraint codes (e.g. 1 3 4): ")
            raw_c = [int(x) for x in raw.split()]
        constraints = raw_c

    # ------------------------------------------------------------------
    # 12. CLOSURE INPUT FILES (for titration)
    # ------------------------------------------------------------------
    vclos1 = np.zeros((int(experiment_ranges[-1]), num_exp_v))
    if 3 in constraints and args.closure_pattern:
        conc_cons_all = []
        for k, folder in enumerate(input_dirs):
            cpaths = sorted(Path(folder).glob(args.closure_pattern))
            if not cpaths:
                print(f"  Warning: no closure files matching '{args.closure_pattern}' in {folder}")
                continue
            for cp in cpaths:
                conc_cons_all.append(np.loadtxt(cp))

        if conc_cons_all:
            conc_cons_all = np.vstack(conc_cons_all)
            for i in range(int(experiment_ranges[1])):
                vclos1[i, 0] = conc_cons_all[0, 0] / scale_factors[i]
            for i in range(int(experiment_ranges[1]), int(experiment_ranges[2])):
                vclos1[i, 1] = conc_cons_all[int(experiment_ranges[1]), 1] / scale_factors[i]

    # ------------------------------------------------------------------
    # 13. CONVERGENCE SETTINGS
    # ------------------------------------------------------------------
    tol_sigma      = args.tol
    num_iterations = args.max_iter
    if interactive:
        tol_sigma      = _prompt_float("Convergence criterion (%)", tol_sigma)
        num_iterations = _prompt_int("Maximum iterations", num_iterations)

    print(f"\nConvergence: {tol_sigma}%  Max iter: {num_iterations}")

    # ------------------------------------------------------------------
    # 14. ALS WORKFLOW
    # ------------------------------------------------------------------
    # The 8 combinations of representations (Absolute, Holtzer, Kratky, Porod)
    combinations_wf = [list(c) for c in COMBINATIONS]
    n_tests = len(combinations_wf)

    # Storage structures (equivalent to MATLAB global structs).
    # chi_average_list and chi_square_list grow as lists so they work
    # correctly when --auto-clean adds extra rounds beyond the initial 8.
    statistics_workflow:    List[dict]        = []
    copt_workflow:          List[np.ndarray]  = []
    species_workflow:       List[np.ndarray]  = []
    curvas_eliminadas_wf:   List[List[int]]   = []
    estim_iniciales_wf:     List[np.ndarray]  = []
    curvas_estim_inic_wf:   List[List[int]]   = []
    reconstruct_workflow:   List[np.ndarray]  = []
    fit_workflow:           List[np.ndarray]  = []
    copt_elim_workflow:     List[np.ndarray]  = []
    chi_average_list:       List[float]       = []   # converted to array at end
    chi_square_list:        List[np.ndarray]  = []   # each entry: (n_total,)
    names_wf:               List[str]         = []

    ev_abs_val = float(np.linalg.svd(mat_abs, compute_uv=False)[0])

    curvas_eliminadas: List[int] = []
    intensities_actual = intensities.copy()

    continue_loop = True
    round_start = 0

    while continue_loop:
        for test_idx, combination in enumerate(combinations_wf[round_start:], start=round_start):
            comb_name = COMBINATION_NAMES[test_idx % len(COMBINATION_NAMES)]
            print(f"\n{'='*70}")
            print(f"  Test {test_idx+1}: {comb_name}")
            print(f"{'='*70}")

            # Zero out eliminated curves in the working matrix
            mat_abs_work = zero_eliminated_curves(
                intensities_actual.T,   # shape (n_curves, n_pts_abs)
                curvas_eliminadas,
            )

            mat_input, puntos_matrices, sh, sk, sp = build_input_matrix(
                mat_abs_work, mat_holtzer, mat_kratky, mat_porod,
                combination, ev_abs_val,
            )

            estim_input = build_initial_estimates(
                mat_abs_work, mat_holtzer, mat_kratky, mat_porod,
                combination, pure_components,
                scale_h=sh, scale_k=sk, scale_p=sp,
            )

            # csel (concentration equality constraints)
            csel = np.full((n_total, n_species), np.nan)
            if 4 in constraints and args.selective_curves:
                sel_curves = [c - 1 for c in args.selective_curves]   # 0-based
                for c in sel_curves:
                    # row c of csel is set from vclos1
                    # (simplified: in MATLAB this reads from ConcConsAll)
                    pass   # user should supply --selective-curves properly

            # ssel (spectra equality constraints)
            ssel = np.full((n_species, mat_input.shape[1]), np.nan)

            n_exp_total = len(puntos_matrices) * num_exp_v
            isp = np.ones((num_exp_v, n_species))

            copt_als, sopt_als, sdopt, r2opt, itopt, solution = als_closure2(
                mat_input,       # d shape: (n_curves, n_cols) — rows are curves
                estim_input,     # x0 shape: (n_species, n_cols)
                n_exp_total,
                puntos_matrices,
                num_exp_v,
                experiment_ranges,
                n_iter=num_iterations,
                tol_sigma=tol_sigma,
                isp=isp,
                csel=csel,
                ssel=ssel,
                vclos1=vclos1,
                vclos2=0,
                constraints=constraints,
                verbose=True,
            )

            species_col = sopt_als.T    # shape (n_cols, n_species)

            # Reconstruct and compute chi-square
            reconstr = reconstruct_curves(copt_als, sopt_als)  # (n_pts, n_curves)
            chi_all, fits = compute_chi_all(
                intensities, errors, q_values, reconstr, points_abs
            )

            # Zero chi for eliminated curves
            for elim in curvas_eliminadas:
                chi_all[elim] = 0.0
                fits[:, elim] = 0.0

            n_valid = n_total - len(curvas_eliminadas)
            chi_avg = np.sum(chi_all) / max(n_valid, 1)

            # Build coptElimin (remove eliminated rows)
            keep = [i for i in range(n_total) if i not in curvas_eliminadas]
            copt_elim = copt_als[keep, :]

            # ---------- Store results ------------------------------------
            statistics_workflow.append({
                "lof_pca":  float(sdopt[0]),
                "lof_exp":  float(sdopt[1]),
                "r2":       float(r2opt),
                "iteration": int(itopt),
                "solution":  int(solution),
            })
            copt_workflow.append(copt_als)
            species_workflow.append(species_col)
            curvas_eliminadas_wf.append(list(curvas_eliminadas))
            estim_iniciales_wf.append(estim_input)
            curvas_estim_inic_wf.append(list(pure_components))
            reconstruct_workflow.append(reconstr)
            fit_workflow.append(fits)
            copt_elim_workflow.append(copt_elim)
            chi_average_list.append(chi_avg)
            chi_square_list.append(chi_all.copy())
            names_wf.append(comb_name)

            # ---------- Save per-test outputs ----------------------------
            _save_test_outputs(
                output_folder, test_idx, comb_name,
                q_abs, species_col[:points_abs, :],
                copt_elim, chi_all, chi_avg, solution, sdopt, r2opt, itopt,
                points_abs, n_species, units,
                corte_abs, corte_holtzer, corte_kratky, corte_porod,
                tipo_als, elim_points, pure_components,
                curvas_eliminadas, matrices_used=combination,
            )

            # ---------- Save reconstruction curves -----------------------
            _save_reconstruction(
                output_folder, test_idx, comb_name,
                q_values, intensities, errors, fits,
                points_abs, n_total,
            )

        # ------------------------------------------------------------------
        # Summary after one round of 8 combinations
        # ------------------------------------------------------------------
        n_done = len(statistics_workflow)
        valid_chi = [
            (chi_average_list[i], i)
            for i in range(n_done)
            if statistics_workflow[i]["solution"] == 1
        ]

        print("\n" + "=" * 70)
        print("  WORKFLOW SUMMARY")
        print("=" * 70)
        for il in range(n_done):
            sol = statistics_workflow[il]["solution"]
            sol_str = {1: "OK", 2: "DIVERGE", 3: "MAX_ITER"}.get(sol, "?")
            print(f"  Test {il+1:2d} ({names_wf[il]:8s}):  χ²={chi_average_list[il]:.2f}  [{sol_str}]")

        if valid_chi:
            best_chi, best_idx = min(valid_chi)
            print(f"\n  Best: Test {best_idx+1} ({names_wf[best_idx]})  χ²={best_chi:.2f}")
        else:
            best_idx = int(np.argmin(chi_average_list))
            best_chi = chi_average_list[best_idx]
            print(f"\n  Best (no convergence): Test {best_idx+1}  χ²={best_chi:.2f}")

        if show_plots:
            for il in range(n_done):
                plot_als_result(
                    q_values[:points_abs],
                    species_workflow[il][:points_abs],
                    copt_elim_workflow[il],
                    chi_average_list[il],
                    f"Test {il+1}: {names_wf[il]}",
                    statistics_workflow[il]["solution"],
                    chi_square_list[il],
                    show=True,
                )

        # Worst-chi curve elimination (disabled by default — matches MATLAB eliminar='n')
        print("\n  Skipping automatic curve elimination (set --auto-clean to enable).")
        continue_loop = False

        if args.auto_clean:
            worst_idx = int(np.argmax(chi_square_list[best_idx]))
            worst_chi = chi_square_list[best_idx][worst_idx]
            worst_top10 = np.sort(chi_square_list[best_idx])[::-1][:10]
            print(f"\n  10 worst χ²: {worst_top10}")
            print(f"  Curve {worst_idx+1} has worst χ²={worst_chi:.2f}")
            if worst_idx not in pure_components:
                curvas_eliminadas.append(worst_idx)
                intensities_actual[:, worst_idx] = 0.0
                round_start = len(statistics_workflow)
                combinations_wf.extend(COMBINATIONS)
                continue_loop = True
                print(f"  Removing curve {worst_idx+1} and repeating...")
            else:
                print(f"  Curve {worst_idx+1} is an initial estimate — not removing.")
                continue_loop = False

    # Convert list accumulators to arrays for reporting
    chi_average_workflow = np.array(chi_average_list)
    chi_square_all       = np.column_stack(chi_square_list)   # (n_total, n_tests_done)

    # ------------------------------------------------------------------
    # 15. SAVE INFO FILE
    # ------------------------------------------------------------------
    _save_info_file(
        output_folder, input_dirs, all_names, n_species, tipo_als,
        tol_sigma, num_iterations, curvas_eliminadas, units,
        corte_abs, corte_holtzer, corte_kratky, corte_porod,
    )

    # ------------------------------------------------------------------
    # 16. HTML REPORT
    # ------------------------------------------------------------------
    style_src = str(Path(__file__).parent.parent.parent / "style.css")
    html = generate_report(
        output_folder=str(output_folder),
        experiment_name=str(output_folder),
        out_file_names=all_names,
        q_values=q_values,
        intensities=intensities,
        mat_holtzer=mat_holtzer,
        mat_kratky=mat_kratky,
        mat_porod=mat_porod,
        points_abs=points_abs,
        points_holtzer=points_holtzer,
        points_kratky=points_kratky,
        points_porod=points_porod,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        matrices_used=combinations_wf[:len(statistics_workflow)],
        combination_names=names_wf,
        statistics_workflow=statistics_workflow,
        chi_average_workflow=chi_average_workflow[:len(statistics_workflow)],
        species_workflow=species_workflow,
        copt_elim_workflow=copt_elim_workflow,
        chi_square_all=chi_square_all[:, :len(statistics_workflow)],
        curvas_estim_inic_wf=curvas_estim_inic_wf,
        curvas_eliminadas_wf=curvas_eliminadas_wf,
        elim_points=elim_points,
        units=units,
        corte_abs=corte_abs,
        corte_holtzer=corte_holtzer,
        corte_kratky=corte_kratky,
        corte_porod=corte_porod,
        n_species=n_species,
        tipo_als=tipo_als,
        tol_sigma=tol_sigma,
        num_iterations=num_iterations,
        style_css_src=style_src if Path(style_src).exists() else None,
    )
    print(f"\nReport written to: {html}")
    print("\nDone.")


# =========================================================================
# Output helpers
# =========================================================================

def units_str(units: str) -> str:
    return "1/Å" if units == "A" else "1/nm"


def _test_folder(output_folder: Path, test_idx: int, comb_name: str) -> Path:
    prefix = f"Test{test_idx+1:02d}" if test_idx < 9 else f"Test{test_idx+1}"
    d = output_folder / f"{prefix}_{comb_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_test_outputs(
    output_folder, test_idx, comb_name,
    q_abs, species_abs, copt_elim, chi_all, chi_avg, solution,
    sdopt, r2opt, itopt, points_abs, n_species, units,
    corte_abs, corte_holtzer, corte_kratky, corte_porod,
    tipo_als, elim_points, pure_components,
    curvas_eliminadas, matrices_used,
):
    d = _test_folder(Path(output_folder), test_idx, comb_name)

    # Species files
    for sp in range(n_species):
        out = np.column_stack([q_abs, species_abs[:, sp]])
        np.savetxt(d / f"species{sp+1}.dat", out)

    # Concentration file
    np.savetxt(d / "concentration.txt", copt_elim)

    # Chi-square per curve
    chi_temp = np.column_stack([np.arange(1, len(chi_all) + 1), chi_all])
    np.savetxt(d / "chiSquareCurves.txt", chi_temp)

    # Per-test info
    sol_str = {1: "CONVERGENCE", 2: "DIVERGENCE", 3: "ITERATIONS EXCEEDED"}.get(solution, "?")
    units_label = units_str(units)
    mat_label = comb_name.replace("+", " + ").replace("A", "Absolute").replace(
        "H", "Holtzer").replace("K", "Kratky").replace("P", "Porod")

    info_lines = [
        f"Matrices used: {mat_label}",
        f"Initial estimations: {[p+1 for p in pure_components]}",
        f"Points removed at beginning: {elim_points}",
        f"Total curves: {copt_elim.shape[0]}",
        f"Cut Absolute: {corte_abs:.2f} {units_label}",
        f"Cut Holtzer:  {corte_holtzer:.2f} {units_label}",
        f"Cut Kratky:   {corte_kratky:.2f} {units_label}",
        f"Cut Porod:    {corte_porod:.2f} {units_label}",
        "",
        "RESULTS",
        "-" * 40,
        sol_str,
        f"Optimum at iteration {itopt}",
        f"Lack of fit (PCA): {sdopt[0]:.4e}",
        f"Lack of fit (exp): {sdopt[1]:.4e}",
        f"Variance explained: {r2opt*100:.2f}%",
        f"Chi-square reduced: {chi_avg:.2f}",
    ]
    if curvas_eliminadas:
        info_lines.append(f"Removed curves: {[e+1 for e in curvas_eliminadas]}")

    (d / f"infoTest{test_idx+1}.txt").write_text("\n".join(info_lines))


def _save_reconstruction(
    output_folder, test_idx, comb_name,
    q_values, intensities, errors, fits,
    points_abs, n_total,
):
    d = _test_folder(Path(output_folder), test_idx, comb_name) / "Reconstruction"
    d.mkdir(exist_ok=True)

    for j in range(n_total):
        tag = f"{j+1:02d}" if j + 1 <= 9 else str(j + 1)
        mcr = np.column_stack([q_values[:points_abs], fits[:points_abs, j]])
        exp = np.column_stack([
            q_values[:points_abs],
            intensities[:points_abs, j],
            errors[:points_abs, j] if errors is not None else intensities[:points_abs, j] * 0.02,
        ])
        np.savetxt(d / f"curveMCRALS_{tag}.dat", mcr)
        np.savetxt(d / f"curveExp_{tag}.dat",    exp)


def _save_info_file(
    output_folder, input_dirs, file_names, n_species, tipo_als,
    tol_sigma, num_iterations, curvas_eliminadas, units,
    corte_abs, corte_holtzer, corte_kratky, corte_porod,
):
    tipo_labels = {
        1: "Fibrillation or folding",
        2: "Titration",
        3: "SEC-SAXS",
        4: "User-defined",
    }
    ul = units_str(units)
    lines = [
        f"Input folders: {', '.join(input_dirs)}",
        "",
        "LIST OF FILES",
    ]
    for i, name in enumerate(file_names):
        lines.append(f"  Curve {i+1} - {name}")
    lines += [
        "",
        f"Number of species: {n_species}",
        "",
        "CONSTRAINTS",
        f"  Experiment type: {tipo_labels.get(tipo_als, str(tipo_als))}",
        f"  Non-negativity: concentrations and spectra (nnls)",
        f"  Closure: concentrations (equal, constant=1.0)",
        "",
        f"Cut Absolute: {corte_abs:.2f} {ul}",
        f"Cut Holtzer:  {corte_holtzer:.2f} {ul}",
        f"Cut Kratky:   {corte_kratky:.2f} {ul}",
        f"Cut Porod:    {corte_porod:.2f} {ul}",
        "",
        f"Convergence criterion: {tol_sigma:.3f}%",
        f"Maximum iterations:    {num_iterations}",
    ]
    if curvas_eliminadas:
        lines.append(f"Removed curves: {[e+1 for e in curvas_eliminadas]}")

    (Path(output_folder) / "info.txt").write_text("\n".join(lines))


# =========================================================================
# CLI entry point
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cosmics",
        description="COSMiCS — Chemometric decomposition Of SAXS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", nargs="+", required=True,
                   help="One or more directories containing SAXS curve files.")
    p.add_argument("--output", "-o", required=True,
                   help="Output directory for results and report.")
    p.add_argument("--pattern", "-p", default="curve*.dat",
                   help="Glob pattern for curve files.")
    p.add_argument("--header",  type=int, default=3,
                   help="Header lines to skip in each file.")
    p.add_argument("--columns", type=int, default=3,
                   help="Number of data columns (q, I[, E]).")
    p.add_argument("--units", choices=["A", "N"], default="A",
                   help="Q-axis units: A=1/Å, N=1/nm.")
    p.add_argument("--elim-points", type=int, default=0, metavar="N",
                   help="Number of leading q-points to discard.")
    p.add_argument("--n-species",  type=int, default=3,
                   help="Number of species (components) to decompose.")
    p.add_argument("--init-method", type=int, default=1, choices=[1, 2],
                   help="Initial-estimate method: 1=dissimilarity, 2=chi-rank.")
    p.add_argument("--two-estimates", type=int, nargs=2, metavar=("E1", "E2"),
                   help="Two 1-based curve indices for --init-method 2.")
    p.add_argument("--experiment-type", type=int, default=1, choices=[1, 2, 3, 4],
                   help="1=fibril/fold, 2=titration, 3=SEC-SAXS, 4=user.")
    p.add_argument("--tol",      type=float, default=0.01,
                   help="Convergence tolerance (percent change in sigma).")
    p.add_argument("--max-iter", type=int,   default=1000,
                   help="Maximum ALS iterations.")
    p.add_argument("--cut-abs",     type=float, default=None,
                   help="Q cutoff for Absolute representation.")
    p.add_argument("--cut-holtzer", type=float, default=None,
                   help="Q cutoff for Holtzer representation.")
    p.add_argument("--cut-kratky",  type=float, default=None,
                   help="Q cutoff for Kratky representation.")
    p.add_argument("--cut-porod",   type=float, default=None,
                   help="Q cutoff for Porod representation.")
    p.add_argument("--closure-pattern", default=None,
                   help="Glob pattern for closure input files (titration only).")
    p.add_argument("--selective-curves", type=int, nargs="+", default=None,
                   metavar="N", help="1-based curve indices to apply selectivity.")
    p.add_argument("--no-plots",    action="store_true",
                   help="Suppress all matplotlib windows.")
    p.add_argument("--interactive", action="store_true",
                   help="Re-enable interactive prompts (mirrors MATLAB behaviour).")
    p.add_argument("--auto-clean",  action="store_true",
                   help="Automatically remove the worst-chi curve and repeat.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
