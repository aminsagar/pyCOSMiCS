"""
MCR-ALS optimisation with closure constraint.
Translated from als_closure2.m (COSMiCS / MCR-ALS).

Key equivalences MATLAB → Python
---------------------------------
  d / abss         → d @ pinv(abss)          (solve conc @ abss = d for conc)
  conc \\ d        → lstsq(conc, d)           (solve conc @ abss = d for abss)
  fnnls(A'A, A'b)  → scipy.optimize.nnls(A, b)  (standard-form NNLS)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls

from .closure import closure_multi
from .pca import pcarep
from .unimod import unimod


def als_closure2(
    d: np.ndarray,
    x0: np.ndarray,
    n_exp: int,
    puntos_matrices: List[int],
    num_exp: int,
    experiment_ranges: np.ndarray,
    n_iter: int = 1000,
    tol_sigma: float = 0.01,
    plot_cb: Optional[Callable] = None,
    isp: Optional[np.ndarray] = None,
    csel: Optional[np.ndarray] = None,
    ssel=None,
    vclos1=0,
    vclos2=0,
    constraints: Optional[List[int]] = None,
    sto1: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """
    Alternating least-squares optimisation (MCR-ALS) with optional constraints.

    Parameters
    ----------
    d                 : (n_rows, n_cols) data matrix (PCA-reproduced internally)
    x0                : initial estimates — spectra (n_species, n_cols) or
                        concentrations (n_rows, n_species)
    n_exp             : total number of sub-experiments × representations
    puntos_matrices   : list of column counts per representation block
    num_exp           : number of experiments (samples)
    experiment_ranges : cumulative curve counts, shape (num_exp+1,)
    n_iter            : maximum iterations
    tol_sigma         : convergence threshold (% change in sigma)
    plot_cb           : optional callback(conc, abss, iteration, stage)
    isp               : (num_exp, n_species) species-presence matrix
    csel              : (n_rows, n_species) concentration equality constraints
    ssel              : (n_species, n_cols) spectra equality constraints or 0
    vclos1            : vectorial first closure or scalar 0
    vclos2            : vectorial second closure (q-values in original script)
    constraints       : list of active constraint codes
                         1=non-negativity, 2=unimodality, 3=closure,
                         4=csel, 5=ssel
    sto1              : stoichiometry per species (default all ones)
    verbose           : print iteration diagnostics

    Returns
    -------
    copt   : optimal concentrations, (n_rows, n_species)
    sopt   : optimal spectra,        (n_species, n_cols)
    sdopt  : [lof_pca, lof_exp] at optimum
    r2opt  : R² at optimum
    itopt  : iteration number of optimum
    solution : 1=converged, 2=diverged, 3=max iterations exceeded
    """
    if constraints is None:
        constraints = [1, 3]

    has_nneg    = 1 in constraints
    has_unimod  = 2 in constraints
    has_closure = 3 in constraints
    has_csel    = 4 in constraints
    has_ssel    = 5 in constraints

    n_row, n_col = d.shape

    # ------------------------------------------------------------------ #
    # Determine initial-estimate type and n_species                       #
    # ------------------------------------------------------------------ #
    r2, c2 = x0.shape
    if r2 == n_row:
        n_sign, ils = c2, 1          # x0 is concentrations
    elif c2 == n_row:
        x0 = x0.T; n_sign, ils = x0.shape[1], 1
    elif c2 == n_col:
        n_sign, ils = r2, 2          # x0 is spectra
    elif r2 == n_col:
        x0 = x0.T; n_sign, ils = x0.shape[0], 2
    else:
        n_sign, ils = r2, 2          # fallback: treat as spectra

    if isp is None:
        isp = np.ones((num_exp, n_sign))

    if csel is None:
        csel = np.full((n_row, n_sign), np.nan)

    if ssel is None or (np.isscalar(ssel) and ssel == 0):
        ssel = np.full((n_sign, n_col), np.nan)

    if sto1 is None:
        sto1 = np.ones(n_sign)

    # ------------------------------------------------------------------ #
    # Data-set structure (mirrors datamod logic)                          #
    # ------------------------------------------------------------------ #
    mat_c = num_exp                  # C sub-matrices (one per experiment)
    mat_r = len(puntos_matrices)     # S sub-matrices (one per representation)

    n_rinic = np.zeros(mat_c, dtype=int)
    n_rfin  = np.zeros(mat_c, dtype=int)
    n_cinic = np.zeros(mat_r, dtype=int)
    n_cfin  = np.zeros(mat_r, dtype=int)

    for i in range(mat_c):
        n_rows_i = int(experiment_ranges[i + 1] - experiment_ranges[i])
        n_rinic[i] = 0 if i == 0 else n_rfin[i - 1]
        n_rfin[i]  = n_rinic[i] + n_rows_i

    for i in range(mat_r):
        n_cinic[i] = 0 if i == 0 else n_cfin[i - 1]
        n_cfin[i]  = n_cinic[i] + puntos_matrices[i]

    # ------------------------------------------------------------------ #
    # Closure setup — one closure per C sub-matrix (iclos=1, iclos1=1)   #
    # Matches the hard-coded defaults in als_closure2.m:                  #
    #   sclos1(i,:) = [0,0,0]; sclos1(i,i) = 1; sclos1(i,3) = 1;        #
    # ------------------------------------------------------------------ #
    i_clos  = np.ones(mat_c, dtype=int)
    i_clos1 = np.ones(mat_c, dtype=int)   # equal condition
    # Default closure constant = 1.0 (sum-to-one, standard for fibrillation).
    # For titration, vclos1 carries the actual concentrations so this value
    # is never reached (the vclos1_n branch is taken instead).
    t_clos1 = np.ones(mat_c)
    t_clos2 = np.zeros(mat_c)
    sclos1  = np.zeros((mat_c, n_sign))
    sclos2  = np.zeros((mat_c, n_sign))

    for i in range(mat_c):
        if mat_c == 1:
            # Single experiment: all species participate in the one closure
            # (concentrations sum to 1 globally).
            sclos1[i, :] = 1
        else:
            # Multi-experiment: experiment i owns species i and the shared
            # last species — mirrors the MATLAB hard-coded pattern:
            #   sclos1(i,i)=1; sclos1(i,n_species)=1
            sclos1[i, i % n_sign] = 1
            sclos1[i, n_sign - 1] = 1

    # ------------------------------------------------------------------ #
    # Non-negativity bookkeeping                                          #
    # ------------------------------------------------------------------ #
    cneg   = np.ones((mat_c, n_sign))
    spneg  = np.ones((n_sign, mat_r))

    # Unimodality (concentrations only, average mode)
    sp_mod = np.ones((mat_c, n_sign))
    r_mod  = 1.0001
    c_mod  = 2

    i_norm = 0   # no normalisation when closure is active

    # ------------------------------------------------------------------ #
    # Initialise conc and abss                                            #
    # ------------------------------------------------------------------ #
    if ils == 1:
        conc = x0.copy()
        abss, _, _, _ = np.linalg.lstsq(conc, d, rcond=None)
    else:
        abss = x0.copy()
        conc = d @ np.linalg.pinv(abss)

    # Apply any known initial concentration values
    if has_csel:
        mask = np.isfinite(csel)
        conc[mask] = csel[mask]

    if has_ssel:
        mask = np.isfinite(ssel)
        abss[mask] = ssel[mask]

    # ------------------------------------------------------------------ #
    # PCA reproduction of data                                            #
    # ------------------------------------------------------------------ #
    d_n = d.copy()
    _, _, _, d, _ = pcarep(d_n, n_sign)

    sst_n  = np.sum(d_n ** 2)
    sst    = np.sum(d ** 2)
    sigma2 = np.sqrt(sst_n)    # initial sigma (large → first iter always improves)

    # ------------------------------------------------------------------ #
    # Tracking variables                                                  #
    # ------------------------------------------------------------------ #
    copt  = conc.copy()
    sopt  = abss.copy()
    sdopt = np.zeros(2)
    r2opt = 0.0
    itopt = 0
    total_conc = np.ones((n_sign, mat_c))

    i_dev    = 0
    i_dev_max = 50    # matches the check in als_closure2.m (idev > 50)
    solution  = 3     # default: max-iterations exceeded

    # ------------------------------------------------------------------ #
    # ALS iteration loop                                                  #
    # ------------------------------------------------------------------ #
    for niter in range(1, n_iter + 1):

        # ---- E: estimate concentrations --------------------------------
        conc = d @ np.linalg.pinv(abss)

        # Non-negativity (fnnls equivalent via scipy nnls)
        if has_nneg:
            for i in range(mat_c):
                ki, kf = n_rinic[i], n_rfin[i]
                conc2 = conc[ki:kf, :]
                if np.all(cneg[i, :] == 1):
                    for j in range(ki, kf):
                        x, _ = nnls(abss.T, d[j, :])
                        conc2[j - ki, :] = x
                conc[ki:kf, :] = conc2

        # Zero absent species
        if mat_c > 1:
            for i in range(mat_c):
                for j in range(n_sign):
                    if isp[i, j] == 0:
                        conc[n_rinic[i]:n_rfin[i], j] = 0.0

        # Unimodality on concentrations
        if has_unimod:
            for i in range(mat_c):
                ki, kf = n_rinic[i], n_rfin[i]
                conc2 = conc[ki:kf, :]
                for ii in range(n_sign):
                    if sp_mod[i, ii] == 1:
                        conc2[:, ii] = unimod(conc2[:, ii], r_mod, c_mod)
                conc[ki:kf, :] = conc2

        # Equality constraints on concentrations
        if has_csel:
            mask = np.isfinite(csel)
            conc[mask] = csel[mask]

        # Closure
        if has_closure:
            for i in range(mat_c):
                ki, kf = n_rinic[i], n_rfin[i]
                conc2 = conc[ki:kf, :]

                if i_clos[i] >= 1:
                    if t_clos1[i] == 0 and not np.isscalar(vclos1) and np.ndim(vclos1) > 0:
                        vclos1_n = vclos1[ki:kf, i] if vclos1.ndim > 1 else vclos1[ki:kf]
                    else:
                        vclos1_n = 0

                    vclos2_n = 0
                    if i_clos[i] == 2 and t_clos2[i] == 0 and not np.isscalar(vclos2):
                        vclos2_n = vclos2[ki:kf, 0] if np.ndim(vclos2) > 1 else vclos2[ki:kf]

                    conc2 = closure_multi(
                        conc2, i_clos[i],
                        sclos1[i, :], i_clos1[i],
                        t_clos1[i], t_clos2[i],
                        sclos2[i, :], 1,
                        vclos1_n, vclos2_n,
                        sto1=sto1,
                    )
                conc[ki:kf, :] = conc2

        if plot_cb is not None:
            plot_cb(conc, abss, niter, 'conc')

        # Quantitative bookkeeping (for rt, area)
        for j in range(n_sign):
            for inexp in range(mat_c):
                total_conc[j, inexp] = np.sum(conc[n_rinic[inexp]:n_rfin[inexp], j])

        # ---- F: estimate spectra ----------------------------------------
        abss, _, _, _ = np.linalg.lstsq(conc, d, rcond=None)

        # Non-negativity for spectra
        if has_nneg:
            for i in range(mat_r):
                kic, kfc = n_cinic[i], n_cfin[i]
                abss2 = abss[:, kic:kfc]
                if np.all(spneg[:, i] == 1):
                    for j in range(kic, kfc):
                        x, _ = nnls(conc, d[:, j])
                        abss2[:, j - kic] = x
                abss[:, kic:kfc] = abss2

        # Equality constraints on spectra
        if has_ssel:
            mask = np.isfinite(ssel)
            abss[mask] = ssel[mask]

        # Normalisation (only when closure is absent)
        if not has_closure:
            if i_norm == 1:
                mx = np.max(abss, axis=1)
                for i in range(n_sign):
                    if mx[i] > 0:
                        abss[i, :] /= mx[i]
            elif i_norm == 2:
                nrm = np.linalg.norm(abss, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                abss /= nrm

        if plot_cb is not None:
            plot_cb(conc, abss, niter, 'spectra')

        # ---- G: convergence check ---------------------------------------
        res   = d   - conc @ abss
        res_n = d_n - conc @ abss

        u   = np.sum(res   ** 2)
        u_n = np.sum(res_n ** 2)

        sigma   = np.sqrt(u   / (n_row * n_col))
        sigma_n = np.sqrt(u_n / (n_row * n_col))   # noqa: F841 (kept for symmetry)

        change = (sigma2 - sigma) / sigma if sigma > 0 else 0.0

        if change < 0.0:
            i_dev += 1
        else:
            i_dev = 0

        change_pct = change * 100.0
        lof_pca = np.sqrt(u   / max(sst,   1e-300)) * 100.0
        lof_exp = np.sqrt(u_n / max(sst_n, 1e-300)) * 100.0
        r2      = (sst_n - u_n) / sst_n if sst_n > 0 else 0.0

        if verbose:
            print(
                f"  [{niter:4d}]  lof(PCA)={lof_pca:.4f}%  "
                f"lof(exp)={lof_exp:.4f}%  R²={100*r2:.4f}%  "
                f"Δσ={change_pct:+.4f}%"
            )

        if change > 0.0 or niter == 1:
            sigma2 = sigma
            copt   = conc.copy()
            sopt   = abss.copy()
            sdopt  = np.array([lof_pca, lof_exp])
            r2opt  = r2
            itopt  = niter

        if abs(change_pct) < tol_sigma:
            if verbose:
                print("  >> Convergence achieved.")
            solution = 1
            return copt, sopt, sdopt, r2opt, itopt, solution

        if i_dev > i_dev_max:
            if verbose:
                print("  >> Divergence detected (fit not improving).")
            solution = 2
            return copt, sopt, sdopt, r2opt, itopt, solution

    if verbose:
        print("  >> Maximum iterations exceeded.")
    solution = 3
    return copt, sopt, sdopt, r2opt, itopt, solution
