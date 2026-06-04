# COSMiCS Python Port — Developer Notes

## What this is

COSMiCS (Chemometric decomposition Of SAXS data) was originally written in MATLAB and depends on the MCR-ALS toolbox. This directory contains a complete Python translation of the COSMiCS2021 codebase. The goal was a faithful, line-by-line port — not a redesign — so researchers familiar with the MATLAB version can read the Python and recognise it.

---

## Repository layout

```
cosmics_python/
├── cosmics/
│   ├── __init__.py
│   ├── pca.py               pcarep() — truncated SVD
│   ├── io.py                file loading and curve alignment
│   ├── preprocessing.py     normalisation, scaling, q-cuts, transforms
│   ├── closure.py           closure constraint (closure_multi.m)
│   ├── unimod.py            unimodality constraint (unimod from MCR-ALS)
│   ├── metrics.py           chi-square and curve reconstruction
│   ├── initial_estimates.py seed-curve selection (chi-rank + dissimilarity)
│   ├── matrix_builder.py    augmented input matrix builder
│   ├── als.py               MCR-ALS optimisation loop (als_closure2.m)
│   ├── plots.py             matplotlib plots
│   ├── report.py            HTML report generator
│   └── main.py              CLI entry point
├── requirements.txt
└── setup.py
```

---

## Module-by-module correspondence

### `pca.py` ← `pcarep()` from MCR-ALS

`pcarep` is a helper from the MCR-ALS MATLAB toolbox, not included in the COSMiCS repository. It performs a truncated SVD and returns the PCA-reproduced data matrix. The Python version uses `numpy.linalg.svd` directly:

```python
U, s, Vt = np.linalg.svd(data, full_matrices=False)
d_reprod  = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

The fitting error `sd` is returned as a percentage of total variance, matching the MATLAB output.

---

### `io.py` ← COSMiCS_multi.m §1 (file loading)

MATLAB uses `uigetdir` (GUI folder picker) and `textscan`. Python replaces these with `pathlib.Path.glob` and `numpy.loadtxt`. Curve alignment mirrors the three-case logic in COSMiCS_multi.m lines 162–206:

1. All curves same length → no change.
2. Q-vectors agree on the minimum length → trim to that.
3. Q-vectors disagree → find the largest starting q-value, trim starts, then trim to common length.

---

### `preprocessing.py` ← COSMiCS_multi.m §2–§6

Four operations are translated:

**Normalisation** (lines 217–223): Every curve is scaled so the minimum I(0) across all curves equals 100.

**Scaling** (lines 228–235): Every curve is divided by its intensity at a reference point (row index 19, i.e. the 20th q-point in MATLAB 1-indexing). This equates the curves at a mid-q reference value.

**Q-range cutting**: `find_cut_index` returns the first array index where q ≥ a threshold, clamped to valid bounds. Default thresholds per unit system are encoded in `default_cuts()`.

**Representation transforms** (crearMat_manualpure_rank.m lines 26–54): The function `apply_transforms` computes all four SAXS representations:

| Representation | Formula |
|---|---|
| Absolute | I(q) |
| Holtzer | I(q) · q |
| Kratky | I(q) · q² |
| Porod | I(q) · q⁴ |

The output matrices follow the MATLAB row-is-curve convention: shape `(n_curves, n_q_points)`.

---

### `closure.py` ← `closure_multi.m`

Direct translation of the closure constraint. For each experiment sub-matrix, species concentrations are rescaled so that their weighted sum equals a target value (equal-condition closure) or stays below it (lower-than-condition closure).

The MATLAB code contains a quirk preserved intentionally: the second-closure branch uses `iclos1` (the type of the first closure) rather than `iclos2`. This is kept as-is to maintain numerical equivalence with the original.

The stoichiometry vector `sto1` is hard-coded as `[1, 1, 1]` in `closure_multi.m`. In Python it defaults to `np.ones(n_species)` and can be overridden.

---

### `unimod.py` ← `unimod()` from MCR-ALS

The unimodality constraint is not in the COSMiCS repository but is called from `als_closure2.m`. It was re-implemented from the standard MCR-ALS algorithm description:

1. Locate the peak index.
2. Sweep left-to-peak: if a value drops below `previous / tolerance`, correct it.
3. Sweep peak-to-right: if a value rises above `previous * tolerance`, correct it.

Three correction modes: vertical (clip current), horizontal (clip previous), average (replace both with their mean). The default mode used by COSMiCS is `cmod = 2` (average).

---

### `metrics.py` ← `compare2curves.m`, `reconstCurvas.m`

**`compare2curves`**: Computes the reduced chi-square between an experimental and a synthetic curve. The synthetic curve is first optimally scaled by a least-squares factor `ratio = Σ(I_exp · I_syn · w) / Σ(I_syn² · w)` where `w = 1/E²`. Then `χ² = Σ((I_exp − ratio · I_syn)² · w) / (N−1)`.

**`reconstruct_curves`**: Matrix multiplication `(conc @ spectra).T`. The MATLAB triple-nested loop is replaced by a single numpy operation.

**`compute_chi_all`**: Applies `compare2curves` to every curve in the dataset and returns per-curve chi-square values and fitted curves.

---

### `initial_estimates.py` ← `ChiSelection.m`, crearMat_manualpure_rank.m (Method 2 path)

Two methods for selecting the starting curves (initial estimates):

**Method 1 — dissimilarity-based** replaces the MATLAB `pure()` function (SIMPLISMA algorithm from MCR-ALS, not included in COSMiCS). The replacement uses greedy farthest-point selection in normalised intensity space: seed with the curve that has the largest norm, then repeatedly pick the curve whose minimum distance to the current selected set is largest. This is not identical to SIMPLISMA but produces similarly well-separated initial estimates.

**Method 2 — chi-rank based** is a faithful port of `ChiSelection.m`. Given two user-provided seed curves:
1. Compute chi-square of every other curve vs seed 1 → rank list R1.
2. Compute chi-square of every other curve vs seed 2 → rank list R2.
3. The third estimate is the curve with the highest average rank `(R1 + R2) / 2`.

---

### `matrix_builder.py` ← `crearMat_manualpure_rank.m`

Builds the augmented input matrix passed to ALS. The eight combinations of representations (A, A+H, A+K, A+P, A+H+K, A+H+P, A+K+P, A+H+K+P) are defined as `COMBINATIONS`. For each included non-absolute representation, a scale factor is computed as the ratio of the first eigenvalue of the absolute matrix to the first eigenvalue of that representation's matrix:

```
scale_holtzer = eigenvalue1(Abs) / eigenvalue1(Holtzer)
```

This equalises the relative contribution of each representation block in the augmented matrix. The `build_initial_estimates` function extracts the corresponding rows of each block for the selected seed curves.

---

### `als.py` ← `als_closure2.m`

The core MCR-ALS loop. The most complex translation in the port.

**Matrix conventions**: In both MATLAB and Python, the data matrix `d` has shape `(n_curves, n_q_total)` — rows are curves, columns are q-points. Initial estimates are spectra, shape `(n_species, n_q_total)`.

**Key MATLAB → Python equivalences**:

| MATLAB | Python | Meaning |
|--------|--------|---------|
| `conc = d / abss` | `conc = d @ pinv(abss)` | Solve conc·abss = d for conc |
| `abss = conc \ d` | `lstsq(conc, d)` | Solve conc·abss = d for abss |
| `fnnls(A'A, A'b)` | `scipy.optimize.nnls(A, b)` | Non-negative least squares |

**`fnnls` vs `scipy.optimize.nnls`**: The MATLAB `fnnls` function takes the normal-equation form `(AᵀA, Aᵀb)`. Scipy's `nnls` takes the standard form `(A, b)` and forms the normal equations internally. The equivalence is:

- For concentrations: `fnnls(abss·abssᵀ, abss·d[j,:]ᵀ)` → `nnls(abssᵀ, d[j,:])`
- For spectra: `fnnls(concᵀ·conc, concᵀ·d[:,j])` → `nnls(conc, d[:,j])`

**Data-structure handling**: The MATLAB code supports three augmented-matrix modes (column-augmented, row-augmented, both). COSMiCS uses mode 3 (both) for multi-experiment multi-representation runs. Python computes `n_rinic`/`n_rfin` (row block boundaries) and `n_cinic`/`n_cfin` (column block boundaries) using the same cumulative-sum logic.

**Closure defaults**: In `als_closure2.m` the closure species assignment is hard-coded:
```matlab
sclos1(i,:) = [0,0,0];
sclos1(i,i) = 1;
sclos1(i,3) = 1;
```
(for 3 species, 1-indexed). Python mirrors this: `sclos1[i, i % n_sign] = 1; sclos1[i, n_sign-1] = 1`.

**Convergence**: Three exit conditions match the MATLAB exactly:
1. `|Δσ| < tol_sigma` → solution=1 (converged)
2. `i_dev > 50` → solution=2 (diverged; note: `idevmax=10` in the MATLAB comments is not the actual check)
3. Loop exhausted → solution=3 (max iterations)

---

### `plots.py` ← `plots.m`, inline figure code

MATLAB figures → matplotlib. All `semilogy`, `plot`, `bar`, `subplot`, and `scatter` calls are translated. The `jet` colormap is used throughout to match the MATLAB default. Figures can be shown interactively or suppressed with `--no-plots`.

---

### `report.py` ← `publishReport_two.m`

HTML report generation using plain Python string building (no template engine dependency). Figures are saved as SVG using matplotlib's `Agg` backend. A minimal CSS is written if the original `style.css` is not found. The report structure — data info, analysis options, PCA, results table, per-test solution plots — mirrors the MATLAB original exactly.

---

### `main.py` ← `COSMiCS_multi.m`

The orchestration script. All MATLAB `input()`, `pause`, and `uigetdir` calls are replaced by `argparse` arguments. The `--interactive` flag re-enables terminal prompts for exploratory sessions.

The 8-combination workflow loop is preserved. The automatic curve-elimination loop (which is disabled by default in the MATLAB version — `eliminar = 'n'`) is also off by default; it can be enabled with `--auto-clean`.

---

## What was not ported

| MATLAB feature | Status |
|---|---|
| `pure()` / SIMPLISMA | Replaced by greedy dissimilarity selection |
| `mcrbandsAUTO` (MCR bands) | Commented out in original, not ported |
| `montecarlo` module | Commented out in original, not ported |
| `moduloSECSAXS` | Stub (SEC-SAXS mode sets non-negativity-only constraints) |
| `autoPR` / AutoGnom | Commented out in original, not ported |
| MATLAB `.mat` workspace save | Not ported (use Python pickle if needed) |
| `dbtype` (print file header) | Replaced by `numpy.loadtxt` with `skiprows` |

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Basic run (fibrillation experiment, 3 species, Angstrom units):

```bash
cosmics \
  --input /data/experiment1 /data/experiment2 \
  --output /results/my_run \
  --pattern "curve*.dat" \
  --header 3 \
  --columns 3 \
  --units A \
  --n-species 3 \
  --init-method 1 \
  --experiment-type 1 \
  --tol 0.01 \
  --max-iter 1000
```

Chi-based initial estimates (Method 2), with curves 1 and 51 as seeds:

```bash
cosmics \
  --input /data/titration \
  --output /results/titration \
  --init-method 2 \
  --two-estimates 1 51 \
  --experiment-type 2
```

Interactive mode (mirrors the original MATLAB prompts):

```bash
cosmics --input /data/exp --output /results/exp --interactive
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.22 | All matrix math |
| scipy | ≥ 1.8 | `scipy.optimize.nnls` (replaces fnnls) |
| matplotlib | ≥ 3.5 | All plotting and report figures |

No other dependencies. The MCR-ALS MATLAB toolbox is not required.

---

## Numerical equivalence

The Python port is expected to produce numerically equivalent results to the MATLAB version given the same input data and settings. Known minor differences:

- **`nnls` vs `fnnls`**: `scipy.optimize.nnls` uses an active-set method; `fnnls` uses a faster projected-gradient approach. Both find the same solution but may take a different number of internal iterations. The ALS outer-loop convergence trajectory may differ slightly.
- **`dissimilarity_based_selection` vs `pure`**: The replacement seed-selection algorithm is not identical to SIMPLISMA. For Method 2 (chi-based), results are numerically identical.
- **SVD sign convention**: NumPy's SVD may flip the sign of singular vectors relative to MATLAB's. This does not affect the ALS solution since the algorithm is invariant to sign flips in the initialisation.

To validate numerically against the MATLAB version: run both on the same dataset, compare `species*.dat` and `concentration.txt` outputs in the `Test01_A/` folder (the simplest, absolute-only combination). These should agree to within floating-point rounding.
