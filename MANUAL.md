# COSMiCS Python — User Manual

## Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Input data](#3-input-data)
4. [Running COSMiCS](#4-running-cosmics)
5. [Command-line reference](#5-command-line-reference)
6. [Experiment types and constraints](#6-experiment-types-and-constraints)
7. [Initial estimates](#7-initial-estimates)
8. [Q-range cuts](#8-q-range-cuts)
9. [Understanding the output](#9-understanding-the-output)
10. [Worked examples](#10-worked-examples)
11. [Troubleshooting](#11-troubleshooting)
12. [Algorithm reference](#12-algorithm-reference)

---

## 1. Overview

COSMiCS (Chemometric decomposition Of SAXS data) decomposes a series of small-angle X-ray scattering (SAXS) curves measured under varying conditions into the individual scattering profiles and concentration profiles of each contributing species.

The method uses MCR-ALS (Multivariate Curve Resolution — Alternating Least Squares), a chemometric technique that finds non-negative, physically constrained solutions without requiring prior knowledge of species structures. COSMiCS extends MCR-ALS to SAXS by simultaneously analysing four complementary representations of the data: Absolute, Holtzer (I·q), Kratky (I·q²), and Porod (I·q⁴).

**Typical applications:**

- **Titration experiments** — resolving monomer and complex scattering profiles from a series of curves at increasing ligand concentration
- **Fibrillation / folding** — tracking species along an assembly or folding time course
- **SEC-SAXS** — separating overlapping species from a size-exclusion elution

---

## 2. Installation

**Requirements:** Python ≥ 3.9

```bash
git clone https://github.com/your-org/cosmics-python.git
cd cosmics-python
pip install -e .
```

This installs the `cosmics` command globally. Alternatively, run without installing:

```bash
python -m cosmics.main [options]
```

**Dependencies** (installed automatically):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.22 | Matrix math |
| scipy | ≥ 1.8 | Non-negative least squares |
| matplotlib | ≥ 3.5 | Plots and report figures |

---

## 3. Input data

### File format

Each SAXS curve is a plain-text file with three whitespace-separated columns and no header:

```
q₁   I(q₁)   σ(q₁)
q₂   I(q₂)   σ(q₂)
...
```

- **Column 1** — momentum transfer q (in 1/Å or 1/nm, consistent across all files)
- **Column 2** — scattering intensity I(q)
- **Column 3** — experimental error σ(q) (required for chi-square weighting)

If your files have header lines, pass `--header N` to skip them. If they have only two columns (no error), pass `--columns 2`; chi-square values will then be unweighted.

### File naming

All curves for one experiment must live in a single directory and share a common filename pattern, e.g. `curve_001.dat`, `curve_002.dat`, ... Files are loaded in alphabetical order. The order must correspond to the progression of experimental conditions (increasing ligand concentration, time points, etc.).

### Multiple experiments

To analyse two datasets simultaneously (e.g. a ChainA-only series and a ChainA+ChainB titration series), pass two directories to `--input`:

```bash
cosmics --input /data/exp1 /data/exp2 ...
```

Both directories must use the same `--pattern`, `--header`, and `--columns` settings.

---

## 4. Running COSMiCS

### Minimal run

```bash
cosmics \
  --input /path/to/data \
  --output /path/to/results \
  --pattern "curve*.dat" \
  --n-species 3
```

### With all common options

```bash
cosmics \
  --input /path/to/titration \
  --output /path/to/results \
  --pattern "ChainA-*.dat" \
  --header 0 \
  --columns 3 \
  --units N \
  --elim-points 2 \
  --n-species 3 \
  --init-method 1 \
  --experiment-type 1 \
  --tol 0.01 \
  --max-iter 1000 \
  --no-plots
```

### Interactive mode

To be prompted for each parameter (closest to the original MATLAB experience):

```bash
cosmics --input /path/to/data --output /path/to/results --interactive
```

---

## 5. Command-line reference

| Option | Default | Description |
|--------|---------|-------------|
| `--input DIR [DIR ...]` | *(required)* | One or more input directories |
| `--output DIR` | *(required)* | Output directory (created if absent) |
| `--pattern GLOB` | `curve*.dat` | Filename pattern for curve files |
| `--header N` | `3` | Number of header lines to skip |
| `--columns N` | `3` | Number of data columns (2 or 3) |
| `--units {A,N}` | `A` | Q units: `A` = 1/Å, `N` = 1/nm |
| `--elim-points N` | `0` | Remove N points from the start of all curves |
| `--n-species N` | `3` | Number of species to resolve |
| `--init-method {1,2}` | `1` | Initial estimate method (see §7) |
| `--two-estimates E1 E2` | — | 1-based curve indices for method 2 |
| `--experiment-type {1,2,3,4}` | `1` | Constraint preset (see §6) |
| `--tol FLOAT` | `0.01` | Convergence threshold (% change in σ) |
| `--max-iter N` | `1000` | Maximum ALS iterations |
| `--cut-abs FLOAT` | *(units-dependent)* | Q cutoff for Absolute representation |
| `--cut-holtzer FLOAT` | *(units-dependent)* | Q cutoff for Holtzer representation |
| `--cut-kratky FLOAT` | *(units-dependent)* | Q cutoff for Kratky representation |
| `--cut-porod FLOAT` | *(units-dependent)* | Q cutoff for Porod representation |
| `--closure-pattern GLOB` | — | Glob for closure input files (titration) |
| `--no-plots` | `False` | Suppress all matplotlib windows |
| `--interactive` | `False` | Enable interactive prompts |
| `--auto-clean` | `False` | Remove worst-χ² curve and repeat |

---

## 6. Experiment types and constraints

Select the experiment type with `--experiment-type`. This sets the active constraints passed to the ALS optimiser.

### Type 1 — Fibrillation / folding (default)

```
--experiment-type 1
```

**Constraints:** non-negativity + closure (sum to 1)

Use for time-course experiments where the total scattering protein concentration is constant: fibrillation, aggregation, folding/unfolding. All species concentrations sum to 1 at every measured point.

### Type 2 — Titration

```
--experiment-type 2
```

**Constraints:** non-negativity + closure + selectivity

Use when one component is absent at the endpoints of the series. For example, in a protein–protein titration, the pure species A curve at point 1 should have zero species B and zero complex. The selectivity constraint enforces known zero-concentration points.

Provide selectivity curve indices with `--selective-curves`:

```bash
cosmics ... --experiment-type 2 --selective-curves 1 50
```

### Type 3 — SEC-SAXS

```
--experiment-type 3
```

**Constraints:** non-negativity only (no closure, no normalisation)

Use when the total scattering intensity varies across the elution profile (i.e. the protein concentration is not constant). No closure constraint is applied.

### Type 4 — User-defined

```
--experiment-type 4
```

Applies whatever constraints are listed. Currently uses `[1, 3]` as default; edit `main.py` or the source directly for full control.

---

## 7. Initial estimates

The ALS algorithm requires starting estimates for the pure-component scattering curves. Two methods are available.

### Method 1 — Dissimilarity-based (default)

```
--init-method 1
```

Selects `n_species` curves that are maximally dissimilar from each other, using a greedy farthest-point algorithm in normalised intensity space. No user input required.

This replaces the SIMPLISMA (`pure`) algorithm from the original MATLAB MCR-ALS toolbox, which is not available in Python. Results are generally comparable.

**Recommendation:** use method 1 as a first pass. If the solution is poor, switch to method 2 with manually identified pure-component curves.

### Method 2 — Chi-rank based

```
--init-method 2 --two-estimates E1 E2
```

Given two curves you identify as likely pure components (by curve number, 1-based), finds the third as the curve most dissimilar from both, ranked by chi-square. Requires three species; extend the source for more.

**Example:** if curve 1 is pure species A and curve 50 is pure species B:

```bash
cosmics ... --init-method 2 --two-estimates 1 50
```

**Recommendation:** if you can identify at least two pure-component curves from the dataset (e.g. the first and last points of a titration), method 2 will give better-defined starting points.

---

## 8. Q-range cuts

COSMiCS fits four SAXS representations simultaneously. Each can be truncated at a maximum q value to exclude noisy high-q data. The cuts are set with `--cut-abs`, `--cut-holtzer`, `--cut-kratky`, `--cut-porod`.

**Default values:**

| Units | Abs | Holtzer | Kratky | Porod |
|-------|-----|---------|--------|-------|
| 1/Å (`A`) | 0.5 | 0.16 | 0.16 | 0.07 |
| 1/nm (`N`) | 3.0 | 1.6 | 1.6 | 0.7 |

If the data q-range does not reach a default cutoff, all points are included (no truncation).

**Recommendations:**

- For Holtzer and Kratky representations, include the peak but exclude the high-q tail where noise dominates. The exact value is not critical but including too much noise can degrade the decomposition.
- For Porod, a more conservative cutoff is appropriate.
- Absolute scale is usually cut at the upper end of meaningful signal.

Override with explicit values:

```bash
cosmics ... --cut-abs 0.4 --cut-holtzer 0.25 --cut-kratky 0.25 --cut-porod 0.15
```

---

## 9. Understanding the output

### HTML report

Open `results/Report/report.html` in any browser. It contains:

- **Data information** — input paths, file list, dataset plots in all four representations
- **Analysis options** — units, cuts, constraints, convergence settings
- **PCA** — eigenvalue bar chart to help judge the number of components
- **Results table** — all 8 tests with combination, lack-of-fit, variance explained, χ², convergence status
- **Per-test panels** — species curves (semilog) + concentration profiles + per-curve χ²

### Selecting the best test

The best combination is the converged test with the lowest average reduced chi-square (χ²). A value near 1.0 indicates a good fit. The software reports the best test automatically. In general:

- χ² ≈ 1 — excellent fit
- χ² < 2 — good fit
- χ² > 5 — poor fit, consider adjusting the number of species or initial estimates

### Species files

`TestXX_*/species1.dat`, `species2.dat`, `species3.dat` — two-column files (q, I) containing the recovered pure-component scattering profiles. These can be plotted with any SAXS analysis software (PRIMUS, SasView, etc.) or directly with Python/gnuplot.

### Concentration file

`TestXX_*/concentration.txt` — matrix of shape (n_curves × n_species). Row i gives the relative concentrations of each species in curve i. For closure-constrained experiments (types 1 and 2), each row sums to 1.0.

### Reconstruction

`TestXX_*/Reconstruction/curveMCRALS_NN.dat` — the fitted (reconstructed) version of each input curve.  
`TestXX_*/Reconstruction/curveExp_NN.dat` — the corresponding experimental curve (with errors).

Overlay these in a plot to assess fit quality per curve.

### Per-curve chi-square

`TestXX_*/chiSquareCurves.txt` — two columns: curve index and reduced χ². High values indicate curves that are poorly fit and may warrant investigation (outliers, contamination, transitions not captured by the model).

---

## 10. Worked examples

### Example 1 — Single-experiment fibrillation

```bash
cosmics \
  --input /data/fibril_timecourse \
  --output /results/fibril \
  --pattern "fibril_t*.dat" \
  --header 0 \
  --units N \
  --n-species 2 \
  --experiment-type 1 \
  --tol 0.01 \
  --max-iter 1000 \
  --no-plots
```

Two species (monomer + fibril). Closure enforces that the sum of concentrations equals 1 at all time points.

### Example 2 — Protein–protein titration (method 2 estimates)

```bash
cosmics \
  --input /data/titration \
  --output /results/titration \
  --pattern "ChainA-*.dat" \
  --header 0 \
  --units N \
  --n-species 3 \
  --init-method 2 \
  --two-estimates 1 50 \
  --experiment-type 1 \
  --tol 0.01 \
  --max-iter 1000 \
  --no-plots
```

Curve 1 is pure species A, curve 50 is at large excess of B. Method 2 selects the third estimate as the curve most dissimilar from both.

### Example 3 — Remove noisy leading points

```bash
cosmics \
  --input /data/experiment \
  --output /results/exp \
  --pattern "curve*.dat" \
  --header 3 \
  --units A \
  --elim-points 5 \
  --n-species 3 \
  --experiment-type 1 \
  --no-plots
```

Removes the first 5 q-points from all curves (useful when low-q data shows inter-particle effects or beam artefacts).

### Example 4 — Custom q-cuts

```bash
cosmics \
  --input /data/experiment \
  --output /results/exp \
  --pattern "curve*.dat" \
  --units N \
  --cut-abs 0.45 \
  --cut-holtzer 0.30 \
  --cut-kratky 0.30 \
  --cut-porod 0.20 \
  --n-species 3 \
  --no-plots
```

---

## 11. Troubleshooting

### All 8 tests diverge

- Try a different number of species (`--n-species 2` or `4`).
- Check the PCA eigenvalues in the report — the number of significantly elevated eigenvalues suggests the number of components.
- Try method 2 initial estimates with manually identified pure-component curves.
- Increase `--max-iter` or loosen `--tol`.

### Concentration of one species reaches unrealistic values

This typically means one species is not covered by the closure constraint. Ensure you are using the correct `--experiment-type`. For a single-experiment run, type 1 applies closure over all species simultaneously.

### Species curves contain negative intensities

Non-negativity is enforced on both concentrations and spectra. If negatives appear, check that:
- Errors in column 3 are not zero (zero errors cause numerical issues in chi-square weighting).
- The number of species is not too high relative to the information content of the data.

### Chi-square is very large (> 10)

The model does not fit the data well. Possible causes:
- Wrong number of species.
- Poor initial estimates — try method 2.
- Q-range too wide, including noisy data — tighten the q-cuts.
- Data quality issues (outlier curves) — inspect `chiSquareCurves.txt` and consider using `--auto-clean`.

### Report figures are blank or missing

Ensure matplotlib is installed. If running on a headless server, `--no-plots` must be set; the report still generates SVG figures via the non-interactive Agg backend.

---

## 12. Algorithm reference

### MCR-ALS

The data matrix **D** (n_curves × n_q) is factored as:

```
D ≈ C · S
```

where **C** (n_curves × n_species) contains concentration profiles and **S** (n_species × n_q) contains pure-component scattering curves. The ALS procedure alternates between solving for **C** and **S** by least squares, applying physical constraints after each step until convergence.

### Constraints

| Code | Constraint | Applied to |
|------|------------|-----------|
| 1 | Non-negativity (NNLS) | Concentrations and spectra |
| 2 | Unimodality | Concentrations |
| 3 | Closure (sum to constant) | Concentrations |
| 4 | Selectivity (known zeros) | Concentrations |
| 5 | Equality (fixed curves) | Spectra |

### PCA reproduction

Before ALS, the data matrix is reproduced by its first n_species principal components. This removes noise and ensures the model is not overfitting measurement noise.

### Representation scaling

Holtzer, Kratky, and Porod matrices are each scaled by the ratio of the first eigenvalue of the Absolute matrix to their own first eigenvalue. This equalises the weight of each representation in the augmented matrix and prevents any single representation from dominating the solution.

### Convergence

The optimisation converges when the relative change in the fitting standard deviation between consecutive iterations falls below `--tol` (%). It also terminates early if the fit fails to improve for 50 consecutive iterations (divergence). The solution is always taken from the iteration with the lowest fitting error, not the last iteration.

### Representations tested

COSMiCS runs ALS on all 8 combinations of the four representations:

| Test | Combination |
|------|-------------|
| 1 | Absolute |
| 2 | Absolute + Holtzer |
| 3 | Absolute + Kratky |
| 4 | Absolute + Porod |
| 5 | Absolute + Holtzer + Kratky |
| 6 | Absolute + Holtzer + Porod |
| 7 | Absolute + Kratky + Porod |
| 8 | Absolute + Holtzer + Kratky + Porod |

The best combination is selected as the converged test with the lowest average reduced chi-square.

---

*COSMiCS Python — developed by Amin Sagar | Pau Bernadó Group, CBS CNRS*  
*Original MATLAB version by Fatima Herranz-Trillo and Amin Sagar*
