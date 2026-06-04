# COSMiCS — Python

**Chemometric decomposition Of SAXS data**

Python port of the [COSMiCS2021](https://github.com/aminsagar/COSMiCS2021) MATLAB software.  
Decomposes multi-curve SAXS datasets into individual scattering components using MCR-ALS (Multivariate Curve Resolution — Alternating Least Squares).

> If you use this software, please cite:  
> Herranz-Trillo F., Groenning M., van Maarschalkerweerd A., Tauler R., Vestergaard B., Bernadó P.  
> *Structure and Thermodynamics of Transient Protein–Protein Complexes by Chemometric Decomposition of SAXS Data Sets.*  
> **Structure** 2017, 25, 5–15. https://doi.org/10.1016/j.str.2016.10.023

---

## What it does

COSMiCS takes a series of SAXS curves measured at varying conditions (titration, fibrillation, SEC-SAXS) and resolves them into the scattering profiles and concentration profiles of each contributing species — without requiring prior knowledge of their structures.

The decomposition runs across eight combinations of SAXS representations (Absolute, Holtzer, Kratky, Porod) simultaneously and selects the best solution by reduced chi-square.

---

## Installation

**Requirements:** Python ≥ 3.9, numpy ≥ 1.22, scipy ≥ 1.8, matplotlib ≥ 3.5

```bash
git clone https://github.com/your-org/cosmics-python.git
cd cosmics-python
pip install -e .
```

Or without installing:

```bash
pip install numpy scipy matplotlib
python -m cosmics.main --help
```

---

## Quick start

```bash
cosmics \
  --input /path/to/data \
  --output /path/to/results \
  --pattern "curve*.dat" \
  --header 0 \
  --n-species 3 \
  --units N \
  --no-plots
```

Results are written to the output directory. Open `results/Report/report.html` in a browser to view the summary.

---

## Input format

Plain-text files with columns: **q  I(q)  σ(q)**

```
0.001   2.161e+09   1.182e+08
0.002   2.262e+09   8.357e+07
...
```

All curves must cover the same q-range (or be alignable to one). Errors in the third column are required for chi-square calculation.

---

## Output

```
results/
├── Report/
│   ├── report.html          # Full HTML report with all figures
│   └── *.svg                # Individual figures
├── Test01_A/
│   ├── species1.dat         # Recovered scattering curve, species 1
│   ├── species2.dat
│   ├── species3.dat
│   ├── concentration.txt    # Concentration profiles (n_curves × n_species)
│   ├── chiSquareCurves.txt  # Per-curve fit quality
│   └── Reconstruction/      # Fitted vs experimental curves
├── Test02_A+H/
│   └── ...                  # Same structure for each of the 8 combinations
├── eigenvalues.txt
├── eigenvectors.txt
├── scale.txt
└── info.txt
```

---

## Differences from the MATLAB version

| Feature | MATLAB | Python |
|---------|--------|--------|
| Non-negative LS | `fnnls` (MCR-ALS toolbox) | `scipy.optimize.nnls` |
| Initial estimates (Method 1) | `pure` / SIMPLISMA (MCR-ALS) | Greedy dissimilarity selection |
| Initial estimates (Method 2) | Chi-rank (`ChiSelection.m`) | Identical port |
| UI | Interactive prompts + GUI folder picker | `argparse` CLI; `--interactive` for prompts |
| Report | MATLAB `publish` + HTML | matplotlib SVG + HTML |
| Dependencies | MATLAB + MCR-ALS toolbox | numpy, scipy, matplotlib |

Numerical results are expected to be equivalent. See `PORTING_NOTES.md` for full details.

---

## License

See `LICENSE`.  
Original MATLAB code © Fatima Herranz-Trillo, Amin Sagar, Pau Bernadó Group, CBS CNRS.
