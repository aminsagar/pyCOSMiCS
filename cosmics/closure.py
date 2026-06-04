"""
Closure constraint for MCR-ALS concentration profiles.
Translated from closure_multi.m (COSMiCS) / closure.m (MCR-ALS).
"""

from __future__ import annotations

import numpy as np


def closure_multi(
    conc: np.ndarray,
    iclos: int,
    sclos1: np.ndarray,
    iclos1: int,
    tclos1: float,
    tclos2: float,
    sclos2: np.ndarray,
    iclos2: int,
    vclos1,
    vclos2,
    sto1: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply closure to a concentration sub-matrix.

    Parameters
    ----------
    conc   : ndarray (n_rows, n_species)
    iclos  : 0=no closure, 1=one, 2=two closure conditions
    sclos1 : (n_species,) 1/0 — species in first closure
    iclos1 : 1=equal, 2=lower-or-equal
    tclos1 : first closure constant (0 → use vclos1)
    tclos2 : second closure constant
    sclos2 : (n_species,) species in second closure
    iclos2 : type for second closure
    vclos1 : vectorial first closure (array or scalar 0)
    vclos2 : vectorial second closure
    sto1   : stoichiometry per species (default: all ones)
    """
    n_rows, n_spec = conc.shape
    if sto1 is None:
        sto1 = np.ones(n_spec)

    def _is_zero(v):
        return np.isscalar(v) and v == 0

    # Weighted sums
    summ1 = np.zeros(n_rows)
    summ2 = np.zeros(n_rows)
    for ns in range(n_spec):
        if iclos >= 1 and sclos1[ns] == 1:
            summ1 += sto1[ns] * conc[:, ns]
        if iclos == 2 and sclos2[ns] == 1:
            summ2 += conc[:, ns]

    summ1[summ1 == 0] = 1.0
    if iclos == 2:
        summ2[summ2 == 0] = 1.0

    max1 = np.max(summ1)
    max2 = np.max(summ2) if iclos == 2 else 1.0

    conc = conc.copy()

    for ns in range(n_spec):
        if iclos == 1 and sclos1[ns] == 1:
            if iclos1 == 1:
                if _is_zero(vclos1):
                    conc[:, ns] = conc[:, ns] * tclos1 * sto1[ns] / summ1
                else:
                    conc[:, ns] = conc[:, ns] * vclos1 / summ1
            elif iclos1 == 2:
                if _is_zero(vclos1):
                    conc[:, ns] = conc[:, ns] * tclos1 / max1
                else:
                    conc[:, ns] = conc[:, ns] * vclos1 / max1

        elif iclos == 2:
            if sclos1[ns] == 1 and sclos2[ns] == 1:
                raise ValueError("Species cannot belong to both closures simultaneously.")

            if sclos1[ns] == 1:
                if iclos1 == 1:
                    if _is_zero(vclos1):
                        conc[:, ns] = conc[:, ns] * tclos1 / summ1
                    else:
                        conc[:, ns] = conc[:, ns] * vclos1 / summ1
                elif iclos1 == 2:
                    if _is_zero(vclos1):
                        conc[:, ns] = conc[:, ns] * tclos1 / max1
                    else:
                        conc[:, ns] = conc[:, ns] * vclos1 / max1

            if sclos2[ns] == 1:
                # Original MATLAB uses iclos1 (not iclos2) here — preserved.
                if iclos1 == 1:
                    if _is_zero(vclos2):
                        conc[:, ns] = conc[:, ns] * tclos2 / summ2
                    else:
                        conc[:, ns] = conc[:, ns] * vclos2 / summ2
                elif iclos1 == 2:
                    if _is_zero(vclos2):
                        conc[:, ns] = conc[:, ns] * tclos2 / max2
                    else:
                        conc[:, ns] = conc[:, ns] * vclos2 / max2

    return conc
