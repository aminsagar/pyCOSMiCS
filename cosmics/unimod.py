"""
Unimodality constraint — translated from unimod() in the MCR-ALS MATLAB toolbox.
"""

import numpy as np


def unimod(profile: np.ndarray, tolerance: float, mode: int) -> np.ndarray:
    """
    Enforce unimodality on a 1-D concentration or spectral profile.

    Parameters
    ----------
    profile   : 1-D ndarray
    tolerance : float, allowed violation factor (e.g. 1.0001)
    mode      : 0=vertical (clip current point)
                1=horizontal (clip previous point)
                2=average (average the two conflicting points)

    Returns
    -------
    Unimodal copy of *profile*.
    """
    n = len(profile)
    if n < 3:
        return profile.copy()

    profile = profile.copy()
    peak = int(np.argmax(profile))

    # Enforce non-decreasing before the peak
    for i in range(1, peak + 1):
        if profile[i] < profile[i - 1] / tolerance:
            if mode == 0:
                profile[i] = profile[i - 1]
            elif mode == 1:
                profile[i - 1] = profile[i]
            else:
                avg = (profile[i] + profile[i - 1]) / 2.0
                profile[i] = avg
                profile[i - 1] = avg

    # Enforce non-increasing after the peak
    for i in range(peak + 1, n):
        if profile[i] > profile[i - 1] * tolerance:
            if mode == 0:
                profile[i] = profile[i - 1]
            elif mode == 1:
                profile[i - 1] = profile[i]
            else:
                avg = (profile[i] + profile[i - 1]) / 2.0
                profile[i] = avg
                profile[i - 1] = avg

    return profile
