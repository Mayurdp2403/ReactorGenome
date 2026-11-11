"""Readings and published results from the 2026 CHE331 lab reports.

Used only to test that the engine reproduces a report's numbers when given the
report's conventions. Never loaded by the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CSTR (cstr_lab.pdf)
CSTR_V = 2.15
CSTR_CA0 = 0.05
CSTR_LINF = 0.10
CSTR_LAM0 = {31.0: 8.25, 46.0: 10.76, 60.0: 12.42}
CSTR_READ = [(31, 15, (4.79, 4.78, 4.77)), (31, 20, (5.05, 5.05, 5.04)), (31, 25, (5.36, 5.32, 5.30)),
             (46, 15, (5.21, 5.21, 5.21)), (46, 20, (5.45, 5.44, 5.44)), (46, 25, (5.64, 5.61, 5.58)),
             (60, 15, (5.54, 5.55, 5.53)), (60, 20, (5.78, 5.77, 5.79)), (60, 25, (5.98, 5.99, 5.95))]
CSTR_TAU = {15: 8.60, 20: 6.45, 25: 5.16}
CSTR_X = [0.42577, 0.39289, 0.35869, 0.52064, 0.49875, 0.48311, 0.55844, 0.53896, 0.52327]
CSTR_KPOINT = [3.003, 3.308, 3.381, 5.269, 6.155, 7.009, 6.661, 7.862, 8.924]
CSTR_K = {31.0: 3.1629, 46.0: 5.8541, 60.0: 7.4362}
CSTR_R2 = {31.0: 0.9972, 46.0: 0.9869, 60.0: 0.9861}
CSTR_EA_KJ = 25.07
CSTR_A = 6.71e4


def cstr_frame(lam0_once=True) -> pd.DataFrame:
    """As typed on the lab sheet: total flow, τ, Λ0 (once per temperature), three readings."""
    rows = []
    for T, Q, r in CSTR_READ:
        first = Q == 15
        rows.append({"T_C": float(T), "Q_LPH": float(Q), "tau_min": CSTR_TAU[Q],
                     "lam0": CSTR_LAM0[T] if (first or not lam0_once) else np.nan, "r1": r[0], "r2": r[1], "r3": r[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- PFR (PFR_Report.pdf)
PFR_V = 0.55
PFR_CA0 = 0.035
PFR_LINF = 0.1
PFR_LAM0 = {40.0: 10.58, 50.0: 11.42, 60.0: 12.69}
PFR_TAU = {20: 1.650, 30: 1.100, 40: 0.825}
PFR_READ = [(40, 20, (7.35, 6.92, 6.82, 6.29)), (40, 30, (7.32, 7.07, 6.78, 6.76)), (40, 40, (7.07, 7.15, 7.20, 7.06)),
            (50, 20, (7.64, 6.65, 7.06, 7.22)), (50, 30, (7.29, 6.54, 7.05, 7.29)), (50, 40, (7.32, 6.49, 6.62, 7.66)),
            (60, 20, (7.04, 7.27, 7.17, 7.36)), (60, 30, (7.06, 7.47, 7.52, 7.76)), (60, 40, (7.83, 7.84, 8.15, 8.31))]
PFR_KPOINT = [9.5886, 13.5767, 17.0695, 10.5179, 16.3776, 21.9999, 13.3463, 18.5024, 20.3340]
PFR_K = {40.0: 11.73829, 50.0: 13.74857, 60.0: 15.72971}
PFR_EA_KJ = 12.700
PFR_A = 1545.705


def pfr_frame() -> pd.DataFrame:
    rows = []
    for T, Q, r in PFR_READ:
        rows.append({"T_C": float(T), "Q_LPH": float(Q), "tau_min": PFR_TAU[Q], "lam0": PFR_LAM0[T],
                     "r1": r[0], "r2": r[1], "r3": r[2], "r4": r[3]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RTD (RTD_Report.pdf)
RTD_DT = 0.5
RTD_CAL = (1.71174563e-05, -1.68852096e-04)
RTD_RUNS = {
    "CSTR 15 LPH": [2, 191, 188, 176, 166, 161, 153, 142, 129, 135, 128, 121, 115, 108, 102, 83, 89, 85, 79, 73, 73, 77, 68, 62, 59, 59, 55, 53, 49,
                    48, 44, 43, 41, 39, 35, 36, 34, 33, 30, 29, 24, 26, 25, 24, 24, 18, 17, 16, 16, 15, 14, 14, 14, 13, 10, 10, 10, 11, 10, 9, 9],
    "CSTR 10 LPH": [1, 186, 175, 174, 170, 159, 159, 151, 144, 143, 136, 131, 127, 126, 124, 116, 110, 109, 108, 100, 91, 88, 81, 81, 79, 72, 78,
                    63, 65, 67, 64, 62, 59, 56, 55, 55, 52, 49, 48, 45, 44, 44, 40, 40, 39, 37, 37, 35, 34, 32, 31, 30, 28, 28, 27, 27, 26, 25,
                    24, 24, 23, 23, 21, 17, 16, 16, 15, 15, 15, 14, 13, 13, 11, 11, 11, 11, 10, 11, 9, 9],
    "Packed bed 15 LPH": [1, 1, 1, 1, 1, 1, 83, 231, 542, 719, 683, 470, 323, 222, 125, 108, 53, 40, 37, 15, 10, 8, 6, 4],
    "Packed bed 10 LPH": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 26, 72, 194, 367, 515, 513, 553, 496, 403, 324, 214, 164, 125, 101, 78, 58, 44, 41,
                          31, 26, 16, 15, 11, 10, 8, 7, 6, 5, 4],
}
RTD_RESULTS = {  # tm (min), σ² (min²), σ (min), N or Pe
    "CSTR 15 LPH": {"tm": 8.244, "var": 45.276, "sigma": 6.729, "N": 1.501},
    "CSTR 10 LPH": {"tm": 11.263, "var": 84.276, "sigma": 9.180, "N": 1.505},
    "Packed bed 15 LPH": {"tm": 5.169, "var": 1.653, "sigma": 1.286, "Pe": 35.93, "tau": 4.896},
    "Packed bed 10 LPH": {"tm": 9.083, "var": 4.087, "sigma": 2.022, "Pe": 44.04, "tau": 8.688},
}


def rtd_sheet() -> pd.DataFrame:
    """One time column and one column per run, as the app's RTD sheet takes it."""
    n = max(len(v) for v in RTD_RUNS.values())
    out = {"t_min": np.arange(n) * RTD_DT}
    for i, v in enumerate(RTD_RUNS.values(), start=1):
        out[f"s{i}"] = np.r_[np.asarray(v, float), np.full(n - len(v), np.nan)]
    return pd.DataFrame(out)
