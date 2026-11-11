"""Numbers taken from the four lab reports used to check the app.

CSTR and PFR: Group 8, October 2026.
Batch and RTD: Group 21.
The timed CSTR set is the earlier conductivity sheet in which the readings were
taken at τ, τ+1 and τ+2. It is the run the proposal talks about, where 28 °C
came out ahead of 40 °C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CSTR, Group 8. V = 2.15 L, both feeds 0.07 N, Λ∞ = 3.75 mS/cm.
# Four readings. The report drops R0 and averages R1–R3.
# ---------------------------------------------------------------------------

CSTR_GROUP8 = {
    "name": "Group 8 CSTR",
    "V": 2.15,
    "C_fresh_A": 0.07,
    "C_fresh_B": 0.07,
    "lam_inf": 3.75,
    "drop_first": True,
    "runs": [
        # T, qA, qB, lam0, readings R0 R1 R2 R3
        (31, 7.5, 7.5, 8.17, [4.91, 4.91, 4.77, 4.78]),
        (31, 10.0, 10.0, 8.17, [5.12, 5.12, 5.10, 5.10]),
        (31, 12.5, 12.5, 8.17, [5.31, 5.31, 5.29, 5.30]),
        (41, 7.5, 7.5, 9.56, [5.68, 5.63, 5.58, 5.54]),
        (41, 10.0, 10.0, 9.56, [5.32, 5.29, 5.28, 5.26]),
        (41, 12.5, 12.5, 9.56, [5.64, 5.62, 5.61, 5.59]),
        (50, 7.5, 7.5, 11.59, [5.55, 5.56, 5.55, 5.54]),
        (50, 10.0, 10.0, 11.59, [5.79, 5.80, 5.81, 5.82]),
        (50, 12.5, 12.5, 11.59, [5.91, 5.90, 5.89, 5.89]),
    ],
}

# Published graph-based k, L/mol/min, and Arrhenius results. Used only as checks.
CSTR_GROUP8_PUBLISHED = {
    "k": {31: 37.3586, 41: 32.6264, 50: 49.1770},
    "Ea_kJ": 11.15,
    "A": 2814.29,
    "r2": 0.3852,
    "X": {
        (31, 15): 0.7579,
        (31, 20): 0.6931,
        (31, 25): 0.6493,
        (41, 15): 0.6845,
        (41, 20): 0.7372,
        (41, 25): 0.6804,
        (50, 15): 0.7704,
        (50, 20): 0.7372,
        (50, 25): 0.7266,
    },
}

# Timed readings: Λ at τ, τ+1, τ+2. Λ∞ = 0.1 mS/cm. V = 2.15 L.
CSTR_TIMED = {
    "name": "Timed CSTR (28, 40, 50 °C)",
    "V": 2.15,
    "C_fresh_A": 0.07,
    "C_fresh_B": 0.07,
    "lam_inf": 0.1,
    "drop_first": False,
    "runs": [
        (28, 7.5, 7.5, 8.16, [4.11, 4.21, 4.29], "tau+"),
        (28, 10, 10, 8.16, [5.18, 5.18, 5.18], "tau+"),
        (28, 12.5, 12.5, 8.16, [5.56, 5.53, 5.52], "tau+"),
        (40, 7.5, 7.5, 9.51, [5.32, 5.30, 5.29], "tau+"),
        (40, 10, 10, 9.51, [5.53, 5.52, 5.52], "tau+"),
        (40, 12.5, 12.5, 9.51, [5.88, 5.84, 5.81], "tau+"),
        (50, 7.5, 7.5, 11.61, [5.72, 5.74, 5.75], "tau+"),
        (50, 10, 10, 11.61, [5.96, 5.99, 5.95], "tau+"),
        (50, 12.5, 12.5, 11.61, [6.13, 6.10, 6.07], "tau+"),
    ],
}

CSTR_TIMED_PUBLISHED = {
    "k": {28: 5.221, 40: 5.264, 50: 7.950},
    "Ea_kJ": 14.76,
    "A": 1775.0,
    "X15": {28: 0.4909, 40: 0.4470, 50: 0.5103},
}

# ---------------------------------------------------------------------------
# PFR, Group 8. Coil volume 0.55 L. Λ∞ = 0.1 mS/cm.
# Λ1 near one residence time, then 8, 9 and 10 min.
# ---------------------------------------------------------------------------

PFR_GROUP8 = {
    "name": "Group 8 PFR",
    "V": 0.55,
    "C_fresh_A": 0.07,
    "C_fresh_B": 0.07,
    "lam_inf": 0.1,
    "runs": [
        # T, Q total LPH, lam0, four readings
        (40, 20, 10.52, [7.03, 7.16, 7.04, 6.93]),
        (40, 30, 10.52, [7.04, 7.22, 7.09, 7.03]),
        (40, 40, 10.52, [7.12, 7.20, 7.07, 7.21]),
        (50, 20, 11.31, [7.29, 7.20, 7.43, 7.47]),
        (50, 30, 11.31, [7.48, 7.48, 7.54, 7.63]),
        (50, 40, 11.31, [7.55, 7.55, 7.56, 7.78]),
        (60, 20, 12.69, [7.35, 7.35, 7.82, 6.94]),
        (60, 30, 12.69, [7.50, 7.35, 7.34, 7.65]),
        (60, 40, 12.69, [7.57, 7.59, 7.56, 7.85]),
    ],
}

PFR_GROUP8_PUBLISHED = {
    "k": {40: 0.81, 50: 1.90, 60: 2.35},  # from the free fit, L/mol/min
    "delta_min": {40: 16.07, 50: 6.59, 60: 7.37},
    "Ea_kJ": 46.5,
    "A": 5.03e7,
    "X": {(40, 20): 0.3340, (50, 20): 0.3535, (60, 20): 0.4230},
}

# ---------------------------------------------------------------------------
# Batch, Group 21. Full 50.2 °C conductivity trace, μS/cm.
# 30 °C and 40 °C are the rate constants printed in the conclusion. The raw
# traces for those two temperatures were figures, not a number table.
# ---------------------------------------------------------------------------

BATCH_50_EC = [
    15430, 10370, 9410, 9190, 8920, 8670, 8380, 8290, 8040, 7880, 7720, 7570,
    7450, 7320, 7200, 7100, 6990, 6890, 6800, 6720, 6650, 6580, 6510, 6450,
    6400, 6340, 6300, 6250, 6200, 6160, 6120, 6090, 6040, 6010, 5980, 5950,
    5930, 5900, 5870, 5840, 5820, 5800, 5780, 5760, 5740, 5720, 5700, 5680,
    5670, 5650, 5630, 5620, 5610, 5590, 5580, 5560, 5550, 5540, 5530, 5520,
    5510, 5500, 5480, 5470, 5460, 5450, 5450, 5440, 5430, 5420, 5410, 5400,
    5400, 5390,
]
BATCH_50 = {
    "T": 50.2,
    "C0": 0.040,
    "lam0": 15430.0,  # μS/cm
    "lam_inf": 5390.0,
    "t_s": list(range(0, 5 * len(BATCH_50_EC), 5)),
    "ec": BATCH_50_EC,
    "window": (10.0, 280.0),
    "k_integral_reported": 4.5968,  # L/mol/s
    "k_diff_reported": 3.6760,
    "n_reported": 0.987,
}

# Conclusion table of the batch report, L/mol/s.
BATCH_REPORTED_K = {
    30: {"integral": 0.3995, "differential": 0.3587},
    40: {"integral": 2.3375, "differential": 2.2387},
    50: {"integral": 4.5968, "differential": 3.6760},
}

# ---------------------------------------------------------------------------
# RTD, Group 21. TDS vs time, minutes. Background subtracted in the analysis.
# ---------------------------------------------------------------------------

def _pairs(start, step, values):
    return [(round(start + i * step, 4), v) for i, v in enumerate(values)]


_CSTR10 = [
    3, 209, 201, 196, 190, 184, 177, 170, 165, 158, 151, 147, 141, 136, 132, 128,
    122, 119, 115, 110, 106, 103, 99, 95, 92, 89, 85, 83, 80, 77, 74, 71, 69, 66,
    64, 62, 60, 58, 56, 54, 52, 50, 49, 47, 46, 45, 43, 42, 40, 39, 38, 37, 36, 34,
    32, 31, 30, 29, 28, 28, 27, 26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 17, 16,
    15, 14, 13, 13, 12, 11, 11, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9,
]
_CSTR15 = [
    2, 202, 196, 188, 177, 170, 160, 153, 145, 136, 129, 123, 117, 111, 105, 100,
    94, 89, 84, 79, 75, 72, 68, 64, 61, 58, 55, 53, 50, 46, 45, 43, 41, 39, 37, 35,
    34, 32, 31, 29, 27, 24, 22, 21, 20, 19, 18, 17, 15, 16, 15, 14, 14, 13, 13, 12,
    12, 11, 11, 10, 10, 9, 9, 9, 9,
]
_PB10 = [
    3, 8, 3, 3, 3, 3, 3, 2, 2, 4, 9, 31, 83, 199, 381, 595, 575, 513, 431, 360,
    280, 224, 173, 137, 101, 84, 64, 51, 41, 33, 27, 22, 18, 15, 12, 11, 10, 8, 8, 6,
]
_PB15 = [
    3, 2, 2, 2, 2, 2, 2, 5, 38, 165, 467, 696, 684, 648, 398, 273, 188, 122, 90,
    64, 42, 31, 23, 15, 11, 9, 6, 5,
]

RTD_RUNS = [
    {"name": "CSTR 10 LPH", "kind": "CSTR", "Q": 10, "V": 2.4, "bg": 3.0, "points": _pairs(0, 0.5, _CSTR10)},
    {"name": "CSTR 15 LPH", "kind": "CSTR", "Q": 15, "V": 2.4, "bg": 2.0, "points": _pairs(0, 0.5, _CSTR15)},
    {"name": "Packed bed 10 LPH", "kind": "Packed bed", "Q": 10, "V": 2.4, "bg": 3.0, "points": _pairs(0, 0.5, _PB10)},
    {"name": "Packed bed 15 LPH", "kind": "Packed bed", "Q": 15, "V": 2.4, "bg": 3.0, "points": _pairs(0, 0.5, _PB15)},
]

RTD_PUBLISHED = {
    "CSTR 10 LPH": {"tm": 11.833, "var": 97.492, "N_tau": 2.127, "tau": 14.4},
    "CSTR 15 LPH": {"tm": 8.313, "var": 48.827, "N_tau": 1.887, "tau": 9.6},
    "Packed bed 10 LPH": {"tm": 9.062, "var": 4.579, "Pe": 39.50},
    "Packed bed 15 LPH": {"tm": 6.441, "var": 1.856, "Pe": 48.40},
}

# NaCl calibration standards: TDS reading, concentration mg/L.
RTD_CALIBRATION = [
    (2, 0), (47, 40), (89, 80), (128, 120), (169, 160), (208, 200), (253, 240),
    (295, 280), (335, 320), (375, 360), (424, 400), (456, 440), (489, 480),
    (529, 520), (558, 560), (607, 600), (649, 640), (688, 680), (727, 720),
    (766, 760), (807, 800), (844, 840), (893, 880), (938, 920), (976, 960), (1000, 1000),
]

LITERATURE_EA = (40.0, 50.0)  # kJ/mol, the range quoted in the PFR report


def _cstr_frame(spec, timed=False):
    rows = []
    for run in spec["runs"]:
        if timed:
            T, qA, qB, lam0, vals, _ = run
        else:
            T, qA, qB, lam0, vals = run
        Q = qA + qB
        tau = 60.0 * spec["V"] / Q
        times = []
        if timed:
            times = [tau + i for i in range(len(vals))]
        rec = {
            "T_C": T,
            "qA_LPH": qA,
            "qB_LPH": qB,
            "lam0": lam0,
        }
        for i in range(4):
            rec[f"r{i+1}"] = vals[i] if i < len(vals) else np.nan
            rec[f"t{i+1}"] = times[i] if i < len(times) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def cstr_group8_frame():
    return _cstr_frame(CSTR_GROUP8, timed=False)


def cstr_timed_frame():
    return _cstr_frame(CSTR_TIMED, timed=True)


def pfr_group8_frame():
    rows = []
    for T, Q, lam0, vals in PFR_GROUP8["runs"]:
        tau = 60.0 * PFR_GROUP8["V"] / Q
        times = [tau, 8.0, 9.0, 10.0]
        rec = {"T_C": T, "Q_LPH": Q, "lam0": lam0}
        for i in range(4):
            rec[f"r{i+1}"] = vals[i]
            rec[f"t{i+1}"] = times[i]
        rows.append(rec)
    return pd.DataFrame(rows)


def batch_50_frame():
    return pd.DataFrame({"t_s": BATCH_50["t_s"], "ec_uS": BATCH_50["ec"]})


def rtd_frame(run_name=None):
    rows = []
    for run in RTD_RUNS:
        if run_name and run["name"] != run_name:
            continue
        for t, tds in run["points"]:
            rows.append({"run": run["name"], "kind": run["kind"], "t_min": t, "TDS": tds})
    return pd.DataFrame(rows)
