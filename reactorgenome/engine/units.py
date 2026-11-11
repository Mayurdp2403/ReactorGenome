"""Unit conversions and input checks.

The lab reads flow in L/h (LPH), time in seconds or minutes, conductivity in
mS/cm and temperature in °C. The engine works in L, L/min, min, mol/L and K.
Conversions happen once, here, at the edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_GAS = 8.314462618  # J/mol/K
RHO_CP_WATER = 4184.0  # J/L/K, dilute aqueous solution
T_ABS = 273.15


def lph_to_lpm(q):
    return np.asarray(q, dtype=float) / 60.0 if np.ndim(q) else float(q) / 60.0


def lpm_to_lph(q):
    return np.asarray(q, dtype=float) * 60.0 if np.ndim(q) else float(q) * 60.0


def s_to_min(t):
    return np.asarray(t, dtype=float) / 60.0 if np.ndim(t) else float(t) / 60.0


def min_to_s(t):
    return np.asarray(t, dtype=float) * 60.0 if np.ndim(t) else float(t) * 60.0


def c_to_k(T_C):
    return np.asarray(T_C, dtype=float) + T_ABS if np.ndim(T_C) else float(T_C) + T_ABS


def k_to_c(T_K):
    return np.asarray(T_K, dtype=float) - T_ABS if np.ndim(T_K) else float(T_K) - T_ABS


def k_per_s_to_per_min(k):
    """Second-order k in L/mol/s to L/mol/min."""
    return float(k) * 60.0


def k_per_min_to_per_s(k):
    return float(k) / 60.0


def kj_to_j(E):
    return float(E) * 1000.0


def arrhenius_k(k_ref: float, T_ref_C: float, Ea_J: float, T_C):
    """k(T) = k_ref exp(-Ea/R (1/T - 1/T_ref)). Works for scalars and arrays."""
    T = c_to_k(T_C)
    Tr = c_to_k(T_ref_C)
    return k_ref * np.exp(-float(Ea_J) / R_GAS * (1.0 / T - 1.0 / Tr))


@dataclass(frozen=True)
class Bound:
    label: str
    unit: str
    lo: float
    hi: float
    lo_open: bool = False  # True means the value must be strictly above lo


# One table of sane ranges. The UI widgets use the same numbers so a value the
# engine rejects cannot be typed in the first place.
BOUNDS: dict[str, Bound] = {
    "V": Bound("Reactor volume", "L", 0.0, 1000.0, lo_open=True),
    "Q": Bound("Total flow", "L/min", 0.0, 1000.0, lo_open=True),
    "q": Bound("Feed flow", "L/min", 0.0, 1000.0),
    "C": Bound("Concentration", "mol/L", 0.0, 20.0),
    "T_C": Bound("Temperature", "°C", -10.0, 300.0),
    "k": Bound("Rate constant", "L/mol/min", 0.0, 1e9),
    "Ea": Bound("Activation energy", "J/mol", 0.0, 400e3),
    "frac": Bound("Fraction", "–", 0.0, 1.0),
    "bypass": Bound("Bypass fraction", "–", 0.0, 0.95),
    "dead": Bound("Stagnant fraction", "–", 0.0, 0.95),
    "N": Bound("Number of tanks", "–", 1.0, 50.0),
    "Pe": Bound("Peclet number", "–", 0.0, 1e5, lo_open=True),
    "UA": Bound("UA", "J/min/K", 0.0, 1e9),
    "dH": Bound("Heat of reaction", "J/mol", -1e6, 1e6),
    "time": Bound("Time", "min", 0.0, 1e6),
}


def check(name: str, value, key: str | None = None) -> str | None:
    """Return a plain-language problem, or None if the value is usable."""
    b = BOUNDS[key or name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{b.label} is not a number."
    if not np.isfinite(v):
        return f"{b.label} is missing."
    if b.lo_open and v <= b.lo:
        return f"{b.label} must be above {b.lo:g} {b.unit}."
    if not b.lo_open and v < b.lo:
        return f"{b.label} cannot be below {b.lo:g} {b.unit}."
    if v > b.hi:
        return f"{b.label} above {b.hi:g} {b.unit} is outside what this model was built for."
    return None


def check_all(pairs) -> list[str]:
    """pairs: iterable of (bound key, value). Returns every problem found."""
    out = []
    for key, value in pairs:
        msg = check(key, value)
        if msg:
            out.append(msg)
    return out


def safe_log(x, floor: float = 1e-300):
    """log that never sees zero or a negative number."""
    return np.log(np.clip(np.asarray(x, dtype=float), floor, None))


class EngineError(RuntimeError):
    """A model could not produce a trustworthy answer. The message says why."""
