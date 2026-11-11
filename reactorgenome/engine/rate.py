"""Rate law shared by every reactor model.

NaOH (A) + CH3COOC2H5 (B) -> CH3COONa (C) + C2H5OH (D)

    -rA = k(T) [CA CB - (k2/k1) CC CD]

k(T) follows Arrhenius from a reference value. Setting Ea = 0 makes k
independent of temperature. Setting k2/k1 = 0 gives the irreversible law the
lab manuals use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import R_GAS, RHO_CP_WATER, c_to_k

NU = np.array([-1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Kinetics:
    k_ref: float  # L/mol/min at T_ref
    T_ref_C: float = 30.0
    Ea: float = 0.0  # J/mol
    k2_ratio: float = 0.0  # reverse / forward rate constant, same T
    dH: float = -75.0e3  # J/mol of A reacted. Model input, not a measured value.
    rho_cp: float = RHO_CP_WATER  # J/L/K

    def k(self, T_K):
        """Forward k at absolute temperature T_K (scalar or array)."""
        Tr = c_to_k(self.T_ref_C)
        T = np.asarray(T_K, dtype=float)
        val = self.k_ref * np.exp(-self.Ea / R_GAS * (1.0 / T - 1.0 / Tr))
        return float(val) if val.ndim == 0 else val

    def k_C(self, T_C):
        return self.k(c_to_k(T_C))


def rate(c, k: float, k2_ratio: float):
    """Net forward rate of reaction, mol/L/min. c = [CA, CB, CC, CD].

    Negative concentrations from solver overshoot are treated as zero so the
    rate cannot run away.
    """
    cA, cB, cC, cD = (max(float(x), 0.0) for x in c[:4])
    return k * (cA * cB - k2_ratio * cC * cD)


def rate_vec(cA, cB, cC, cD, k, k2_ratio):
    cA, cB, cC, cD = (np.clip(np.asarray(x, dtype=float), 0.0, None) for x in (cA, cB, cC, cD))
    return k * (cA * cB - k2_ratio * cC * cD)


def equilibrium_conversion(CA0: float, CB0: float, k2_ratio: float) -> float:
    """Conversion of A where the net rate is zero, starting from no products."""
    if k2_ratio <= 0 or CA0 <= 0:
        lim = min(CA0, CB0) / CA0 if CA0 > 0 else 0.0
        return float(lim)
    # CA0 CB0 (1-X)(M-X) = K2 CA0^2 X^2 with M = CB0/CA0
    M = CB0 / CA0
    a = 1.0 - k2_ratio
    b = -(1.0 + M)
    c = M
    if abs(a) < 1e-12:
        return float(c / -b)
    disc = b * b - 4 * a * c
    roots = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    ok = [r for r in roots if -1e-12 <= r <= min(1.0, M) + 1e-12]
    return float(min(ok)) if ok else float("nan")
