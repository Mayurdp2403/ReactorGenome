"""Conductivity to conversion.

The lab manuals use one straight line,

    X = (Λ0 - Λ) / (Λ0 - Λ∞)

and then pick Λ∞ by hand. The CSTR report uses 3.75 mS/cm. The PFR report uses
0.1 mS/cm. Those cannot both be right for the same reaction.

NaOH is the strong conductor. Ethyl acetate and ethanol barely conduct. Sodium
acetate does, but the acetate ion is much less mobile than hydroxide. So the
reading falls as hydroxide is replaced by acetate, and it falls in a straight
line with conversion:

    Λ(X) = Λ0 + (Λ∞ - Λ0) X

Λ0 is measured (NaOH at the mixed-feed concentration). Λ∞ is the conductivity
of the same sodium, now paired with acetate instead of hydroxide. With the
infinite-dilution ion conductivities that ratio is fixed, so

    Λ∞ = Λ0 * (λ_Na + λ_OAc) / (λ_Na + λ_OH)

The user can still type the manual's Λ∞. The two answers sit next to each other.
"""

from __future__ import annotations

import numpy as np

# Infinite-dilution molar ionic conductivities at 25 °C, S cm^2 mol^-1.
# κ (mS/cm) ≈ λ * C (mol/L) for a single salt at high dilution.
LAM_NA = 50.11
LAM_OH = 198.6
LAM_OAC = 40.9


def ion_ratio(lam_na: float = LAM_NA, lam_oh: float = LAM_OH, lam_oac: float = LAM_OAC) -> float:
    return (lam_na + lam_oac) / (lam_na + lam_oh)


def lambda_inf_ion(lambda0: float, lam_na: float = LAM_NA, lam_oh: float = LAM_OH, lam_oac: float = LAM_OAC) -> float:
    """Estimated conductivity at full conversion, anchored to a measured Λ0."""
    return float(lambda0) * ion_ratio(lam_na, lam_oh, lam_oac)


def conversion(lam, lam0, lam_inf):
    """Fractional conversion of the limiting reactant from a conductivity reading.

    Works for an array or a single number. NaN if the denominator vanishes.
    """
    lam = np.asarray(lam, dtype=float)
    den = float(lam0) - float(lam_inf)
    out = (float(lam0) - lam) / den if abs(den) > 1e-12 else np.full(np.shape(lam), np.nan)
    return out


def lambda_from_conversion(X, lam0, lam_inf):
    X = np.asarray(X, dtype=float)
    return float(lam0) + (float(lam_inf) - float(lam0)) * X


def hydroxide_left(lam, lam0, lam_inf, c_b0: float):
    """Unreacted NaOH concentration, mol/L, from the same straight line."""
    X = conversion(lam, lam0, lam_inf)
    return np.asarray(c_b0, dtype=float) * (1.0 - X)
