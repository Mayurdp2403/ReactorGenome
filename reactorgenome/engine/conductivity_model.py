"""Which Λ∞ to use.

'ion'   Λ∞ = Λ0 (λNa + λOAc)/(λNa + λOH) ≈ 0.366 Λ0, from limiting ionic
        conductivities at 25 °C. Anchored to the measured Λ0, so it moves
        with temperature and concentration the way Λ0 does.
'fixed' a typed constant, as in the lab manuals.

The two manuals for this reaction use different constants (3.75 mS/cm for the
CSTR, 0.1 mS/cm for the PFR). That is recorded in docs/VALIDATION.md; the app
lets the user pick and shows which one was used.
"""

from __future__ import annotations

from ..conductivity import ion_ratio


def lam_inf_of(lam0: float, mode: str, fixed: float) -> float:
    if mode == "ion":
        return float(lam0) * ion_ratio()
    return float(fixed)
