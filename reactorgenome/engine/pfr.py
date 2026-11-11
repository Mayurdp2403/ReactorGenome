"""Tubular reactor (the helical coil and the packed bed).

Plug flow, written in residence-time coordinate s = V'/Q:

    dc/ds = ν r(c, T)          (constant density, so dFA/dV = rA becomes this)
    dT/ds = (-ΔH) r / ρcp      (adiabatic option)

Other flow pictures
    dispersion  axial dispersion with Danckwerts boundaries, Peclet number Pe
    laminar     each streamline is a batch reactor; E(t) of a straight laminar tube
    recycle     part of the outlet goes back to the inlet (plug flow only)

Sample delay: the lab collects outlet liquid in a beaker and reads it later,
so the reaction keeps running for δ minutes as a batch. That shifts every
measured conversion upwards by the same time, which is why the free
intercept of τ against X/(1-X) comes out negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq

from .rate import NU, Kinetics, rate
from .units import EngineError, c_to_k, check_all, k_to_c

FLOWS = {"plug": "Plug flow", "dispersion": "Axial dispersion", "laminar": "Laminar, segregated"}


@dataclass(frozen=True)
class PFRModel:
    V: float  # L
    Q: float  # L/min total
    CA0: float  # mol/L at the inlet, after mixing
    CB0: float
    T_C: float
    kin: Kinetics = field(default_factory=lambda: Kinetics(k_ref=6.0))
    flow: str = "plug"
    Pe: float = 50.0
    recycle: float = 0.0  # R = recycled flow / fresh flow
    adiabatic: bool = False
    delay: float = 0.0  # min of batch reaction after the outlet

    @property
    def tau(self) -> float:
        return self.V / self.Q if self.Q > 0 else float("inf")

    @property
    def M(self) -> float:
        return self.CB0 / self.CA0 if self.CA0 > 0 else float("nan")

    def problems(self) -> list[str]:
        out = check_all([("V", self.V), ("Q", self.Q), ("C", self.CA0), ("C", self.CB0), ("T_C", self.T_C), ("k", self.kin.k_ref)])
        if self.CA0 <= 0:
            out.append("There is no NaOH at the inlet.")
        if self.flow not in FLOWS:
            out.append(f"Unknown flow model '{self.flow}'.")
        if self.flow == "dispersion" and not (self.Pe > 0):
            out.append("Peclet number must be above zero.")
        if self.recycle < 0:
            out.append("Recycle ratio cannot be negative.")
        if self.flow != "plug" and self.recycle > 0:
            out.append("Recycle is only built for plug flow here.")
        if self.flow != "plug" and self.adiabatic:
            out.append("The adiabatic option is only built for plug flow here.")
        if self.delay < 0:
            out.append("Sample delay cannot be negative.")
        return out


def _plug(c_in, T_in_K, tau, kin: Kinetics, adiabatic: bool, n: int = 121):
    def f(_s, y):
        T = y[4] if adiabatic else T_in_K
        r = rate(y[:4], kin.k(T), kin.k2_ratio)
        d = list(NU * r)
        d.append((-kin.dH) * r / kin.rho_cp if adiabatic else 0.0)
        return d

    s = np.linspace(0.0, tau, n)
    y0 = list(c_in) + [T_in_K]
    sol = solve_ivp(f, (0.0, tau), y0, t_eval=s, method="LSODA", rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise EngineError(f"Plug-flow integration failed: {sol.message}")
    return sol.t, sol.y


def batch_advance(c, T_K, kin: Kinetics, dt: float):
    """Let a sample keep reacting as a batch for dt minutes (isothermal)."""
    if dt <= 0:
        return np.asarray(c, dtype=float)
    _s, y = _plug(c, T_K, dt, kin, False, n=2)
    return y[:4, -1]


def dispersion_profile(c_in, k, K, tau, Pe, n=81):
    """Danckwerts closed vessel at constant T, any inlet composition.

    (1/Pe) CA'' - CA' - τ r(CA) = 0 on η ∈ [0, 1]
    CA(0) - CA'(0)/Pe = CA_in,  CA'(1) = 0
    CB - CA and CA + CC are constant along the bed, so CA is the only unknown.
    Returns η and the full concentration profile (4 × n).
    """
    ca_in, cb_in, cc_in, cd_in = (float(x) for x in c_in)
    d = cb_in - ca_in
    scale = max(ca_in, 1e-12)

    def r(ca):
        ca = np.clip(ca, 0.0, None)
        return k * (ca * np.clip(ca + d, 0.0, None) - K * np.clip(cc_in + ca_in - ca, 0, None) * np.clip(cd_in + ca_in - ca, 0, None))

    eta = np.linspace(0.0, 1.0, n)

    def fun(_e, y):
        return np.vstack([y[1], Pe * y[1] + Pe * tau * r(y[0] * scale) / scale])

    def bc(ya, yb):
        return np.array([ya[0] - ya[1] / Pe - ca_in / scale, yb[1]])

    _s, yp = _plug(c_in, 300.0, tau, Kinetics(k_ref=k, k2_ratio=K), False, n=n)
    u0 = yp[0] / scale
    y0 = np.vstack([u0, np.gradient(u0, eta)])
    sol = solve_bvp(fun, bc, eta, y0, tol=1e-6, max_nodes=20000)
    if not sol.success:
        raise EngineError(f"The dispersion model did not converge at Pe = {Pe:.3g}: {sol.message}")
    ca = np.clip(sol.sol(eta)[0] * scale, 0.0, None)
    prof = np.vstack([ca, ca + d, cc_in + ca_in - ca, cd_in + ca_in - ca])
    return eta, prof


def _dispersion(CA0, CB0, k, K, tau, Pe, n=81):
    eta, prof = dispersion_profile([CA0, CB0, 0.0, 0.0], k, K, tau, Pe, n)
    return eta, prof[0] / CA0


def _laminar_X(CA0, CB0, k, K, tau, n_w=400):
    """Cup-mixed conversion of a straight laminar tube with no diffusion.

    F(t) = 1 - τ²/(4t²) for t ≥ τ/2. With w = τ²/(4t²), X = ∫₀¹ Xb(τ/(2√w)) dw.
    """
    w = (np.arange(n_w) + 0.5) / n_w
    t = tau / (2.0 * np.sqrt(w))
    tmax = float(t.max())
    kin = Kinetics(k_ref=k, k2_ratio=K)
    _s, y = _plug([CA0, CB0, 0.0, 0.0], 300.0, tmax, kin, False, n=4000)
    CA = np.interp(t, _s, y[0])
    return float(np.mean(1.0 - CA / CA0))


def solve(m: PFRModel, n: int = 101) -> dict:
    """Outlet and axial profile. z runs 0 → 1 along the tube."""
    probs = m.problems()
    if probs:
        raise EngineError(" ".join(probs))
    kin = m.kin
    T_in = c_to_k(m.T_C)
    c_feed = np.array([m.CA0, m.CB0, 0.0, 0.0])
    tau = m.tau
    out = {"tau": tau, "flow": m.flow}
    if m.flow == "plug":
        if m.recycle > 0:
            R = m.recycle
            tau_p = m.V / ((1.0 + R) * m.Q)
            if not m.adiabatic:
                def g(ca_out):
                    d = m.CB0 - m.CA0
                    c_rec = np.array([ca_out, ca_out + d, m.CA0 - ca_out, m.CA0 - ca_out])
                    c_mix = (c_feed + R * c_rec) / (1.0 + R)
                    _s, y = _plug(c_mix, T_in, tau_p, kin, False, n=2)
                    return y[0, -1] - ca_out

                lo = max(0.0, m.CA0 - m.CB0) if m.CB0 < m.CA0 else 0.0
                ca = brentq(g, lo, m.CA0, xtol=1e-13) if g(lo) * g(m.CA0) < 0 else m.CA0
                d = m.CB0 - m.CA0
                c_rec = np.array([ca, ca + d, m.CA0 - ca, m.CA0 - ca])
                c_in, T_mix = (c_feed + R * c_rec) / (1.0 + R), T_in
            else:
                c_rec, T_rec = c_feed.copy(), T_in
                for _ in range(400):
                    c_in = (c_feed + R * c_rec) / (1.0 + R)
                    T_mix = (T_in + R * T_rec) / (1.0 + R)
                    _s, y = _plug(c_in, T_mix, tau_p, kin, True, n=2)
                    new_c, new_T = y[:4, -1], y[4, -1]
                    if np.max(np.abs(new_c - c_rec)) < 1e-11 and abs(new_T - T_rec) < 1e-8:
                        break
                    c_rec, T_rec = 0.5 * (c_rec + new_c), 0.5 * (T_rec + new_T)
                else:
                    raise EngineError("Adiabatic recycle did not converge.")
            s, y = _plug(c_in, T_mix, tau_p, kin, m.adiabatic, n=n)
            z = s / tau_p
        else:
            s, y = _plug(c_feed, T_in, tau, kin, m.adiabatic, n=n)
            z = s / tau
        CA, CC, T = y[0], y[2], y[4]
        c_out = y[:4, -1]
        T_out = y[4, -1]
    elif m.flow == "dispersion":
        k = kin.k(T_in)
        z, u = _dispersion(m.CA0, m.CB0, k, kin.k2_ratio, tau, m.Pe, n=n)
        CA = u * m.CA0
        CC = m.CA0 - CA
        T = np.full_like(z, T_in)
        ca = CA[-1]
        c_out = np.array([ca, ca + m.CB0 - m.CA0, m.CA0 - ca, m.CA0 - ca])
        T_out = T_in
    else:
        k = kin.k(T_in)
        z = np.linspace(0.0, 1.0, max(n // 4, 21))
        X = np.array([0.0] + [_laminar_X(m.CA0, m.CB0, k, kin.k2_ratio, tau * zz, n_w=200) for zz in z[1:]])
        CA = m.CA0 * (1.0 - X)
        CC = m.CA0 - CA
        T = np.full_like(z, T_in)
        ca = m.CA0 * (1.0 - _laminar_X(m.CA0, m.CB0, k, kin.k2_ratio, tau))
        c_out = np.array([ca, ca + m.CB0 - m.CA0, m.CA0 - ca, m.CA0 - ca])
        T_out = T_in
    X_out = 1.0 - c_out[0] / m.CA0
    c_meas = batch_advance(c_out, T_out, kin, m.delay) if m.delay > 0 else c_out
    out.update(
        z=z, CA=CA, CC=CC, X=1.0 - CA / m.CA0, T_C=k_to_c(np.asarray(T)), c_out=c_out,
        X_out=float(X_out), T_out_C=float(k_to_c(T_out)), X_measured=float(1.0 - c_meas[0] / m.CA0),
    )
    return out


def ideal_X(Da: float, M: float = 1.0) -> float:
    """Analytic plug-flow conversion for -rA = k CA CB, Da = k CA0 τ."""
    if Da <= 0:
        return 0.0
    if abs(M - 1.0) < 1e-12:
        return float(Da / (1.0 + Da))
    E = np.exp((M - 1.0) * Da)
    return float(M * (E - 1.0) / (M * E - 1.0))
