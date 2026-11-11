"""Residence time distribution, tanks-in-series, dispersion, micromixing."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp, trapezoid
from scipy.optimize import brentq


def calibrate_tds(tds, slope=1.71175e-5, intercept=-1.68852e-4):
    """NaCl mol/L from the Group 21 calibration line."""
    return slope * np.asarray(tds, dtype=float) + intercept


def moments(t, signal, background=0.0):
    """Trapezoidal E(t), mean and variance.

    The signal may be TDS or concentration. A constant background is removed.
    E(t) does not care about a positive scale factor, so TDS minus background
    is enough and matches the RTD report.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(signal, dtype=float) - float(background)
    c = np.clip(c, 0.0, None)
    order = np.argsort(t)
    t, c = t[order], c[order]
    # drop duplicate times
    if len(t) >= 2:
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, c = t[keep], c[keep]
    if len(t) < 3:
        return None
    area = float(trapezoid(c, t))
    if area <= 0:
        return None
    E = c / area
    tm = float(trapezoid(t * E, t))
    var = float(trapezoid((t - tm) ** 2 * E, t))
    var = max(var, 0.0)
    return {
        "t": t,
        "C": c,
        "E": E,
        "area": area,
        "tm": tm,
        "var": var,
        "var_over_tm2": var / tm**2 if tm > 0 else np.nan,
    }


def tanks_N(tm, var, tau=None):
    """N = t_ref^2 / σ^2. The RTD itself uses tm. The report also quotes τ^2/σ^2."""
    out = {}
    if var and var > 0 and tm and tm > 0:
        out["N_from_tm"] = float(tm**2 / var)
    if var and var > 0 and tau and tau > 0:
        out["N_from_tau"] = float(tau**2 / var)
    return out


def peclet_open(var_over_tm2: float):
    """Open-open dispersion: σ^2/tm^2 = 2/Pe + 8/Pe^2.

    Pe = [1 + sqrt(1 + 8 v)] / v, with v = σ^2/tm^2.
    """
    v = float(var_over_tm2)
    if v <= 1e-12:
        return np.inf
    return float((1.0 + np.sqrt(1.0 + 8.0 * v)) / v)


def peclet_closed(var_over_tm2: float):
    """Closed-closed: σ^2/τ^2 = 2/Pe - 2(1-e^{-Pe})/Pe^2. Solved numerically.

    The report's packed-bed numbers use the open-open formula. This one is
    here so the two boundary conditions can be compared.
    """
    v = float(var_over_tm2)
    if v <= 1e-8:
        return np.inf
    if v >= 1.0:
        return np.nan
    lo, hi = 1e-6, max(2.0 / v * 5.0, 10.0)

    def f(Pe):
        return 2.0 / Pe - 2.0 * (1.0 - np.exp(-Pe)) / Pe**2 - v

    # f decreases from +inf to 0
    if f(hi) > 0:
        return hi
    # bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def ideal_cstr_E(t, tau):
    t = np.asarray(t, dtype=float)
    tau = float(tau)
    return np.exp(-t / tau) / tau


def tanks_E(t, tau, N: int):
    """E(t) for N equal tanks. N = 1 is the ideal CSTR."""
    from scipy.special import gamma as gamma_fn

    t = np.asarray(t, dtype=float)
    N = int(N)
    tau_i = float(tau) / N
    # t^{N-1} exp(-t/τi) / ((N-1)! τi^N)
    coef = 1.0 / (gamma_fn(N) * tau_i**N)
    tt = np.clip(t, 0, None)
    return coef * tt ** (N - 1) * np.exp(-tt / tau_i)


def segregated_X(t, E, k, CA0, M: float = 1.0):
    """Danckwerts complete segregation: each fluid element is a batch reactor."""
    from reactorgenome.kinetics import batch_X_of_time

    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    Xb = batch_X_of_time(t, k, CA0, M)
    area = float(trapezoid(E, t))
    if area <= 0:
        return np.nan
    return float(trapezoid(Xb * E, t) / area)


def maximum_mixedness_X(t, E, k, CA0, M: float = 1.0):
    """Zwietering maximum mixedness, integrated from the tail back to λ = 0.

    For an ideal CSTR this lands on the ordinary CSTR conversion. For order
    above 1 it is lower than segregation.
    """
    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    order = np.argsort(t)
    t, E = t[order], E[order]
    # cumulative F and survival
    dt = np.diff(t, prepend=t[0])
    # trapezoid cumulative
    F = np.zeros_like(t)
    for i in range(1, len(t)):
        F[i] = F[i - 1] + 0.5 * (E[i] + E[i - 1]) * (t[i] - t[i - 1])
    if F[-1] > 0:
        F = F / F[-1]
        E = E / max(trapezoid(E, t), 1e-15)
    # survival S = 1-F. Start where S is still measurable.
    S = np.clip(1.0 - F, 0.0, None)
    # interpolate E and S
    # integrate dC/dλ = -k C (M*CA0 - (CA0-C)) + (C-CA0) E/S   if CB = C + (CB0-CA0)
    # For equimolar, CB=C, dC/dλ = -k C^2 + (C - CA0) E/S
    # We integrate λ from t_max down to 0, so the ODE is in backward time.
    CB0 = M * CA0

    idx = np.where(S > 0.02)[0]
    lam0 = float(t[idx[-1]]) if len(idx) else float(t[int(0.8 * (len(t) - 1))])
    e0 = float(np.interp(lam0, t, E))
    s0 = max(float(np.interp(lam0, t, S)), 1e-4)
    ratio = e0 / s0

    def balance(C):
        CB = max(C + (CB0 - CA0), 0.0)
        return -float(k) * C * CB + (CA0 - C) * ratio

    if balance(0.0) * balance(CA0 * 0.999) < 0:
        C = brentq(balance, 0.0, CA0 * 0.999)
    else:
        C = 0.5 * CA0
    # March from the tail back to the exit on a fine grid. An explicit step
    # stays well behaved here; the stiff solver does not.
    lams = np.linspace(lam0, 0.0, 900)
    for i in range(len(lams) - 1):
        dl = lams[i + 1] - lams[i]
        mid = 0.5 * (lams[i] + lams[i + 1])
        e = float(np.interp(mid, t, E))
        s = max(float(np.interp(mid, t, S)), 1e-4)
        CB = max(C + (CB0 - CA0), 0.0)
        dC = -float(k) * C * CB + (CA0 - C) * e / s
        C = float(np.clip(C + dC * dl, 0.0, CA0))
    return float((CA0 - C) / CA0)


def dispersion_outlet(Pe: float, Da: float, n_mesh: int = 60):
    """Second-order equimolar axial dispersion with Danckwerts boundaries.

    Da = k CA0 τ, Pe = uL/D. Returns conversion at the exit.
    As Pe grows this goes to the PFR. As Pe falls it goes toward the CSTR.
    """
    Pe = float(Pe)
    Da = float(max(Da, 0.0))
    if Pe < 0.3:
        from reactorgenome.kinetics import cstr_X

        return float(cstr_X(Da)), None, None
    if Pe > 250:
        from reactorgenome.kinetics import pfr_X

        return float(pfr_X(Da)), None, None

    eta = np.linspace(0.0, 1.0, n_mesh)

    def fun(_eta, y):
        c, dc = y
        c_pos = np.maximum(c, 0.0)
        return np.vstack([dc, Pe * dc + Pe * Da * c_pos * c_pos])

    def bc(ya, yb):
        return np.array([ya[1] - Pe * (ya[0] - 1.0), yb[1]])

    # PFR-like initial guess
    from reactorgenome.kinetics import pfr_X

    X_pfr_local = []
    for z in eta:
        X_pfr_local.append(pfr_X(Da * z))
    c_guess = 1.0 - np.array(X_pfr_local)
    dc_guess = np.gradient(c_guess, eta)
    y0 = np.vstack([c_guess, dc_guess])
    sol = solve_bvp(fun, bc, eta, y0, tol=1e-4, max_nodes=2000)
    if not sol.success:
        # tanks stand-in, N ≈ Pe/2
        from reactorgenome.kinetics import tanks_in_series

        N = max(int(round(Pe / 2.0)), 1)
        CA, _, _ = tanks_in_series(1.0, 1.0, 1.0, Da, N)
        return float(1.0 - CA), None, None
    c = sol.y[0]
    X = float(np.clip(1.0 - c[-1], 0.0, 1.0))
    return X, sol.x, c


def laminar_pipe_E(t, tau, n: int = 400):
    """RTD of laminar flow in a straight tube. E = 0 for t < τ/2, else τ^2/(2 t^3)."""
    t = np.asarray(t, dtype=float)
    tau = float(tau)
    E = np.zeros_like(t)
    mask = t > tau / 2.0
    E[mask] = tau**2 / (2.0 * t[mask] ** 3)
    return E


def water_mu(T_C: float) -> float:
    """Dynamic viscosity of water, Pa·s, from a short table."""
    pts = np.array([20, 30, 40, 50, 60, 70, 80], dtype=float)
    mu = np.array([1.00, 0.80, 0.65, 0.55, 0.47, 0.40, 0.36], dtype=float) * 1e-3
    return float(np.interp(T_C, pts, mu))


def coil_flow(Q_Lph, V_L, d_mm, D_coil_mm, T_C):
    """Reynolds and Dean numbers for a helical coil. Geometry is an estimate unless measured."""
    d = float(d_mm) / 1000.0
    Dc = float(D_coil_mm) / 1000.0
    area = np.pi * d**2 / 4.0
    Q = float(Q_Lph) / 1000.0 / 3600.0  # m^3/s
    u = Q / area if area > 0 else np.nan
    rho = 1000.0 * (1.0 - max(T_C - 20.0, 0.0) * 0.0004)
    mu = water_mu(T_C)
    Re = rho * u * d / mu
    De = Re * np.sqrt(d / Dc) if Dc > 0 else np.nan
    L = (float(V_L) / 1000.0) / area if area > 0 else np.nan
    return {"u": u, "Re": float(Re), "De": float(De), "L_m": float(L), "rho": rho, "mu": mu}
