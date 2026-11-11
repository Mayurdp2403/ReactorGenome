"""What the lab cannot run: trains of reactors, recycle, Levenspiel, heat."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

from reactorgenome.kinetics import (
    R_GAS,
    cstr_X,
    k_arrhenius,
    pfr_X,
    pfr_integrate,
    tank_outlet,
    tanks_in_series,
)

RHO = 1000.0  # g/L
CP = 4.184  # J/g/K
RHO_CP = RHO * CP  # J/L/K


def k_of(stage, T_fallback):
    if "k" in stage and stage["k"] is not None:
        return float(stage["k"])
    T = stage.get("T", T_fallback)
    return float(k_arrhenius(T, stage["A"], stage["Ea"]))


def run_train(stages, CA0, CB0, v0, k2: float = 0.0):
    """stages: list of {kind: 'cstr'|'pfr', V, T, k or A+Ea}.

    v0 in L/min, V in L, k in L/mol/min.
    """
    CA, CB, CC, CD = float(CA0), float(CB0), 0.0, 0.0
    rows = []
    X_prev = 0.0
    for i, s in enumerate(stages):
        tau = float(s["V"]) / float(v0) if v0 > 0 else np.nan
        k = k_of(s, s.get("T", 30.0))
        kind = s["kind"]
        if kind == "cstr":
            CA, CB, CC, CD = tank_outlet(CA, CB, k, tau, CC, CD, k2=k2)
            prof_t = np.array([0.0, tau])
            prof_CA = np.array([rows[-1]["CA"] if rows else CA0, CA])
        else:
            CA_out, CB_out, prof_t, prof_CA, prof_CB = pfr_integrate(
                CA, CB, k, tau, n_steps=120, k2=k2, CC_in=CC, CD_in=CD
            )
            # products along the PFR are not returned; rebuild from extent
            extent = (CA - CA_out)
            CC = CC + extent
            CD = CD + extent
            CA, CB = CA_out, CB_out
        X = (CA0 - CA) / CA0 if CA0 else np.nan
        rows.append(
            {
                "i": i + 1,
                "kind": kind,
                "V": float(s["V"]),
                "T": float(s.get("T", np.nan)),
                "tau": tau,
                "k": k,
                "CA": CA,
                "CB": CB,
                "X_in": X_prev,
                "X": X,
                "t_prof": prof_t if kind == "pfr" else None,
                "CA_prof": prof_CA if kind == "pfr" else None,
            }
        )
        X_prev = X
    return rows


def cstr_recycle_X(k, CA0, CB0, tau, R):
    """Recycle leaves a single ideal CSTR unchanged. Computed both ways so the plot can show it."""
    CA, CB, _, _ = tank_outlet(CA0, CB0, k, tau)
    direct = (CA0 - CA) / CA0
    # The long way: inlet is mixed with recycle, space time inside is tau/(1+R).
    R = float(max(R, 0.0))

    def mismatch(CA_out):
        CB_out = CA_out + (CB0 - CA0)
        CAf = (CA0 + R * CA_out) / (1.0 + R)
        CBf = (CB0 + R * CB_out) / (1.0 + R)
        tau_in = tau / (1.0 + R)
        CAo, _, _, _ = tank_outlet(CAf, CBf, k, tau_in)
        return CAo - CA_out

    try:
        CA_out = brentq(mismatch, 1e-9, max(CA0, CB0) * 0.999)
        via = (CA0 - CA_out) / CA0
    except ValueError:
        via = direct
    return direct, via


def pfr_recycle_X(k, CA0, CB0, tau, R, k2: float = 0.0):
    R = float(max(R, 0.0))
    if R < 1e-9:
        CA, CB, _, _, _ = pfr_integrate(CA0, CB0, k, tau, k2=k2)
        return (CA0 - CA) / CA0

    def mismatch(CA_out):
        CB_out = max(CA_out + (CB0 - CA0), 0.0)
        CAf = (CA0 + R * CA_out) / (1.0 + R)
        CBf = (CB0 + R * max(CB_out, 0.0)) / (1.0 + R)
        CAo, _, _, _, _ = pfr_integrate(CAf, CBf, k, tau / (1.0 + R), k2=k2)
        return CAo - CA_out

    CA_out = brentq(mismatch, 1e-8, max(CA0, 1e-6) * 0.9999)
    return float((CA0 - CA_out) / CA0)


def rate_inv(X, k, CA0, M):
    """1/(-rA) for A+B, irreversible, second order."""
    X = np.asarray(X, dtype=float)
    CA = CA0 * (1.0 - X)
    CB = CA0 * (M - X)
    rate = np.maximum(float(k) * CA * CB, 1e-15)
    return 1.0 / rate


def volumes_for_target(X, k, CA0, M, v0):
    """Volume of one CSTR, a PFR, and two CSTRs, all reaching the same X.

    v0 in L/min. Returns litres.
    """
    X = float(min(X, min(1.0, M) - 1e-4))
    FA0 = CA0 * v0
    r_exit = 1.0 / rate_inv(X, k, CA0, M)
    V_cstr = FA0 * X / r_exit
    xs = np.linspace(1e-4, X, 300)
    V_pfr = FA0 * float(trapezoid(rate_inv(xs, k, CA0, M), xs))

    def vtot(x1):
        r1 = 1.0 / float(rate_inv(x1, k, CA0, M))
        r2 = r_exit
        return FA0 * (x1 / r1 + (X - x1) / r2)

    grid = np.linspace(0.02 * X, 0.98 * X, 100)
    vals = np.array([vtot(g) for g in grid])
    x1 = float(grid[int(np.argmin(vals))])
    V_two = float(vtot(x1))
    return {"X": X, "V_cstr": float(V_cstr), "V_pfr": float(V_pfr), "V_two": V_two, "X_mid": x1, "FA0": FA0}


def heat_curves(T_grid, A, Ea, CA0, M, V, v0, dH, UA, Ta, T0):
    """Qg(T) and Qr(T) for a single CSTR, J/min.

    dH is the heat of reaction, J/mol, negative if exothermic.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    tau = float(V) / float(v0) if v0 > 0 else np.nan
    X = np.zeros_like(T_grid)
    Qg = np.zeros_like(T_grid)
    Qr = np.zeros_like(T_grid)
    for i, T in enumerate(T_grid):
        k = float(k_arrhenius(T, A, Ea))
        Da = k * CA0 * tau
        Xi = float(cstr_X(Da, M))
        CA = CA0 * (1.0 - Xi)
        CB = CA0 * (M - Xi)
        rate = k * max(CA, 0.0) * max(CB, 0.0)
        X[i] = Xi
        Qg[i] = (-float(dH)) * rate * float(V)
        Qr[i] = float(UA) * (T - float(Ta)) + RHO_CP * float(v0) * (T - float(T0))
    return X, Qg, Qr


def steady_states(T_grid, Qg, Qr):
    """Crossings of Qg and Qr. Stable when removal steepens faster than generation."""
    diff = np.asarray(Qg) - np.asarray(Qr)
    T_grid = np.asarray(T_grid, dtype=float)
    found = []
    for i in range(len(diff) - 1):
        if diff[i] == 0 or diff[i] * diff[i + 1] < 0:
            w = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]) + 1e-15)
            T = float(T_grid[i] * (1 - w) + T_grid[i + 1] * w)
            # slopes
            dG = (Qg[i + 1] - Qg[i]) / (T_grid[i + 1] - T_grid[i])
            dR = (Qr[i + 1] - Qr[i]) / (T_grid[i + 1] - T_grid[i])
            stable = dR > dG
            found.append({"T": T, "stable": bool(stable)})
    # drop near-duplicates
    cleaned = []
    for f in found:
        if not cleaned or abs(f["T"] - cleaned[-1]["T"]) > 0.4:
            cleaned.append(f)
    return cleaned


def trajectory(A, Ea, CA0, M, V, v0, dH, UA, Ta, T0, T_init, CA_init, t_end, n: int = 400):
    """Startup of a non-isothermal CSTR. Returns t, CA, T, X."""
    tau = float(V) / float(v0)
    CB0 = M * CA0
    CB_init = CA_init + (CB0 - CA0)

    def f(_t, y):
        CA, CB, T = y
        k = A * np.exp(-Ea / (R_GAS * (T + 273.15)))
        rate = k * max(CA, 0.0) * max(CB, 0.0)
        dCA = (CA0 - CA) / tau - rate
        dCB = (CB0 - CB) / tau - rate
        dT = (T0 - T) / tau + ((-dH) / RHO_CP) * rate - (UA / (RHO_CP * V)) * (T - Ta)
        return [dCA, dCB, dT]

    t_eval = np.linspace(0.0, float(t_end), n)
    sol = solve_ivp(
        f,
        (0.0, float(t_end)),
        [float(CA_init), float(CB_init), float(T_init)],
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-8,
        method="BDF",
        max_step=float(t_end) / 100.0,
    )
    if not sol.success:
        return t_eval, np.full_like(t_eval, CA_init), np.full_like(t_eval, T_init)
    X = (CA0 - sol.y[0]) / CA0
    return sol.t, sol.y[0], sol.y[2], X


def adiabatic_rise(dH, CA0):
    """ΔT if every mole reacted and no heat left the liquid, °C."""
    return float((-dH) * CA0 / RHO_CP)
