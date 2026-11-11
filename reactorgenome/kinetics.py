"""Algebra for second-order saponification, A + B → products.

Equimolar irreversible CSTR:  X / (1-X)^2 = k CA0 τ
Equimolar irreversible PFR:   X / (1-X)    = k CA0 τ
Batch, equimolar:             1/C = 1/C0 + k t

M = CB0/CA0. M = 1 is the lab. The same functions cover M ≠ 1.
Keq = k1/k2. Keq = inf is the irreversible lab assumption.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

R_GAS = 8.314462618  # J mol^-1 K^-1


def _as_positive(Da):
    Da = np.asarray(Da, dtype=float)
    return Da


def cstr_X(Da, M: float = 1.0):
    """Steady conversion of an ideal isothermal CSTR.

    Da = k * CA0 * τ.  M = CB0 / CA0.
    Returns a value in [0, min(1, M)).
    """
    Da_in = np.asarray(Da, dtype=float)
    scalar = Da_in.ndim == 0
    Da_arr = np.atleast_1d(Da_in).astype(float)
    X = np.zeros_like(Da_arr)
    M = float(M)
    limit = min(1.0, M) - 1e-12
    for i, a in enumerate(Da_arr):
        if not np.isfinite(a) or a <= 1e-14:
            X[i] = 0.0
            continue
        # a (1-X)(M-X) - X = 0
        b = a * (M + 1.0) + 1.0
        disc = b * b - 4.0 * a * a * M
        if disc < 0:
            X[i] = limit
            continue
        root = (b - np.sqrt(disc)) / (2.0 * a)
        # The physical root of the quadratic written in X is the minus branch
        # only when we expand correctly. Check the residual and pick the root
        # that lands inside the interval.
        alt = (b + np.sqrt(disc)) / (2.0 * a)
        cands = []
        for r in (root, alt):
            if 0.0 <= r < min(1.0, M) + 1e-9:
                cands.append(r)
        if not cands:
            # solve by bisection, always safe
            def f(x, a=a):
                return a * (1.0 - x) * (M - x) - x

            X[i] = brentq(f, 0.0, limit)
        else:
            X[i] = min(cands)
    return float(X[0]) if scalar else X


def pfr_X(Da, M: float = 1.0):
    """Steady conversion of an ideal isothermal PFR. Da = k CA0 τ."""
    Da_in = np.asarray(Da, dtype=float)
    scalar = Da_in.ndim == 0
    a = np.atleast_1d(Da_in).astype(float)
    M = float(M)
    X = np.zeros_like(a)
    for i, da in enumerate(a):
        if not np.isfinite(da) or da <= 1e-14:
            X[i] = 0.0
            continue
        if abs(M - 1.0) < 1e-10:
            X[i] = da / (1.0 + da)
        else:
            # da = ln[(M - X) / (M (1 - X))] / (M - 1)  →  X = M (E - 1) / (M E - 1), E = exp((M - 1) da)
            E = np.exp((M - 1.0) * da)
            X[i] = min(M * (E - 1.0) / (M * E - 1.0), min(1.0, M) - 1e-12)
    return float(X[0]) if scalar else X


def batch_X_of_time(t, k, CA0, M: float = 1.0):
    """Conversion in a constant-volume batch reactor. t and k in consistent units."""
    t = np.asarray(t, dtype=float)
    k = float(k)
    CA0 = float(CA0)
    M = float(M)
    kt = k * t
    if abs(M - 1.0) < 1e-10:
        Da = kt * CA0
        return Da / (1.0 + Da)
    # kt = 1/(CB0-CA0) ln[ (CB/CA) / (CB0/CA0) ]
    # CB0 - CA0 = CA0 (M-1)
    delta = CA0 * (M - 1.0)
    # Solve per point
    scalar = t.ndim == 0
    tt = np.atleast_1d(t)
    X = np.zeros_like(tt, dtype=float)
    hi = min(1.0, M) - 1e-12
    for i, ti in enumerate(tt):
        target = k * ti

        def f(x, target=target):
            CA = CA0 * (1.0 - x)
            CB = CA0 * (M - x)
            return target - (1.0 / delta) * np.log((CB / CA) / M)

        if ti <= 0:
            X[i] = 0.0
        elif f(hi) < 0:
            # f(0)= -0? f(0) = target > 0, f increases? log term: as X grows,
            # CB/CA falls if M>1? Let's just bisect on a residual of the design equation.
            X[i] = hi
        else:
            # Use a direct residual that is zero at the answer
            def g(x, target=target):
                CA = CA0 * (1.0 - x)
                CB = CA0 * (M - x)
                pred = (1.0 / delta) * np.log((CB / CA) / (M / 1.0))
                return pred - target

            if g(0.0) == 0:
                X[i] = 0.0
            elif g(0.0) * g(hi) > 0:
                X[i] = 0.0 if abs(g(0.0)) < abs(g(hi)) else hi
            else:
                X[i] = brentq(g, 0.0, hi)
    return float(X[0]) if scalar else X


def cstr_design_y(X, M: float = 1.0):
    """The quantity that is linear in τ for an ideal CSTR: y = k CA0 τ."""
    X = np.asarray(X, dtype=float)
    M = float(M)
    return X / ((1.0 - X) * (M - X))


def pfr_design_y(X, M: float = 1.0):
    """y = k CA0 τ for an ideal PFR."""
    X = np.asarray(X, dtype=float)
    M = float(M)
    if abs(M - 1.0) < 1e-10:
        return X / (1.0 - X)
    return (1.0 / (M - 1.0)) * np.log((M - X) / (M * (1.0 - X)))


def origin_slope(x, y):
    """Least squares through the origin. Returns slope, stderr, centred R²."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    den = float(np.dot(x, x))
    if den <= 0 or len(x) < 2:
        return np.nan, np.nan, np.nan
    m = float(np.dot(x, y) / den)
    resid = y - m * x
    dof = max(len(x) - 1, 1)
    s2 = float(np.dot(resid, resid) / dof)
    se = float(np.sqrt(s2 / den))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / sst) if sst > 0 else np.nan
    return m, se, r2


def free_line(x, y):
    """y = m x + c. Returns m, c, r, stderr of slope."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    res = stats.linregress(x, y)
    return float(res.slope), float(res.intercept), float(res.rvalue**2), float(res.stderr)


def arrhenius(T_C, k):
    """Fit ln k = ln A - Ea/(R T). T in °C, k in any consistent unit.

    Returns dict with Ea_J, Ea_kJ, A, r2, slope, intercept, se_slope, se_Ea_J.
    """
    T = np.asarray(T_C, dtype=float) + 273.15
    k = np.asarray(k, dtype=float)
    mask = np.isfinite(T) & np.isfinite(k) & (k > 0)
    T, k = T[mask], k[mask]
    if len(T) < 2:
        return None
    inv = 1.0 / T
    lnk = np.log(k)
    res = stats.linregress(inv, lnk)
    Ea = -res.slope * R_GAS
    # stderr of slope is res.stderr
    se_slope = float(res.stderr) if res.stderr is not None else np.nan
    se_Ea = se_slope * R_GAS
    df = max(len(T) - 2, 1)
    tcrit = float(stats.t.ppf(0.975, df))
    return {
        "Ea_J": float(Ea),
        "Ea_kJ": float(Ea / 1000.0),
        "A": float(np.exp(res.intercept)),
        "lnA": float(res.intercept),
        "r2": float(res.rvalue**2),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "se_slope": se_slope,
        "se_Ea_kJ": float(se_Ea / 1000.0),
        "Ea_kJ_low": float((Ea - tcrit * se_Ea) / 1000.0),
        "Ea_kJ_high": float((Ea + tcrit * se_Ea) / 1000.0),
        "n": int(len(T)),
        "df": int(df),
        "T_C": (T - 273.15).tolist(),
        "k": k.tolist(),
        "invT": inv.tolist(),
        "lnk": lnk.tolist(),
    }


def k_arrhenius(T_C, A, Ea_J):
    T = np.asarray(T_C, dtype=float) + 273.15
    return float(A) * np.exp(-float(Ea_J) / (R_GAS * T))


def tank_outlet(CA_in, CB_in, k, tau, CC_in=0.0, CD_in=0.0, k2: float = 0.0):
    """One ideal CSTR. Irreversible if k2 = 0. Returns CA, CB, CC, CD."""
    CA_in, CB_in, CC_in, CD_in = map(float, (CA_in, CB_in, CC_in, CD_in))
    k, tau, k2 = float(k), float(tau), float(k2)
    if tau <= 0 or k < 0:
        return CA_in, CB_in, CC_in, CD_in
    # extent ξ (mol/L reacted in this tank):
    # (CA_in - CA)/τ = k CA CB - k2 CC CD
    # CA = CA_in - ξ, CB = CB_in - ξ, CC = CC_in + ξ, CD = CD_in + ξ
    hi = min(CA_in, CB_in)
    if hi <= 1e-15:
        return CA_in, CB_in, CC_in, CD_in

    def f(xi):
        CA = CA_in - xi
        CB = CB_in - xi
        CC = CC_in + xi
        CD = CD_in + xi
        rate = k * CA * CB - k2 * CC * CD
        return xi / tau - rate

    # f(0) = -(k CA_in CB_in) ≤ 0, f(hi) is positive if rate ≤ 0 at the limit.
    f0 = f(0.0)
    if abs(f0) < 1e-14:
        xi = 0.0
    else:
        # If f(hi) has the same sign, step inward.
        lo, hi_b = 0.0, hi * (1 - 1e-10)
        f_hi = f(hi_b)
        if f0 * f_hi > 0:
            xi = 0.0 if abs(f0) < abs(f_hi) else hi_b
        else:
            xi = brentq(f, lo, hi_b)
    return CA_in - xi, CB_in - xi, CC_in + xi, CD_in + xi


def tanks_in_series(CA0, CB0, k, tau_total, n: int, k2: float = 0.0, CC0=0.0, CD0=0.0):
    """n equal tanks. k may be a list of length n (different temperature each)."""
    n = int(max(n, 1))
    tau_i = float(tau_total) / n
    if np.isscalar(k):
        ks = [float(k)] * n
    else:
        ks = [float(v) for v in k]
        if len(ks) != n:
            raise ValueError("k must be a scalar or one value per tank")
    CA, CB, CC, CD = float(CA0), float(CB0), float(CC0), float(CD0)
    path = [(CA, CB)]
    for ki in ks:
        CA, CB, CC, CD = tank_outlet(CA, CB, ki, tau_i, CC, CD, k2=k2)
        path.append((CA, CB))
    return CA, CB, path


def pfr_integrate(CA_in, CB_in, k, tau, n_steps: int = 80, k2: float = 0.0, CC_in=0.0, CD_in=0.0):
    """Integrate a PFR with dC/dτ_local = r. Returns outlet CA, CB and profiles."""
    from scipy.integrate import solve_ivp

    tau = float(tau)
    if tau <= 0:
        return float(CA_in), float(CB_in), np.array([0.0]), np.array([CA_in]), np.array([CB_in])

    def f(_t, y):
        CA, CB, CC, CD = y
        rate = float(k) * CA * CB - float(k2) * CC * CD
        return [-rate, -rate, rate, rate]

    sol = solve_ivp(
        f,
        (0.0, tau),
        [CA_in, CB_in, CC_in, CD_in],
        t_eval=np.linspace(0.0, tau, n_steps),
        rtol=1e-7,
        atol=1e-9,
        method="RK45",
    )
    if not sol.success:
        # fallback Euler
        ts = np.linspace(0.0, tau, n_steps)
        dt = ts[1] - ts[0]
        y = np.array([CA_in, CB_in, CC_in, CD_in], dtype=float)
        CA = [y[0]]
        CB = [y[1]]
        for _ in ts[1:]:
            dy = np.array(f(0, y))
            y = y + dt * dy
            y[0] = max(y[0], 0.0)
            y[1] = max(y[1], 0.0)
            CA.append(y[0])
            CB.append(y[1])
        return y[0], y[1], ts, np.array(CA), np.array(CB)
    return float(sol.y[0, -1]), float(sol.y[1, -1]), sol.t, sol.y[0], sol.y[1]


def conversion_from_CA(CA, CA0):
    if CA0 <= 0:
        return np.nan
    return float((CA0 - CA) / CA0)


def relaxation_time(tau, X):
    """Linearised time constant of a second-order equimolar CSTR near steady state.

    τ_relax = τ (1-X) / (1+X). A non-reacting tracer is slower: it is still
    37% short of steady after one residence time.
    """
    X = float(X)
    if X >= 1:
        return 0.0
    return float(tau) * (1.0 - X) / (1.0 + X)


def startup_cstr(k, CA0, CB0, tau, t_end, n: int = 300, CA_init=None, CB_init=None, k2: float = 0.0):
    """Fill the tank with unreacted feed, then open the flows. Returns t, CA, CB."""
    from scipy.integrate import solve_ivp

    CA0, CB0, tau = float(CA0), float(CB0), float(tau)
    if CA_init is None:
        CA_init = CA0
    if CB_init is None:
        CB_init = CB0
    t_end = float(max(t_end, 1e-3))

    def f(_t, y):
        CA, CB, CC, CD = y
        rate = float(k) * max(CA, 0.0) * max(CB, 0.0) - float(k2) * max(CC, 0.0) * max(CD, 0.0)
        return [
            (CA0 - CA) / tau - rate,
            (CB0 - CB) / tau - rate,
            (0.0 - CC) / tau + rate,
            (0.0 - CD) / tau + rate,
        ]

    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        f,
        (0.0, t_end),
        [CA_init, CB_init, 0.0, 0.0],
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        method="RK45",
        max_step=t_end / 80.0,
    )
    if not sol.success:
        return t_eval, np.full_like(t_eval, CA_init), np.full_like(t_eval, CB_init)
    return sol.t, sol.y[0], sol.y[1]


def X_of_Da_curve(kind: str, M: float = 1.0, n_tanks: int = 1, n: int = 240):
    """Smooth X(Da) for plotting. kind is 'cstr', 'pfr', or 'tanks'."""
    Da = np.linspace(0.0, 8.0, n)
    if kind == "pfr":
        X = np.array([pfr_X(a, M) for a in Da])
    elif kind == "tanks":
        X = []
        for a in Da:
            # Da = k CA0 τ, CA0 cancels: use CA0=1, k=1, τ=a
            CA, CB, _ = tanks_in_series(1.0, M, 1.0, a if a > 0 else 0.0, n_tanks)
            X.append(1.0 - CA)
        X = np.array(X)
    else:
        X = np.array([cstr_X(a, M) for a in Da])
    return Da, X
