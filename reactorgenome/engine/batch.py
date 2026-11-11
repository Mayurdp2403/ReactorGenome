"""Isothermal batch reactor: concentration from conductivity, rate laws, fits.

Time inside this module is minutes. The lab reads seconds; the UI converts.

Rate laws offered
    first     -rA = k1 CA                    (k1 in 1/min)
    second    -rA = k CA CB, CB = CA + (M-1)CA0
    nth       -rA = k CA^n                   (n fitted)
    reversible -rA = k (CA CB - K CC CD)     (K = k2/k1 fitted)

Each fit is non-linear least squares on concentration, with an optional
clock offset t0 (the clock was started a little after mixing). Parameter
uncertainty comes from the Jacobian: cov = s² (JᵀJ)⁻¹.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .units import EngineError, safe_log

ORDERS = {
    "first": "First order in NaOH",
    "second": "Second order, k·CA·CB",
    "nth": "Power law, k·CA^n (n fitted)",
    "reversible": "Second order, reversible",
}


def conc_from_kappa(kappa, kappa0, kappa_inf, CA0):
    """NaOH concentration from conductivity. Linear because hydroxide is
    replaced one-for-one by acetate."""
    den = kappa0 - kappa_inf
    if abs(den) < 1e-12:
        raise EngineError("κ0 and κ∞ are equal, so conductivity cannot tell how far the reaction went.")
    return CA0 * (np.asarray(kappa, dtype=float) - kappa_inf) / den


def model_C(t, order: str, k: float, CA0: float, M: float = 1.0, n: float = 2.0, K: float = 0.0):
    """CA(t) for one rate law. t in min, k in matching units."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, None)
    if order == "first":
        return CA0 * np.exp(-k * t)
    if order == "second":
        if abs(M - 1.0) < 1e-9:
            return CA0 / (1.0 + k * CA0 * t)
        E = np.exp(np.clip((M - 1.0) * CA0 * k * t, -700, 700))
        X = M * (E - 1.0) / (M * E - 1.0)
        return CA0 * (1.0 - X)
    if order == "nth":
        if abs(n - 1.0) < 1e-9:
            return CA0 * np.exp(-k * t)
        base = CA0 ** (1.0 - n) + (n - 1.0) * k * t
        if n < 1:
            return np.where(base > 0, np.clip(base, 0, None) ** (1.0 / (1.0 - n)), 0.0)
        return np.clip(base, 1e-300, None) ** (1.0 / (1.0 - n))
    if order == "reversible":
        CB0 = M * CA0

        def f(_t, y):
            ca = max(y[0], 0.0)
            x = CA0 - ca
            return [-k * (ca * (CB0 - x) - K * x * x)]

        tmax = float(np.max(t)) if t.size else 0.0
        if tmax <= 0:
            return np.full_like(t, CA0)
        sol = solve_ivp(f, (0.0, tmax), [CA0], t_eval=np.sort(np.unique(t)), method="LSODA", rtol=1e-9, atol=1e-12)
        if not sol.success:
            raise EngineError(f"Reversible batch model failed: {sol.message}")
        return np.interp(t, sol.t, sol.y[0])
    raise EngineError(f"Unknown rate law '{order}'.")


def model_rate(C, order: str, k: float, CA0: float, M: float = 1.0, n: float = 2.0, K: float = 0.0):
    C = np.clip(np.asarray(C, dtype=float), 0.0, None)
    if order == "first":
        return k * C
    if order == "second":
        return k * C * (C + (M - 1.0) * CA0)
    if order == "nth":
        return k * C**n
    x = CA0 - C
    return k * (C * (M * CA0 - x) - K * x * x)


def integral_y(C, CA0, order: str, M: float = 1.0):
    """The linearising function y(C) with y = k t for that order.

    first: ln(CA0/C). second equimolar: 1/C - 1/CA0.
    second non-equimolar: ln[(M - X)/(M(1 - X))] / ((M-1) CA0).
    """
    C = np.clip(np.asarray(C, dtype=float), 1e-12, None)
    if order == "first":
        return safe_log(CA0 / C)
    if abs(M - 1.0) < 1e-9:
        return 1.0 / C - 1.0 / CA0
    X = 1.0 - C / CA0
    return safe_log((M - X) / (M * (1.0 - X))) / ((M - 1.0) * CA0)


def fit(t, C, order: str, CA0: float, M: float = 1.0, *, fit_t0: bool = True, k_guess: float | None = None) -> dict:
    """Least squares on CA(t). Returns parameters, 95% intervals, residuals."""
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)
    ok = np.isfinite(t) & np.isfinite(C)
    t, C = t[ok], C[ok]
    if len(t) < 3:
        raise EngineError("At least three readings inside the fitting window are needed.")
    if CA0 <= 0:
        raise EngineError("Initial NaOH concentration must be above zero.")
    span = max(t.max() - t.min(), 1e-6)
    if k_guess is None or not np.isfinite(k_guess) or k_guess <= 0:
        y = integral_y(C, CA0, "second" if order != "first" else "first", M)
        slope = np.polyfit(t, y, 1)[0] if len(t) >= 2 else 1.0
        k_guess = max(abs(slope), 1e-6)
    names = ["ln k"]
    x0 = [np.log(k_guess)]
    lo = [np.log(k_guess) - 25]
    hi = [np.log(k_guess) + 25]
    if order == "nth":
        names.append("n")
        x0.append(2.0)
        lo.append(0.0)
        hi.append(4.0)
    if order == "reversible":
        names.append("K")
        x0.append(0.05)
        lo.append(0.0)
        hi.append(50.0)
    if fit_t0:
        names.append("t0")
        x0.append(0.0)
        lo.append(-min(t.min(), span))
        hi.append(span)

    def unpack(p):
        d = dict(zip(names, p))
        return np.exp(d["ln k"]), d.get("n", 2.0), d.get("K", 0.0), d.get("t0", 0.0)

    def resid(p):
        k, n, K, t0 = unpack(p)
        return model_C(t + t0, order, k, CA0, M, n, K) - C

    res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
    k, n, K, t0 = unpack(res.x)
    r = res.fun
    dof = len(t) - len(names)
    sse = float(np.sum(r**2))
    sst = float(np.sum((C - C.mean()) ** 2))
    out = {"order": order, "k": k, "n": n, "K": K, "t0": t0, "sse": sse,
           "rmse": float(np.sqrt(sse / dof)) if dof > 0 else np.nan,
           "r2": 1.0 - sse / sst if sst > 0 else np.nan, "resid": r, "t": t, "C": C,
           "names": names, "dof": dof, "ci": {}, "corr": None, "notes": [], "success": bool(res.success)}
    if dof > 0:
        J = res.jac
        try:
            cov = sse / dof * np.linalg.inv(J.T @ J)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            tc = float(stats.t.ppf(0.975, dof))
            for i, nm in enumerate(names):
                if nm == "ln k":
                    out["ci"]["k"] = (float(np.exp(res.x[i] - tc * se[i])), float(np.exp(res.x[i] + tc * se[i])))
                else:
                    out["ci"][nm] = (float(res.x[i] - tc * se[i]), float(res.x[i] + tc * se[i]))
            d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
            corr = cov / np.outer(d, d)
            out["corr"] = corr
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if abs(corr[i, j]) > 0.95:
                        out["notes"].append(
                            f"{names[i].replace('ln k', 'k')} and {names[j]} move together (correlation {corr[i, j]:+.2f}). "
                            "The data cannot pull them apart, so treat each one's interval with care."
                        )
        except np.linalg.LinAlgError:
            out["notes"].append("The parameters are not identifiable from these readings (singular Jacobian).")
    return out


def simulate(CA0: float, M: float, kin, order: str, t_end: float, *, n: float = 2.0, adiabatic: bool = False,
             T0_C: float = 30.0, n_out: int = 241) -> dict:
    """Forward run of the batch reactor with k(T) from the Kinetics object.

    Adiabatic: ρcp dT/dt = (-ΔH)(-rA). The rate law is the one chosen; the
    reverse term uses kin.k2_ratio.
    """
    from .units import c_to_k, k_to_c

    if CA0 <= 0 or t_end <= 0:
        raise EngineError("Initial concentration and run time must be above zero.")
    CB0 = M * CA0

    def r(ca, T):
        k = kin.k(T)
        ca = max(ca, 0.0)
        x = CA0 - ca
        if order == "first":
            return k * ca
        if order == "nth":
            return k * ca**n
        cb = max(CB0 - x, 0.0)
        return k * (ca * cb - kin.k2_ratio * x * x)

    def f(_t, y):
        rr = r(y[0], y[1])
        return [-rr, (-kin.dH) * rr / kin.rho_cp if adiabatic else 0.0]

    t = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(f, (0.0, t_end), [CA0, c_to_k(T0_C)], t_eval=t, method="LSODA", rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise EngineError(f"Batch simulation failed: {sol.message}")
    CA = np.clip(sol.y[0], 0.0, None)
    return {"t": t, "CA": CA, "X": 1.0 - CA / CA0, "T_C": k_to_c(sol.y[1]),
            "rate": np.array([r(c, T) for c, T in zip(CA, sol.y[1])])}


def differential(t, C, window: int = 7) -> dict:
    """Rate from a smoothed derivative; log rate against log C gives the order."""
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)
    if len(t) < 5:
        return {"n": np.nan, "k": np.nan, "r2": np.nan, "C": C, "rate": np.full_like(C, np.nan)}
    w = min(window if window % 2 else window + 1, len(t) if len(t) % 2 else len(t) - 1)
    Cs = savgol_filter(C, w, 2) if w >= 5 else C
    rate = -np.gradient(Cs, t)
    ok = (rate > 0) & (Cs > 0)
    if ok.sum() < 4:
        return {"n": np.nan, "k": np.nan, "r2": np.nan, "C": Cs, "rate": rate}
    x, y = np.log(Cs[ok]), np.log(rate[ok])
    slope, icpt, r, *_ = stats.linregress(x, y)
    return {"n": float(slope), "k": float(np.exp(icpt)), "r2": float(r**2), "C": Cs, "rate": rate, "mask": ok}
