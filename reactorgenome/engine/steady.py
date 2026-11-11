"""Has the tank reached steady state, and if not, where is it heading?

After a step change in flow, the outlet of a CSTR relaxes towards its new
steady state. Near the end the approach is a single exponential,

    Λ(t) = Λss + a exp(-t / τr)

τr comes from the model (the slowest time constant of the linearised
balances, τ(1-X)/(1+X) for one ideal tank) or is fitted. With τr fixed the
model is linear in Λss and a, so two readings already give an estimate and
three give an uncertainty. With τr free, four readings are needed for an
uncertainty.

A start from water excites two modes (sodium washing in on τ and the faster
reaction mode), so early readings are not a single exponential. The
'model_shape' method fits Λ(t) = Λss + a g(t) with g(t) the simulated
transient of the chosen start-up, which carries every mode. It is still
linear in Λss and a.

The last reading is never taken as steady by default.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.optimize import OptimizeWarning, curve_fit

METHODS = {
    "model_shape": "Simulated start-up shape (all modes)",
    "model_tau": "Exponential, τr from the model",
    "free_tau": "Exponential, τr fitted",
    "mean_all": "Average of all readings (lab sheet)",
    "mean_after_first": "Average after the first reading (manual)",
    "last": "Last reading",
}

AVERAGES = ("mean_all", "mean_after_first", "last")

CRITERIA = {
    "approach": "Last reading within a conversion tolerance of the predicted steady value",
    "drift": "Conductivity changing slower than a set rate",
    "elapsed": "Time since the change longer than a multiple of τr",
}


def _clean(t, lam):
    t = np.asarray(t, dtype=float)
    lam = np.asarray(lam, dtype=float)
    ok = np.isfinite(t) & np.isfinite(lam)
    t, lam = t[ok], lam[ok]
    o = np.argsort(t)
    return t[o], lam[o]


def _tcrit(dof):
    return float(stats.t.ppf(0.975, dof)) if dof > 0 else float("nan")


def _linear(out, A, lam, n_unknown_msg):
    n = len(lam)
    coef, *_ = np.linalg.lstsq(A, lam, rcond=None)
    res = lam - A @ coef
    out.update(lam_ss=float(coef[0]), a=float(coef[1]), resid=res)
    dof = n - 2
    if dof > 0:
        s2 = float(np.sum(res**2) / dof)
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = float(np.sqrt(max(cov[0, 0], 0.0)))
        tc = _tcrit(dof)
        out.update(se=se, ci=(coef[0] - tc * se, coef[0] + tc * se), rmse=float(np.sqrt(s2)))
    else:
        out["warn"] = n_unknown_msg
    return out


def fit_plateau(t, lam, method: str = "model_tau", tau_r: float | None = None, shape=None) -> dict:
    """shape: (t_grid, g) simulated deviation of Λ from steady, for 'model_shape'."""
    t, lam = _clean(t, lam)
    n = len(t)
    out = {"method": method, "n": n, "t": t, "lam": lam, "lam_ss": np.nan, "se": np.nan, "ci": (np.nan, np.nan),
           "tau_r": tau_r, "a": np.nan, "rmse": np.nan, "resid": np.full(n, np.nan), "warn": None, "shape": None}
    if n == 0:
        out["warn"] = "No readings."
        return out
    if method == "last":
        out["lam_ss"] = float(lam[-1])
        out["warn"] = "Taking the last reading as steady assumes the tank had already stopped changing."
        return out
    if method in ("mean_all", "mean_after_first"):
        use = lam[1:] if n > 1 and method == "mean_after_first" else lam
        out["lam_ss"] = float(np.mean(use))
        if len(use) > 1:
            se = float(np.std(use, ddof=1) / np.sqrt(len(use)))
            tc = _tcrit(len(use) - 1)
            out.update(se=se, ci=(out["lam_ss"] - tc * se, out["lam_ss"] + tc * se))
        out["resid"] = lam - out["lam_ss"]
        return out
    if n < 2:
        out["lam_ss"] = float(lam[-1])
        out["warn"] = "One reading cannot show a trend. It is used as it is."
        return out
    t0 = t[0]
    two = "Two readings and two unknowns: the curve passes through both, so there is no uncertainty to report."
    if method == "model_shape":
        if shape is not None:
            tg, g = (np.asarray(v, dtype=float) for v in shape)
            gi = np.interp(t, tg, g)
            if np.ptp(gi) > 1e-9 * max(np.max(np.abs(g)), 1e-30) and t[-1] <= tg[-1]:
                g = g / gi[0] if abs(gi[0]) > 0 else g
                out["shape"] = (tg, g)
                return _linear(out, np.column_stack([np.ones(n), np.interp(t, tg, g)]), lam, two)
        out["warn"] = "No usable simulated shape for these reading times, so a single exponential with τr from the model is used."
        method = "model_tau"
    if method == "free_tau" and n >= 3:
        span = max(t[-1] - t0, 1e-6)
        guess_tau = tau_r if tau_r and np.isfinite(tau_r) and tau_r > 0 else span / 2

        def fn(tt, ss, a, tr):
            return ss + a * np.exp(-(tt - t0) / tr)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, cov = curve_fit(fn, t, lam, p0=[lam[-1], lam[0] - lam[-1], guess_tau],
                                   bounds=([-np.inf, -np.inf, span * 0.02], [np.inf, np.inf, span * 50]), maxfev=20000)
            res = lam - fn(t, *p)
            out.update(lam_ss=float(p[0]), a=float(p[1]), tau_r=float(p[2]), resid=res)
            dof = n - 3
            if dof > 0 and np.all(np.isfinite(cov)):
                se = float(np.sqrt(cov[0, 0]))
                tc = _tcrit(dof)
                out.update(se=se, ci=(p[0] - tc * se, p[0] + tc * se), rmse=float(np.sqrt(np.sum(res**2) / dof)))
            else:
                out["warn"] = "Three readings and three unknowns: the curve passes through every point, so there is no uncertainty to report."
            if p[2] >= span * 49:
                out["warn"] = "The fitted τr ran to its upper limit. The readings look like a straight line, so the plateau is a long extrapolation."
            return out
        except (RuntimeError, ValueError):
            out["warn"] = "The three-parameter fit did not converge, so τr from the model is used instead."
            method = "model_tau"
    if not tau_r or not np.isfinite(tau_r) or tau_r <= 0:
        out["warn"] = "No τr available, so the last reading is used."
        out["lam_ss"] = float(lam[-1])
        return out
    out.update(tau_r=float(tau_r), method="model_tau")
    return _linear(out, np.column_stack([np.ones(n), np.exp(-(t - t0) / tau_r)]), lam, two)


FITTED = ("model_shape", "model_tau", "free_tau")


def _dev(fit: dict, t_grid):
    t_grid = np.asarray(t_grid, dtype=float)
    if fit["method"] == "model_shape":
        tg, g = fit["shape"]
        return fit["a"] * np.interp(t_grid, tg, g)
    return fit["a"] * np.exp(-(t_grid - fit["t"][0]) / fit["tau_r"])


def curve(fit: dict, t_grid):
    """Fitted Λ(t) on a grid, or None when the method has no curve."""
    if fit["method"] not in FITTED or not np.isfinite(fit.get("a", np.nan)):
        return None
    return fit["lam_ss"] + _dev(fit, t_grid)


def assess(t, lam, lam0, lam_inf, *, method="model_tau", tau_r=None, criterion="approach",
           tol_X=0.01, tol_drift=0.02, n_relax=4.0, shape=None) -> dict:
    """Fit the plateau and judge whether the last reading is steady.

    tol_X: conversion tolerance for the 'approach' criterion.
    tol_drift: mS/cm per min for the 'drift' criterion.
    n_relax: number of τr for the 'elapsed' criterion.
    Time is measured from the moment the flows were set.
    """
    fit = fit_plateau(t, lam, method, tau_r, shape)
    tt, ll = fit["t"], fit["lam"]
    span = lam0 - lam_inf
    to_X = (lambda v: (lam0 - v) / span) if abs(span) > 1e-12 else (lambda v: np.nan)
    X_ss = to_X(fit["lam_ss"]) if np.isfinite(fit["lam_ss"]) else np.nan
    se_X = abs(fit["se"] / span) if np.isfinite(fit["se"]) and abs(span) > 1e-12 else np.nan
    lo, hi = fit["ci"]
    X_ci = tuple(sorted((to_X(lo), to_X(hi)))) if np.isfinite(lo) else (np.nan, np.nan)
    X_last = to_X(ll[-1]) if len(ll) else np.nan

    drift = np.nan
    if len(tt) >= 2 and tt[-1] > tt[-2]:
        drift = float((ll[-1] - ll[-2]) / (tt[-1] - tt[-2]))
    gap = abs(X_last - X_ss) if np.isfinite(X_last) and np.isfinite(X_ss) else np.nan
    tr = fit.get("tau_r") or tau_r
    elapsed = float(tt[-1] / tr) if len(tt) and tr and np.isfinite(tr) and tr > 0 else np.nan

    checks = {
        "approach": {"value": gap, "limit": tol_X, "ok": bool(np.isfinite(gap) and gap <= tol_X),
                     "known": bool(np.isfinite(gap) and fit["method"] in FITTED),
                     "text": (f"last reading is {gap:.3f} in conversion from the predicted plateau (limit {tol_X:.3f})"
                              if np.isfinite(gap) else "needs a fitted plateau")},
        "drift": {"value": drift, "limit": tol_drift, "ok": bool(np.isfinite(drift) and abs(drift) <= tol_drift),
                  "known": bool(np.isfinite(drift)),
                  "text": (f"conductivity moved {drift:+.3f} mS/cm per min over the last interval (limit ±{tol_drift:.3f})"
                           if np.isfinite(drift) else "needs two readings at different times")},
        "elapsed": {"value": elapsed, "limit": n_relax, "ok": bool(np.isfinite(elapsed) and elapsed >= n_relax),
                    "known": bool(np.isfinite(elapsed)),
                    "text": (f"last reading taken {elapsed:.1f} τr after the change (needs {n_relax:.0f} τr)"
                             if np.isfinite(elapsed) else "needs τr and reading times")},
    }
    sel = checks[criterion]
    if not sel["known"]:
        verdict = "unknown"
    else:
        verdict = "steady" if sel["ok"] else "not_steady"

    wait = np.nan
    if fit["method"] == "model_shape" and np.isfinite(fit.get("a", np.nan)) and abs(span) > 1e-12:
        tg = fit["shape"][0]
        dev = np.abs(_dev(fit, tg) / span)
        late = np.nonzero(dev > tol_X)[0]
        if len(late) and late[-1] + 1 < len(tg):
            wait = float(tg[late[-1] + 1])
    elif fit["method"] in FITTED and np.isfinite(fit.get("a", np.nan)) and tr and abs(span) > 1e-12:
        a_X = abs(fit["a"] / span)
        if a_X > tol_X:
            wait = float(tt[0] + tr * np.log(a_X / tol_X))
    return {
        "fit": fit, "X_ss": X_ss, "se_X": se_X, "X_ci": X_ci, "X_last": X_last, "drift": drift,
        "gap_X": gap, "elapsed_tau": elapsed, "checks": checks, "criterion": criterion,
        "verdict": verdict, "t_ready": wait,
    }


def explain(a: dict) -> str:
    """One plain paragraph on why the reading is or is not steady."""
    c = a["checks"][a["criterion"]]
    if a["verdict"] == "unknown":
        return f"Cannot judge with this criterion: it {c['text']}."
    head = "Steady." if a["verdict"] == "steady" else "Not steady yet."
    tail = ""
    if a["verdict"] != "steady" and np.isfinite(a["t_ready"]):
        tail = f" The fit says the outlet should be within tolerance from about t = {a['t_ready']:.1f} min."
    return f"{head} The {c['text']}.{tail}"
