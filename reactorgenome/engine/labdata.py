"""Turn typed lab tables into steady values, conversions and rate constants.

Table columns (one row per run, as on the lab sheet):
    T_C, Q_LPH, tau_min, lam0, r1 … r4, and optionally t1 … t4
Q_LPH is the total flow through the reactor. tau_min may be typed instead of
(or as well as) the flow; a typed τ is used as written. Λ0 may be typed once
per temperature. t is minutes since the flows were set, r is conductivity in
mS/cm. Older tables with separate NaOH and ester flows (qA_LPH, qB_LPH) are
still read.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from ..kinetics import cstr_design_y, cstr_X, pfr_design_y, pfr_X
from . import cstr, steady
from .conductivity_model import lam_inf_of
from .rate import Kinetics
from .units import EngineError, lph_to_lpm

READ_COLS = [("t1", "r1"), ("t2", "r2"), ("t3", "r3"), ("t4", "r4")]

FITS = {
    "y_tau": "y against τ through the origin (lab reports)",
    "tau_y": "τ against y through the origin (manual)",
    "free": "y against τ with an intercept (start-up delay)",
}


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fill_lam0(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a blank Λ0 take the Λ0 typed on another row at the same temperature."""
    if df.empty or "lam0" not in df or "T_C" not in df:
        return df
    out = df.copy()
    T = pd.to_numeric(out["T_C"], errors="coerce")
    lam0 = pd.to_numeric(out["lam0"], errors="coerce")
    first = lam0.groupby(T).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    out["lam0"] = lam0.fillna(first)
    return out


def attach_lam0(df: pd.DataFrame, lam0_table: pd.DataFrame | None) -> pd.DataFrame:
    """Give every run the Λ0 typed for its temperature. A Λ0 typed on the run itself wins."""
    out = df.copy()
    if "lam0" not in out:
        out["lam0"] = np.nan
    if lam0_table is None or lam0_table.empty or "T_C" not in out:
        return out
    t = lam0_table.dropna(subset=["T_C", "lam0"])
    if t.empty:
        return out
    T = pd.to_numeric(out["T_C"], errors="coerce").values
    look = np.array([t.loc[np.isclose(t["T_C"].astype(float), v), "lam0"].iloc[0] if np.isfinite(v) and np.isclose(t["T_C"].astype(float), v).any()
                     else np.nan for v in T], dtype=float)
    out["lam0"] = pd.to_numeric(out["lam0"], errors="coerce").fillna(pd.Series(look, index=out.index))
    return out


def _flows(rec, V):
    """(qA, qB) in LPH and τ in min for one row, or None. Typed τ wins over the flow."""
    qa, qb = _num(rec.get("qA_LPH")), _num(rec.get("qB_LPH"))
    Q = _num(rec.get("Q_LPH"))
    tau = _num(rec.get("tau_min"))
    if not np.isfinite(Q) and np.isfinite(qa) and np.isfinite(qb):
        Q = qa + qb
    split = qa / (qa + qb) if np.isfinite(qa) and np.isfinite(qb) and qa + qb > 0 else 0.5
    if np.isfinite(tau) and tau > 0:
        Q = 60.0 * V / tau
    elif np.isfinite(Q) and Q > 0:
        tau = 60.0 * V / Q
    else:
        return None
    return Q * split, Q * (1 - split), tau


def row_problems(df: pd.DataFrame, V: float | None = None) -> list[str]:
    out = []
    df = fill_lam0(df)
    for i, rec in enumerate(df.to_dict("records"), start=1):
        T, Q, tau, lam0 = (_num(rec.get(k)) for k in ("T_C", "Q_LPH", "tau_min", "lam0"))
        qa, qb = _num(rec.get("qA_LPH")), _num(rec.get("qB_LPH"))
        if not np.isfinite(Q) and np.isfinite(qa) and np.isfinite(qb):
            Q = qa + qb
        reads = [(_num(rec.get(t)), _num(rec.get(r))) for t, r in READ_COLS]
        if all(np.isnan(v) for v in (T, Q, tau, lam0)) and all(np.isnan(r) for _, r in reads):
            continue
        miss = []
        if np.isnan(T):
            miss.append("temperature")
        if np.isnan(Q) and np.isnan(tau):
            miss.append("flow or τ")
        if np.isnan(lam0):
            miss.append(f"Λ0 (add {T:g} °C to the Λ0 table)" if np.isfinite(T) else "Λ0")
        if miss:
            out.append(f"Row {i}: missing {', '.join(miss)}.")
        if not any(np.isfinite(r) for _, r in reads):
            out.append(f"Row {i}: no conductivity readings.")
        if np.isfinite(Q) and Q <= 0:
            out.append(f"Row {i}: the flow must be above zero.")
        if np.isfinite(tau) and tau <= 0:
            out.append(f"Row {i}: τ must be above zero.")
        if V and np.isfinite(Q) and Q > 0 and np.isfinite(tau) and tau > 0:
            tq = 60.0 * V / Q
            if abs(tq - tau) / tau > 0.03:
                out.append(f"Row {i}: typed τ = {tau:.3g} min but V/Q = {tq:.3g} min. The typed τ is used; check the volume or the flow.")
        ts = [t for t, r in reads if np.isfinite(r) and np.isfinite(t)]
        if len(ts) >= 2 and np.any(np.diff(ts) <= 0):
            out.append(f"Row {i}: reading times must increase from t1 to t4.")
        if any(np.isfinite(r) and r < 0 for _, r in reads):
            out.append(f"Row {i}: a conductivity is negative.")
        if np.isfinite(lam0) and any(np.isfinite(r) and r > lam0 * 1.02 for _, r in reads):
            out.append(f"Row {i}: a reading is above Λ0, which would mean negative conversion. Check Λ0 for this temperature.")
    return out


def lam0_suspect(df: pd.DataFrame) -> str | None:
    """Λ0 should be well above every outlet reading. If it sits among them, a reading was probably typed as Λ0."""
    d = df.dropna(subset=["lam0"]) if "lam0" in df else df.iloc[0:0]
    if d.empty:
        return None
    reads = d[[c for _, c in READ_COLS if c in d]].apply(pd.to_numeric, errors="coerce")
    close = (reads.max(axis=1) >= 0.9 * d["lam0"]).sum()
    if close >= max(2, len(d) // 2):
        return (f"In {close} of {len(d)} runs Λ0 is no larger than the outlet readings. Λ0 is the feed before any reaction and should be clearly the "
                "highest value. Check that Λ0 is the feed conductivity for that temperature and not the first outlet reading.")
    return None


def _bottles(qa, qb, CA0, CB0, cA, cB):
    if CA0 is not None:
        Q = qa + qb
        return CA0 * Q / qa, (CB0 if CB0 is not None else CA0) * Q / qb
    return cA, cB


def cstr_runs(df, *, V, kin: Kinetics, CA0=None, CB0=None, cA=None, cB=None, lam_mode="ion", lam_fixed=3.75,
              method="model_tau", criterion="approach", tol_X=0.01, tol_drift=0.02, n_relax=4.0, spacing=2.0,
              start="previous", mixing: dict | None = None) -> pd.DataFrame:
    """Steady-state assessment for every complete row.

    CA0, CB0: concentrations after mixing (mol/L). cA, cB: bottle strengths,
    used only when CA0 is not given. mixing: extra CSTRModel fields (n_tanks,
    bypass, stagnant, exchange) used for τr. start: 'previous' (flows changed
    from an earlier steady run) or 'water' (tank started full of water).
    """
    if CA0 is None and cA is None:
        raise EngineError("Give either the concentrations after mixing or the bottle strengths.")
    mixing = mixing or {}
    rows = []
    for i, rec in enumerate(fill_lam0(df).to_dict("records"), start=1):
        T, lam0 = _num(rec.get("T_C")), _num(rec.get("lam0"))
        fl = _flows(rec, V)
        if fl is None or not np.isfinite(T) or not np.isfinite(lam0):
            continue
        qa, qb, _tau = fl
        reads = [(_num(rec.get(t)), _num(rec.get(r))) for t, r in READ_COLS]
        reads = [(t, r) for t, r in reads if np.isfinite(r)]
        if not reads:
            continue
        assumed = any(not np.isfinite(t) for t, _ in reads)
        tt = np.array([t if np.isfinite(t) else spacing * (j + 1) for j, (t, _) in enumerate(reads)])
        lam = np.array([r for _, r in reads])
        ca, cb = _bottles(qa, qb, CA0, CB0, cA, cB)
        m = cstr.CSTRModel(V=V, qA=lph_to_lpm(qa), qB=lph_to_lpm(qb), cA_feed=ca, cB_feed=cb, T_C=T, kin=kin, **mixing)
        linf = lam_inf_of(lam0, lam_mode, lam_fixed)
        tau_r, shape = np.nan, None
        if method not in steady.AVERAGES:
            try:
                ss = cstr.steady_state(m)
                if start == "previous":
                    prev = cstr.steady_state(replace(m, qA=m.qA * 1.5, qB=m.qB * 1.5))["state"]
                    st_ = cstr.settling(m, ss, y0=prev)
                else:
                    st_ = cstr.settling(m, ss, init="water")
                tau_r = st_["tau_r"]
                if st_["shape"] is not None:
                    sh = st_["shape"]
                    shape = (sh["t"], lam0 * sh["dA"] + linf * sh["dC"])
            except EngineError:
                pass
        a = steady.assess(tt, lam, lam0, linf, method=method, tau_r=tau_r, criterion=criterion,
                          tol_X=tol_X, tol_drift=tol_drift, n_relax=n_relax, shape=shape)
        X = a["X_ss"]
        y = float(cstr_design_y(X, m.M)) if np.isfinite(X) and 0 < X < min(1.0, m.M) else np.nan
        rows.append({
            "run": i, "T_C": T, "Q_LPH": qa + qb, "qA_LPH": qa, "qB_LPH": qb, "Q": m.Q, "tau": m.tau, "CA0": m.CA0,
            "CB0": m.CB0, "M": m.M, "lam0": lam0, "lam_inf": linf, "n_read": len(lam), "lam_avg": float(np.mean(lam)),
            "spread": float(np.ptp(lam)), "lam_ss": a["fit"]["lam_ss"], "lam_ci_lo": a["fit"]["ci"][0],
            "lam_ci_hi": a["fit"]["ci"][1], "X": X, "se_X": a["se_X"], "X_last": a["X_last"], "y": y,
            "k_point": y / (m.CA0 * m.tau) if np.isfinite(y) else np.nan,
            "verdict": a["verdict"], "why": steady.explain(a), "tau_r": a["fit"].get("tau_r") or tau_r,
            "t_ready": a["t_ready"], "times_assumed": assumed, "times_given": not assumed, "fit_warn": a["fit"].get("warn"),
            "_assess": a,
        })
    return pd.DataFrame(rows)


def pfr_runs(df, *, V, CA0, CB0=None, lam_mode="ion", lam_fixed=0.1, method="mean_all") -> pd.DataFrame:
    """Plug-flow runs: one steady outlet value per row from the readings."""
    CB0 = CA0 if CB0 is None else CB0
    M = CB0 / CA0
    rows = []
    for i, rec in enumerate(fill_lam0(df).to_dict("records"), start=1):
        T, lam0 = _num(rec.get("T_C")), _num(rec.get("lam0"))
        fl = _flows(rec, V)
        if fl is None or not np.isfinite(T) or not np.isfinite(lam0):
            continue
        qa, qb, tau = fl
        lam = np.array([v for v in (_num(rec.get(r)) for _, r in READ_COLS) if np.isfinite(v)])
        if not len(lam):
            continue
        use = lam[1:] if method == "mean_after_first" and len(lam) > 1 else (lam[-1:] if method == "last" else lam)
        lam_ss = float(np.mean(use))
        se = float(np.std(use, ddof=1) / np.sqrt(len(use))) if len(use) > 1 else np.nan
        linf = lam_inf_of(lam0, lam_mode, lam_fixed)
        span = lam0 - linf
        X = (lam0 - lam_ss) / span if abs(span) > 1e-12 else np.nan
        y = float(pfr_design_y(X, M)) if np.isfinite(X) and 0 < X < min(1.0, M) else np.nan
        rows.append({
            "run": i, "T_C": T, "Q_LPH": qa + qb, "tau": tau, "CA0": CA0, "CB0": CB0, "M": M, "lam0": lam0, "lam_inf": linf,
            "n_read": len(lam), "lam_avg": float(np.mean(lam)), "lam_ss": lam_ss, "spread": float(np.ptp(lam)),
            "X": X, "se_X": abs(se / span) if np.isfinite(se) and abs(span) > 1e-12 else np.nan, "y": y,
            "k_point": y / (CA0 * tau) if np.isfinite(y) else np.nan,
        })
    return pd.DataFrame(rows)


def design_y(X, M):
    """X / ((1-X)(M-X)): equals k CA0 τ for an ideal tank."""
    X = np.asarray(X, dtype=float)
    return X / ((1 - X) * (M - X))


def y_of_X(X, M, kind, order=2.0):
    """y = k CA0^(n-1) τ for an ideal tank ('cstr') or tube ('pfr').

    Order 2 is −rA = k CA CB and uses M = CB0/CA0. Other orders are
    −rA = k CA^n, with CA = CB for an equimolar feed.
    """
    X = float(X)
    lim = min(1.0, M) if abs(order - 2.0) < 1e-12 else 1.0
    if not np.isfinite(X) or X <= 0 or X >= lim:
        return np.nan
    if abs(order - 2.0) < 1e-12:
        return float(cstr_design_y(X, M) if kind == "cstr" else pfr_design_y(X, M))
    if kind == "cstr":
        return X / (1 - X) ** order
    if abs(order - 1.0) < 1e-12:
        return -np.log(1 - X)
    return ((1 - X) ** (1 - order) - 1) / (order - 1)


def X_of_y(y, M, kind, order=2.0):
    """Inverse of y_of_X."""
    y = float(y)
    if not np.isfinite(y) or y <= 0:
        return 0.0 if np.isfinite(y) else np.nan
    if abs(order - 2.0) < 1e-12:
        return float(cstr_X(y, M) if kind == "cstr" else pfr_X(y, M))
    if kind == "pfr":
        if abs(order - 1.0) < 1e-12:
            return 1 - np.exp(-y)
        base = 1 + (order - 1) * y
        if base <= 0:
            return 1.0
        return 1 - base ** (1 / (1 - order))
    from scipy.optimize import brentq

    return float(brentq(lambda X: X - y * (1 - X) ** order, 0.0, 1.0 - 1e-15))


def with_order(runs: pd.DataFrame, kind: str, order: float = 2.0) -> pd.DataFrame:
    """Recompute y, the scale CA0^(n-1) and the single-run k for a chosen reaction order."""
    if runs.empty:
        return runs
    r = runs.copy()
    r["order"] = order
    r["scale"] = r["CA0"] ** (order - 1)
    r["y"] = [y_of_X(X, M, kind, order) for X, M in zip(r["X"], r["M"])]
    r["k_point"] = r["y"] / (r["scale"] * r["tau"])
    return r


def _origin(x, y):
    """Least squares y = m x. R² measured from zero, as a spreadsheet trendline through the origin reports it."""
    m = float(np.sum(x * y) / np.sum(x * x))
    res = y - m * x
    n = len(x)
    se = float(np.sqrt(np.sum(res**2) / (n - 1) / np.sum(x * x))) if n >= 2 else np.nan
    r2 = 1 - float(np.sum(res**2)) / float(np.sum(y * y)) if n >= 2 and np.sum(y * y) > 0 else np.nan
    return m, se, r2


def fit_lines(runs: pd.DataFrame, how: str = "y_tau") -> pd.DataFrame:
    """One rate constant per temperature from y = k CA0 τ.

    how: 'y_tau' regresses y on τ through the origin, 'tau_y' regresses τ on
    y through the origin (slope 1/(k CA0)), 'free' fits y = a + b τ and reads
    the intercept as a delay δ = a/b that every run shares.
    """
    out = []
    for T, g in runs.groupby("T_C"):
        g = g[np.isfinite(g["y"]) & np.isfinite(g["tau"])]
        if len(g) == 0:
            continue
        x, y = g["tau"].values.astype(float), g["y"].values.astype(float)
        CA0 = float(g["scale"].mean()) if "scale" in g else float(g["CA0"].mean())
        n = len(g)
        tc = float(stats.t.ppf(0.975, n - 1)) if n >= 2 else np.nan
        row = {"T_C": float(T), "n": n, "CA0": CA0, "slope": np.nan, "intercept": 0.0, "delay_min": np.nan, "r2": np.nan,
               "k": np.nan, "k_lo": np.nan, "k_hi": np.nan, "beta": np.nan}
        if how == "tau_y":
            m, se, r2 = _origin(y, x)
            k = 1 / (m * CA0) if m > 0 else np.nan
            lo, hi = (1 / ((m + tc * se) * CA0), 1 / ((m - tc * se) * CA0) if m - tc * se > 0 else np.inf) if n >= 2 else (np.nan, np.nan)
            row.update(slope=m, r2=r2, k=k, k_lo=lo, k_hi=hi)
        elif how == "free" and n >= 2:
            lr = stats.linregress(x, y)
            b, a = float(lr.slope), float(lr.intercept)
            se = float(lr.stderr) if n > 2 else np.nan
            tc2 = float(stats.t.ppf(0.975, n - 2)) if n > 2 else np.nan
            row.update(slope=b, intercept=a, r2=float(lr.rvalue**2) if n > 2 else np.nan, k=b / CA0 if b > 0 else np.nan,
                       k_lo=(b - tc2 * se) / CA0 if n > 2 else np.nan, k_hi=(b + tc2 * se) / CA0 if n > 2 else np.nan,
                       delay_min=a / b if b > 0 else np.nan)
        else:
            m, se, r2 = _origin(x, y)
            row.update(slope=m, r2=r2, k=m / CA0, k_lo=(m - tc * se) / CA0 if n >= 2 else np.nan,
                       k_hi=(m + tc * se) / CA0 if n >= 2 else np.nan)
        out.append(row)
    return pd.DataFrame(out)


def fit_k_design(runs: pd.DataFrame, how: str = "y_tau") -> pd.DataFrame:
    """Manual method for the tank: y = X/((1-X)(M-X)) against τ, per temperature."""
    r = runs.copy()
    if "y" not in r:
        r["y"] = [design_y(X, M) if np.isfinite(X) and 0 < X < min(1, M) else np.nan for X, M in zip(r["X"], r["M"])]
    return fit_lines(r, how)


def fit_lam_inf(runs: pd.DataFrame, kind: str) -> dict | None:
    """Λ∞ as a fraction of Λ0 from the readings themselves.

    For an ideal reactor every run at one temperature shares one k, so the
    outlet readings across several flows fix both k and where Λ would end at
    full conversion. Unknowns: f = Λ∞/Λ0 (shared) and ln k per temperature.
    Needs more runs than unknowns, and is only as good as the spread of τ.
    """
    r = runs[np.isfinite(runs["lam_ss"]) & np.isfinite(runs["tau"])]
    temps = sorted(r["T_C"].unique())
    n, p = len(r), 1 + len(temps)
    if n <= p or any((r["T_C"] == T).sum() < 2 for T in temps):
        return None
    conv = cstr_X if kind == "cstr" else pfr_X
    lam0, lam, tau, CA0, M = (r[c].values.astype(float) for c in ("lam0", "lam_ss", "tau", "CA0", "M"))
    idx = np.array([temps.index(T) for T in r["T_C"]])

    def resid(q):
        k = np.exp(q[1:])[idx]
        X = np.array([float(conv(k[i] * CA0[i] * tau[i], M[i])) for i in range(n)])
        return (lam0 * (1 - X) + q[0] * lam0 * X - lam) / lam0

    k0 = []
    for T in temps:
        g = r[r["T_C"] == T]
        k0.append(np.log(max(float(np.nanmean(g["k_point"])) if np.isfinite(g["k_point"]).any() else 5.0, 1e-3)))
    res = least_squares(resid, [0.37] + k0, bounds=([-0.5] + [-12.0] * len(temps), [0.95] + [12.0] * len(temps)), x_scale="jac")
    dof = n - p
    s2 = float(np.sum(res.fun**2)) / dof
    lo = hi = np.nan
    try:
        cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
        se = float(np.sqrt(max(cov[0, 0], 0.0)))
        tc = float(stats.t.ppf(0.975, dof))
        lo, hi = res.x[0] - tc * se, res.x[0] + tc * se
    except np.linalg.LinAlgError:
        pass
    rms = float(np.sqrt(np.mean((res.fun * lam0) ** 2)))
    signal = float(np.mean([np.ptp(r.loc[r["T_C"] == T, "lam_ss"]) for T in temps]))
    return {"ratio": float(res.x[0]), "lo": float(lo), "hi": float(hi), "k": dict(zip(temps, np.exp(res.x[1:]).tolist())),
            "rms_mScm": rms, "signal_mScm": signal, "snr": signal / rms if rms > 0 else np.inf, "n": n, "dof": dof,
            "at_bound": bool(res.x[0] <= -0.49 or res.x[0] >= 0.94)}


def fit_k_model(runs: pd.DataFrame, *, V, kin: Kinetics, mixing: dict, fit_bypass=False, **_) -> dict:
    """Least squares on conversion with the chosen non-ideal model.

    One k per temperature; optionally one bypass fraction shared by all runs.
    95% intervals from the Jacobian. A bypass fraction and k trade off against
    each other at a single τ, so with few flow rates the interval on β is wide.
    """
    r = runs[np.isfinite(runs["X"])].copy()
    temps = sorted(r["T_C"].unique())
    if not temps:
        raise EngineError("No runs with a usable conversion.")
    names = [f"ln k @ {T:g} °C" for T in temps] + (["β"] if fit_bypass else [])
    k0 = []
    for T in temps:
        g = r[r["T_C"] == T]
        y = design_y(np.clip(g["X"].values, 1e-4, 0.98 * np.minimum(1, g["M"].values)), g["M"].values)
        s = float(np.sum(g["tau"].values * y) / np.sum(g["tau"].values ** 2))
        k0.append(np.log(max(s / g["CA0"].mean(), 1e-6)))
    x0 = k0 + ([0.05] if fit_bypass else [])
    lo = [v - 15 for v in k0] + ([0.0] if fit_bypass else [])
    hi = [v + 15 for v in k0] + ([0.9] if fit_bypass else [])

    def predict(p):
        out = []
        beta = p[-1] if fit_bypass else mixing.get("bypass", 0.0)
        mix = dict(mixing, bypass=beta)
        for rec in r.itertuples():
            k = np.exp(p[temps.index(rec.T_C)])
            Q = rec.qA_LPH + rec.qB_LPH
            m = cstr.CSTRModel(V=V, qA=lph_to_lpm(rec.qA_LPH), qB=lph_to_lpm(rec.qB_LPH), cA_feed=rec.CA0 * Q / rec.qA_LPH,
                               cB_feed=rec.CB0 * Q / rec.qB_LPH, T_C=rec.T_C, kin=replace(kin, k_ref=k, T_ref_C=rec.T_C, Ea=0.0), **mix)
            out.append(cstr.steady_isothermal(m)["X"])
        return np.array(out)

    res = least_squares(lambda p: predict(p) - r["X"].values, x0, bounds=(lo, hi), x_scale="jac")
    resid = res.fun
    n, p = len(resid), len(x0)
    dof = n - p
    ci = {}
    if dof > 0:
        try:
            cov = float(np.sum(resid**2)) / dof * np.linalg.inv(res.jac.T @ res.jac)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            tc = float(stats.t.ppf(0.975, dof))
            for j, nm in enumerate(names):
                ci[nm] = (res.x[j] - tc * se[j], res.x[j] + tc * se[j])
        except np.linalg.LinAlgError:
            pass
    rows = []
    for j, T in enumerate(temps):
        nm = names[j]
        lo_hi = ci.get(nm)
        rows.append({"T_C": float(T), "k": float(np.exp(res.x[j])),
                     "k_lo": float(np.exp(lo_hi[0])) if lo_hi else np.nan,
                     "k_hi": float(np.exp(lo_hi[1])) if lo_hi else np.nan,
                     "n": int((r["T_C"] == T).sum()), "r2": np.nan,
                     "beta": float(res.x[-1]) if fit_bypass else mixing.get("bypass", 0.0)})
    notes = []
    if fit_bypass:
        b = ci.get("β")
        notes.append(f"Fitted bypass β = {res.x[-1]:.3f}" + (f" (95% interval {max(b[0], 0):.3f} to {b[1]:.3f})." if b else "; too few runs for an interval."))
    if dof <= 0:
        notes.append("There are as many unknowns as runs, so the fit passes through every point and says nothing about scatter.")
    r["X_model"] = predict(res.x)
    r["resid"] = r["X"] - r["X_model"]
    return {"kinetics": pd.DataFrame(rows), "runs": r, "notes": notes, "dof": dof}
