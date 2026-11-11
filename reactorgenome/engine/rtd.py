"""Residence time distribution from a tracer pulse.

E(t) = C(t) / ∫C dt, tm = ∫ t E dt, σ² = ∫ (t - tm)² E dt, all by the
trapezoid rule on the measured points. If the curve was stopped before it
returned to baseline, an exponential tail fitted to the last points can be
added analytically; how much area that tail carries is reported so the user
can see whether the cut-off mattered.

Models compared against the measured E(t)
    ideal CSTR        E = exp(-t/τ)/τ
    ideal PFR         a spike at τ (no spread)
    tanks in series   E = (N/tm)^N t^(N-1) exp(-N t/tm) / Γ(N), N fitted (not rounded)
    open dispersion   E = sqrt(Pe/(4π θ)) exp(-Pe (1-θ)²/(4θ)) / τd, θ = t/τd
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq, least_squares
from scipy.special import gammaln

from .units import EngineError


def clean(t, signal, background=0.0):
    t = np.asarray(t, dtype=float)
    c = np.asarray(signal, dtype=float) - float(background)
    ok = np.isfinite(t) & np.isfinite(c)
    t, c = t[ok], c[ok]
    o = np.argsort(t)
    t, c = t[o], np.clip(c[o], 0.0, None)
    if len(t) >= 2:
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, c = t[keep], c[keep]
    return t, c


def _tail(t, c, n_last=None):
    """Exponential through the falling end of the curve: c = c_e exp(-(t - t_e)/λ)."""
    n = len(t)
    n_last = n_last or max(3, n // 4)
    tt, cc = t[-n_last:], c[-n_last:]
    ok = cc > 0
    if ok.sum() < 3:
        return None
    slope, icpt = np.polyfit(tt[ok], np.log(cc[ok]), 1)
    if slope >= 0:
        return None
    lam = -1.0 / slope
    c_e = float(np.exp(icpt + slope * t[-1]))
    return {"lam": float(lam), "c_e": c_e, "t_e": float(t[-1])}


def analyse(t, signal, background=0.0, *, tail: bool = False, V=None, Q=None) -> dict:
    """Moments and curves. V in L, Q in L/min (optional, for τ = V/Q)."""
    t, c = clean(t, signal, background)
    if len(t) < 4:
        raise EngineError("At least four tracer readings are needed for an RTD.")
    m0 = float(trapezoid(c, t))
    if m0 <= 0:
        raise EngineError("The tracer signal is zero everywhere after removing the background.")
    m1 = float(trapezoid(t * c, t))
    m2 = float(trapezoid(t * t * c, t))
    m3 = float(trapezoid(t**3 * c, t))
    tail_info = None
    if tail:
        tl = _tail(t, c)
        if tl is not None:
            L, ce, te = tl["lam"], tl["c_e"], tl["t_e"]
            a0 = ce * L
            a1 = ce * L * (te + L)
            a2 = ce * L * (te**2 + 2 * te * L + 2 * L**2)
            a3 = ce * L * (te**3 + 3 * te**2 * L + 6 * te * L**2 + 6 * L**3)
            m0, m1, m2, m3 = m0 + a0, m1 + a1, m2 + a2, m3 + a3
            tail_info = dict(tl, area_frac=a0 / m0)
    tm = m1 / m0
    var = max(m2 / m0 - tm**2, 0.0)
    mu3 = m3 / m0 - 3 * tm * m2 / m0 + 2 * tm**3
    skew = mu3 / var**1.5 if var > 0 else np.nan
    E = c / m0
    F = cumulative_trapezoid(E, t, initial=0.0)
    tau = V / Q if V and Q and Q > 0 else np.nan
    end_frac = float(c[-1] / c.max()) if c.max() > 0 else np.nan
    return {
        "t": t, "C": c, "E": E, "F": F, "area": m0, "tm": tm, "var": var, "sigma": np.sqrt(var),
        "v": var / tm**2 if tm > 0 else np.nan, "skew": float(skew), "theta": t / tm, "E_theta": E * tm,
        "tau": tau, "tail": tail_info, "end_frac": end_frac,
    }


def E_cstr(t, tau):
    t = np.asarray(t, dtype=float)
    return np.where(t >= 0, np.exp(-t / tau) / tau, 0.0)


def E_tanks(t, tm, N):
    """Gamma-distribution form of tanks in series; N need not be a whole number."""
    t = np.clip(np.asarray(t, dtype=float), 1e-12, None)
    N = float(N)
    logE = N * np.log(N / tm) + (N - 1) * np.log(t) - N * t / tm - gammaln(N)
    return np.exp(logE)


def E_dispersion(t, tau_d, Pe):
    """Open-open vessel. Mean is τd(1 + 2/Pe), variance/τd² = 2/Pe + 8/Pe²."""
    th = np.clip(np.asarray(t, dtype=float) / tau_d, 1e-9, None)
    return np.sqrt(Pe / (4 * np.pi * th)) * np.exp(-Pe * (1 - th) ** 2 / (4 * th)) / tau_d


def N_moments(r: dict) -> float:
    return float(r["tm"] ** 2 / r["var"]) if r["var"] > 0 else np.inf


def Pe_open_tm(v: float) -> float:
    """Pe from σ²/tm² = 2/Pe + 8/Pe². This is the form the lab report uses."""
    if v <= 1e-12:
        return np.inf
    return float((1 + np.sqrt(1 + 8 * v)) / v)


def Pe_open_exact(v: float) -> float:
    """Pe from σ²/tm² with tm = τ(1 + 2/Pe), the consistent open-open result.

    σ²/tm² = (2/Pe + 8/Pe²) / (1 + 2/Pe)². This ratio never exceeds 2.
    """
    if v <= 1e-12:
        return np.inf
    if v >= 2.0:
        return np.nan

    def g(Pe):
        return (2 / Pe + 8 / Pe**2) / (1 + 2 / Pe) ** 2 - v

    return float(brentq(g, 1e-6, 1e8))


def Pe_closed(v: float) -> float:
    """Closed-closed: σ²/tm² = 2/Pe - 2(1 - e^-Pe)/Pe² (tm = τ for a closed vessel)."""
    if v <= 1e-12:
        return np.inf
    if v >= 1.0:
        return np.nan

    def g(Pe):
        return 2 / Pe - 2 * (1 - np.exp(-Pe)) / Pe**2 - v

    return float(brentq(g, 1e-6, 1e8))


def fit_models(r: dict) -> list[dict]:
    """Fit each model to the measured E(t). SSE is on E itself."""
    t, E = r["t"], r["E"]
    sst = float(np.sum((E - E.mean()) ** 2))
    rows = []

    def add(name, params, Ehat, k_params):
        sse = float(np.sum((Ehat - E) ** 2))
        rows.append({"model": name, "params": params, "sse": sse, "r2": 1 - sse / sst if sst > 0 else np.nan,
                     "n_params": k_params, "E": Ehat})

    tau_ref = r["tau"] if np.isfinite(r["tau"]) else r["tm"]
    add("Ideal CSTR (τ)", {"τ": tau_ref}, E_cstr(t, tau_ref), 0)
    add("Ideal CSTR (tm)", {"τ": r["tm"]}, E_cstr(t, r["tm"]), 0)

    def res_t(p):
        return E_tanks(t, p[1], p[0]) - E

    N0 = max(N_moments(r), 0.3) if np.isfinite(N_moments(r)) else 2.0
    try:
        s = least_squares(res_t, [min(N0, 200), r["tm"]], bounds=([0.2, r["tm"] * 0.2], [400.0, r["tm"] * 5]))
        add("Tanks in series (fitted)", {"N": float(s.x[0]), "tm": float(s.x[1])}, E_tanks(t, s.x[1], s.x[0]), 2)
    except ValueError:
        pass
    add("Tanks in series (moments)", {"N": N0, "tm": r["tm"]}, E_tanks(t, r["tm"], N0), 0)

    def res_d(p):
        return E_dispersion(t, p[0], p[1]) - E

    Pe0 = Pe_open_tm(r["v"]) if np.isfinite(r["v"]) else 10.0
    try:
        s = least_squares(res_d, [r["tm"] / (1 + 2 / max(Pe0, 0.5)), min(max(Pe0, 0.5), 2000)],
                          bounds=([r["tm"] * 0.1, 0.05], [r["tm"] * 5, 1e4]))
        add("Axial dispersion, open (fitted)", {"τd": float(s.x[0]), "Pe": float(s.x[1])}, E_dispersion(t, s.x[0], s.x[1]), 2)
    except ValueError:
        pass
    return rows


def summary(r: dict, *, Q_Lmin=None, bed: dict | None = None) -> dict:
    """The numbers a lab report quotes for one tracer run.

    bed: optional {'L_m', 'D_m', 'eps'} for D_L = u L / (ε Pe), with u the
    superficial velocity Q/A. Pe here is the open-vessel lab form, and
    τ = tm / (1 + 2/Pe) is the open-vessel space time.
    """
    N = N_moments(r)
    Pe = Pe_open_tm(r["v"])
    out = {"tm": r["tm"], "var": r["var"], "sigma": r["sigma"], "v": r["v"], "N": N,
           "N_whole": max(int(round(N)), 1) if np.isfinite(N) else np.nan, "Pe": Pe,
           "Pe_exact": Pe_open_exact(r["v"]), "Pe_closed": Pe_closed(r["v"]),
           "tau_Pe": r["tm"] / (1 + 2 / Pe) if np.isfinite(Pe) and Pe > 0 else np.nan,
           "tau_VQ": r["tau"], "skew": r["skew"], "end_frac": r["end_frac"],
           "tail_frac": r["tail"]["area_frac"] if r.get("tail") else 0.0, "D_L": np.nan, "u": np.nan}
    if bed and Q_Lmin and all(bed.get(k, 0) > 0 for k in ("L_m", "D_m", "eps")) and np.isfinite(Pe) and Pe > 0:
        A = np.pi * bed["D_m"] ** 2 / 4
        u = Q_Lmin / 1000.0 / 60.0 / A
        out.update(u=u, D_L=u * bed["L_m"] / (bed["eps"] * Pe))
    if Q_Lmin:
        out["recovered"] = Q_Lmin * r["area"]
    return out


def calibration(conc, signal) -> dict:
    """Straight line C = a·signal + b through standards, as the lab calibrates TDS against NaCl."""
    from scipy import stats

    c = np.asarray(conc, dtype=float)
    s = np.asarray(signal, dtype=float)
    ok = np.isfinite(c) & np.isfinite(s)
    c, s = c[ok], s[ok]
    if len(c) < 2 or np.ptp(s) <= 0:
        raise EngineError("A calibration needs at least two standards with different readings.")
    lr = stats.linregress(s, c)
    return {"a": float(lr.slope), "b": float(lr.intercept), "r2": float(lr.rvalue**2) if len(c) > 2 else np.nan,
            "n": int(len(c)), "conc": c, "signal": s}


def segregated_conversion(t, E, k, CA0, M=1.0, K=0.0):
    """X = ∫ Xbatch(t) E(t) dt using the measured E (renormalised on its own range)."""
    from .batch import model_C

    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    area = trapezoid(E, t)
    if area <= 0:
        return np.nan
    order = "reversible" if K > 0 else "second"
    Xb = 1.0 - model_C(t, order, k, CA0, M, K=K) / CA0
    return float(trapezoid(Xb * E, t) / area)


def bed_frames(tau_d, Pe, t_grid, n_z=60):
    """Tracer cloud along a packed bed for the animation (open-open solution)."""
    z = np.linspace(0, 1, n_z)
    frames = []
    for tt in t_grid:
        th = max(tt / tau_d, 1e-6)
        c = np.sqrt(Pe / (4 * np.pi * th)) * np.exp(-Pe * (z - th) ** 2 / (4 * th))
        frames.append(c)
    F = np.array(frames)
    return z, F / F.max() if F.max() > 0 else F
