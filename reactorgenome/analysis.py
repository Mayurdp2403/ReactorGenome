"""Turn a table of lab readings into k, Ea, and a plain-language diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter

from reactorgenome.conductivity import conversion, lambda_from_conversion, lambda_inf_ion
from reactorgenome.flow import (
    coil_flow,
    ideal_cstr_E,
    laminar_pipe_E,
    moments,
    peclet_closed,
    peclet_open,
    segregated_X,
    tanks_E,
    tanks_N,
)
from reactorgenome.kinetics import (
    arrhenius,
    batch_X_of_time,
    cstr_X,
    cstr_design_y,
    free_line,
    origin_slope,
    pfr_X,
    pfr_design_y,
    relaxation_time,
    startup_cstr,
    tank_outlet,
)

LITERATURE_EA = (40.0, 50.0)  # kJ/mol, published range for alkaline hydrolysis of ethyl acetate


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


def readings(row):
    vals, times = [], []
    for i in range(1, 5):
        v = _f(row.get(f"r{i}"))
        if np.isfinite(v):
            vals.append(v)
            t = _f(row.get(f"t{i}"))
            times.append(t if np.isfinite(t) else np.nan)
    return vals, times


def extrapolate_lambda(times, values):
    """Fit Λ(t) = Λss + a exp(-(t-t0)/τr). Also return the three-point geometric limit."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    geo = geometric_limit(y)
    if len(y) < 3 or not np.all(np.isfinite(t)):
        return {"lam_ss": float(np.mean(y)), "tau_r": np.nan, "geo": geo, "ok": False}
    t0 = float(np.min(t))

    def model(tt, L, a, tau):
        return L + a * np.exp(-(tt - t0) / tau)

    span = max(float(np.ptp(t)), 0.2)
    p0 = [float(y[-1]), float(y[0] - y[-1]), span]
    try:
        popt, pcov = curve_fit(
            model,
            t,
            y,
            p0=p0,
            bounds=([-5.0, -80.0, 0.05], [80.0, 80.0, 400.0]),
            maxfev=8000,
        )
        se = float(np.sqrt(max(pcov[0, 0], 0.0)))
        pred = model(t, *popt)
        sse_fit = float(np.sum((y - pred) ** 2))
        sse_mean = float(np.sum((y - np.mean(y)) ** 2))
        return {
            "lam_ss": float(popt[0]),
            "a": float(popt[1]),
            "tau_r": float(popt[2]),
            "se": se,
            "geo": geo,
            "ok": sse_fit < sse_mean - 1e-6,
            "model": model,
            "popt": popt,
        }
    except Exception:
        return {"lam_ss": float(y[-1] if geo is None else geo), "tau_r": np.nan, "geo": geo, "ok": False}


def geometric_limit(y):
    """If the steps shrink by a constant ratio, finish the infinite tail."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return None
    d1 = y[-2] - y[-3]
    d2 = y[-1] - y[-2]
    if d1 == 0 or d1 * d2 <= 0:
        return None
    r = d2 / d1
    if not (0.05 < r < 0.98):
        return None
    return float(y[-1] + d2 * r / (1.0 - r))


def choose_lambda(vals, times, how, spacing):
    """how: 'all', 'drop_first', 'last', 'extrapolate'."""
    vals = list(vals)
    if how == "drop_first" and len(vals) > 1:
        used = vals[1:]
        used_t = list(times[1:])
    elif how == "last":
        used = vals[-1:]
        used_t = [times[-1]]
    else:
        used = vals
        used_t = list(times)
    mean = float(np.mean(used))
    std = float(np.std(used, ddof=1)) if len(used) > 1 else 0.0
    se = std / np.sqrt(len(used)) if used else np.nan
    extra = {"fit": None, "lam_ss": mean}
    if how == "extrapolate":
        t = np.array(used_t, dtype=float)
        if not np.all(np.isfinite(t)):
            t = np.arange(len(used), dtype=float) * float(spacing)
        extra["fit"] = extrapolate_lambda(t, used)
        extra["lam_ss"] = extra["fit"]["lam_ss"]
        # If the exponential is wild, prefer the geometric tail or the last points.
        L = extra["lam_ss"]
        lo, hi = min(used) - 1.5, max(used) + 1.5
        if not (lo <= L <= hi):
            geo = extra["fit"].get("geo")
            L = float(geo) if geo is not None and lo <= geo <= hi else mean
            extra["lam_ss"] = L
    return {
        "vals": vals,
        "used": used,
        "mean": mean,
        "std": std,
        "se": se,
        "lam": extra["lam_ss"] if how == "extrapolate" else mean,
        "drift": float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0,
        **extra,
    }


def feed_state(qA, qB, cA, cB):
    qA, qB = float(qA), float(qB)
    Q = qA + qB
    if Q <= 0:
        return 0.0, 0.0, 0.0, np.nan
    CA0 = cA * qA / Q
    CB0 = cB * qB / Q
    M = CB0 / CA0 if CA0 > 0 else np.nan
    return Q, CA0, CB0, M


def analyse_cstr(df, V, cA, cB, lam_inf_mode, lam_inf_fixed, how, spacing, k2_ratio=0.0):
    """lam_inf_mode: 'manual' or 'ion'. how: averaging method. k2_ratio = k2/k1."""
    rows_out = []
    for rec in df.to_dict("records"):
        T = _f(rec.get("T_C"))
        qA = _f(rec.get("qA_LPH"))
        qB = _f(rec.get("qB_LPH"))
        lam0 = _f(rec.get("lam0"))
        if not all(np.isfinite(v) for v in (T, qA, qB, lam0)):
            continue
        vals, times = readings(rec)
        if len(vals) < 1:
            continue
        Q, CA0, CB0, M = feed_state(qA, qB, cA, cB)
        tau = 60.0 * float(V) / Q
        chosen = choose_lambda(vals, times, how, spacing)
        if lam_inf_mode == "ion":
            linf = lambda_inf_ion(lam0)
        else:
            linf = float(lam_inf_fixed)
        X = float(conversion(chosen["lam"], lam0, linf))
        se_lam = chosen["se"]
        se_X = abs(se_lam / (lam0 - linf)) if abs(lam0 - linf) > 1e-9 else np.nan
        y = float(cstr_design_y(np.clip(X, 1e-6, min(0.995, M - 1e-4)), M)) if np.isfinite(X) else np.nan
        point_k = y / (tau * CA0) if tau * CA0 > 0 and np.isfinite(y) else np.nan
        rows_out.append(
            {
                "T_C": T,
                "Q": Q,
                "tau": tau,
                "CA0": CA0,
                "CB0": CB0,
                "M": M,
                "lam0": lam0,
                "lam_inf": linf,
                "lam_mean": chosen["mean"],
                "lam_used": chosen["lam"],
                "se_lam": se_lam,
                "drift": chosen["drift"],
                "X": X,
                "se_X": se_X,
                "y": y,
                "k_point": point_k,
                "readings": vals,
                "n_used": len(chosen["used"]),
                "fit": chosen.get("fit"),
                "tau_relax": relaxation_time(tau, X) if np.isfinite(X) and abs(M - 1) < 1e-6 else np.nan,
            }
        )
    table = pd.DataFrame(rows_out)
    if table.empty:
        kin = pd.DataFrame(columns=["T_C", "k", "se_k", "slope", "r2", "CA0", "n"])
        return {"runs": table, "kinetics": kin, "arrhenius": None, "notes": diagnose_cstr(table, kin, None, how, lam_inf_mode)}
    by_T = []
    for T, g in table.groupby("T_C"):
        m, se_m, r2 = origin_slope(g["tau"].values, g["y"].values)
        CA0 = float(g["CA0"].mean())
        k = m / CA0 if CA0 else np.nan
        se_k = se_m / CA0 if CA0 else np.nan
        by_T.append({"T_C": float(T), "k": float(k), "se_k": float(se_k), "slope": float(m), "r2": float(r2), "CA0": CA0, "n": len(g)})
    kin = pd.DataFrame(by_T)
    arr = arrhenius(kin["T_C"], kin["k"]) if len(kin) >= 2 else None
    notes = diagnose_cstr(table, kin, arr, how, lam_inf_mode)
    return {"runs": table, "kinetics": kin, "arrhenius": arr, "notes": notes}


def diagnose_cstr(table, kin, arr, how, lam_mode):
    notes = []
    if table.empty:
        return [{"level": "warn", "text": "No readings yet. Put in temperature, both flow rates, Λ0 and the conductivity numbers."}]
    # drift
    drifters = table[table["drift"].abs() > 0.08]
    for _, r in drifters.iterrows():
        direction = "rising" if r["drift"] > 0 else "falling"
        notes.append(
            {
                "level": "warn",
                "text": (
                    f"At {r['T_C']:.0f} °C and {r['Q']:.0f} LPH the conductivity is still {direction} "
                    f"({r['drift']:+.2f} mS/cm from the first reading to the last). "
                    f"An average of those numbers is not a steady state. "
                    f"The reading used here is {r['lam_used']:.3f} mS/cm, which gives X = {r['X']:.3f}."
                ),
            }
        )
    # conversion should rise with residence time, inside one temperature
    for T, g in table.groupby("T_C"):
        g = g.sort_values("tau")
        X = g["X"].values
        tau = g["tau"].values
        if len(g) >= 3 and X[np.argmax(tau)] < np.max(X) - 0.02:
            bad = g.iloc[int(np.argmax(X.values if hasattr(X, 'values') else X))]
            # the point with highest X that is not the longest tau
            order = np.argsort(X)
            top = g.iloc[int(np.argmax(X))]
            if abs(top["tau"] - tau.max()) > 0.2:
                notes.append(
                    {
                        "level": "warn",
                        "text": (
                            f"At {T:.0f} °C the highest conversion is at {top['Q']:.0f} LPH "
                            f"(τ = {top['tau']:.2f} min, X = {top['X']:.3f}), not at the longest residence time. "
                            f"A longer stay in the tank should give more conversion. "
                            f"That point is pulling the line for this temperature. "
                            f"A flow rate that was really lower than the rotameter, or a bad sample, would do this."
                        ),
                    }
                )
    if arr is not None:
        lo, hi = LITERATURE_EA
        if arr["r2"] < 0.8 or not (lo <= arr["Ea_kJ"] <= hi):
            notes.append(
                {
                    "level": "warn",
                    "text": (
                        f"Arrhenius fit: Ea = {arr['Ea_kJ']:.1f} kJ/mol, A = {arr['A']:.4g} L/mol/min, "
                        f"R² = {arr['r2']:.3f}. With only {arr['n']} temperatures the 95% interval on Ea is "
                        f"{arr['Ea_kJ_low']:.0f} to {arr['Ea_kJ_high']:.0f} kJ/mol, which is too wide to trust. "
                        f"Published values for this reaction sit around {lo:.0f}–{hi:.0f} kJ/mol. "
                        f"If Ea is outside that, the number is describing this data set, not the chemistry."
                    ),
                }
            )
        else:
            notes.append(
                {
                    "level": "good",
                    "text": (
                        f"Arrhenius fit: Ea = {arr['Ea_kJ']:.1f} kJ/mol, A = {arr['A']:.4g} L/mol/min, "
                        f"R² = {arr['r2']:.3f}. That Ea sits inside the {lo:.0f}–{hi:.0f} kJ/mol range quoted for ethyl acetate saponification."
                    ),
                }
            )
    if lam_mode == "ion":
        notes.append(
            {
                "level": "info",
                "text": (
                    "Λ∞ is not the manual's constant. It is Λ0 times (λ_Na + λ_acetate) / (λ_Na + λ_OH), "
                    "about 0.37 Λ0. The CSTR manual uses 3.75 mS/cm and the PFR manual uses 0.1 mS/cm for the same reaction. "
                    "Switch this off to reproduce a report exactly."
                ),
            }
        )
    if how == "extrapolate":
        notes.append(
            {
                "level": "info",
                "text": (
                    "Each run is fitted as Λ(t) = Λss + a exp(−t/τr), and Λss is what goes into the conversion. "
                    "A tracer with no reaction is only 63% of the way to steady after one residence time. "
                    "Near steady state this reaction settles faster, in about τ(1−X)/(1+X). The readings are the check."
                ),
            }
        )
    if not notes:
        notes.append({"level": "good", "text": "The runs line up with an ideal equimolar CSTR. k comes from the slope of X/(1−X)² against τ, forced through the origin."})
    return notes


def predict_cstr_lambda(k, CA0, M, tau, lam0, lam_inf):
    Da = float(k) * float(CA0) * float(tau)
    X = float(cstr_X(Da, M))
    lam = float(lambda_from_conversion(X, lam0, lam_inf))
    return X, lam


def analyse_pfr(df, V, cA, cB, lam_inf_mode, lam_inf_fixed, line_mode):
    """line_mode 'free' (report) or 'origin' (forced through zero)."""
    rows = []
    for rec in df.to_dict("records"):
        T = _f(rec.get("T_C"))
        Q = _f(rec.get("Q_LPH"))
        lam0 = _f(rec.get("lam0"))
        # allow qA qB instead of Q
        if not np.isfinite(Q):
            qA, qB = _f(rec.get("qA_LPH")), _f(rec.get("qB_LPH"))
            Q = qA + qB if np.isfinite(qA) and np.isfinite(qB) else np.nan
            CA0 = cA * qA / Q if Q else np.nan
            CB0 = cB * qB / Q if Q else np.nan
        else:
            CA0 = float(cA) / 2.0 if abs(cA - cB) < 1e-9 else float(cA) * 0.5
            # equal split assumed when only total Q is given
            CA0 = float(cA) / 2.0
            CB0 = float(cB) / 2.0
        if not all(np.isfinite(v) for v in (T, Q, lam0)):
            continue
        vals, _times = readings(rec)
        if not vals:
            continue
        tau = 60.0 * float(V) / float(Q)
        M = CB0 / CA0 if CA0 else np.nan
        if lam_inf_mode == "ion":
            linf = lambda_inf_ion(lam0)
        else:
            linf = float(lam_inf_fixed)
        lam = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        X = float(conversion(lam, lam0, linf))
        y = float(pfr_design_y(np.clip(X, 1e-6, 0.98), M))
        k_point = y / (tau * CA0) if tau * CA0 else np.nan
        rows.append(
            {
                "T_C": T, "Q": float(Q), "tau": tau, "CA0": CA0, "M": M,
                "lam0": lam0, "lam_inf": linf, "lam": lam, "se_lam": se,
                "X": X, "y": y, "k_point": k_point, "spread": float(np.ptp(vals)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        kin = pd.DataFrame(columns=["T_C", "k", "slope_min", "intercept_min", "delta_min", "r2", "CA0", "se_slope"])
        return {"runs": table, "kinetics": kin, "arrhenius": None, "notes": diagnose_pfr(table, kin, None, line_mode, lam_inf_mode)}
    by_T = []
    for T, g in table.groupby("T_C"):
        CA0 = float(g["CA0"].mean())
        if line_mode == "origin":
            # τ = m * y, m = 1/(k CA0)
            m, se_m, r2 = origin_slope(g["y"].values, g["tau"].values)
            c = 0.0
        else:
            m, c, r2, se_m = free_line(g["y"].values, g["tau"].values)
        k = (1.0 / (m * CA0)) if m and m > 0 else np.nan
        by_T.append(
            {
                "T_C": float(T), "k": float(k), "slope_min": float(m), "intercept_min": float(c),
                "delta_min": float(-c), "r2": float(r2), "CA0": CA0, "se_slope": float(se_m),
            }
        )
    kin = pd.DataFrame(by_T)
    arr = arrhenius(kin["T_C"], kin["k"]) if len(kin) >= 2 and np.isfinite(kin["k"]).sum() >= 2 else None
    notes = diagnose_pfr(table, kin, arr, line_mode, lam_inf_mode)
    return {"runs": table, "kinetics": kin, "arrhenius": arr, "notes": notes}


def diagnose_pfr(table, kin, arr, line_mode, lam_mode):
    notes = []
    if table.empty:
        return [{"level": "warn", "text": "No PFR readings yet."}]
    big = kin[kin["delta_min"] > 2] if "delta_min" in kin else kin.iloc[0:0]
    if line_mode == "free" and len(big):
        bits = ", ".join(f"{r.T_C:.0f} °C → about {r.delta_min:.1f} min" for r in big.itertuples())
        notes.append(
            {
                "level": "warn",
                "text": (
                    f"The free fit of τ against X/(1−X) does not pass through the origin. "
                    f"The intercept is minus the extra time the reaction kept going in the beaker and the outlet tube: {bits}. "
                    f"Residence time in this coil is only about a minute, so that extra time is the largest error in the run. "
                    f"k is taken from the slope, which is not moved by a delay that is the same at every flow rate. "
                    f"Forcing the line through the origin mixes that delay into k and makes k look too big, especially at high flow."
                ),
            }
        )
    # point k rising with Q
    for T, g in table.groupby("T_C"):
        g = g.sort_values("Q")
        if len(g) >= 3 and g["k_point"].iloc[-1] > 1.4 * g["k_point"].iloc[0]:
            notes.append(
                {
                    "level": "info",
                    "text": (
                        f"At {T:.0f} °C the single-point k rises from {g['k_point'].iloc[0]:.1f} to "
                        f"{g['k_point'].iloc[-1]:.1f} L/mol/min as the flow goes up. "
                        f"That is what a fixed extra reaction time looks like: it is a small fraction of a long τ and a large fraction of a short one."
                    ),
                }
            )
    noisy = table[table["spread"] > 0.4]
    for _, r in noisy.iterrows():
        notes.append(
            {
                "level": "info",
                "text": (
                    f"At {r['T_C']:.0f} °C and {r['Q']:.0f} LPH the four readings scatter by {r['spread']:.2f} mS/cm. "
                    f"The average is still used. A reading that far from the others is worth a second look before it is believed."
                ),
            }
        )
    if arr is not None:
        lo, hi = LITERATURE_EA
        inside = lo <= arr["Ea_kJ"] <= hi
        notes.append(
            {
                "level": "good" if inside else "warn",
                "text": (
                    f"Ea = {arr['Ea_kJ']:.1f} kJ/mol, A = {arr['A']:.4g} L/mol/min, R² = {arr['r2']:.3f}. "
                    + (
                        f"That is inside the {lo:.0f}–{hi:.0f} kJ/mol range for this reaction."
                        if inside
                        else f"The usual range quoted for this reaction is {lo:.0f}–{hi:.0f} kJ/mol."
                    )
                ),
            }
        )
    if lam_mode == "ion":
        notes.append(
            {
                "level": "info",
                "text": "Λ∞ is estimated from the ion conductivities, not held at 0.1 mS/cm. Conversion will move. Switch back to the manual value to match the report line by line.",
            }
        )
    return notes


def coil_diagnosis(Q, V, d_mm, D_mm, T, k, CA0):
    """Compare ideal plug flow with a laminar straight-tube RTD at this k."""
    info = coil_flow(Q, V, d_mm, D_mm, T)
    tau = 60.0 * float(V) / float(Q)
    Da = float(k) * float(CA0) * tau
    X_plug = float(pfr_X(Da))
    t = np.linspace(tau / 2.0 + 1e-4, tau * 40.0, 800)
    E = laminar_pipe_E(t, tau)
    X_lam = segregated_X(t, E, k, CA0, 1.0)
    info.update({"tau": tau, "X_plug": X_plug, "X_laminar": float(X_lam), "Da": Da})
    if info["Re"] < 2300:
        text = (
            f"At {Q:.0f} LPH, with a {d_mm:.0f} mm tube on a {D_mm:.0f} mm coil, Re ≈ {info['Re']:.0f} and Dean number ≈ {info['De']:.0f}. "
            f"That is laminar. A straight laminar tube would give about X = {X_lam:.2f}. Ideal plug flow at the same k gives X = {X_plug:.2f}. "
            f"A helical coil is used because the Dean vortices stir the tube and pull the real reactor toward plug flow. "
            f"The tube diameter here is a guess until someone measures it."
        )
    else:
        text = (
            f"Re ≈ {info['Re']:.0f}. That is not laminar, so a plug-flow design equation is a fair first picture. "
            f"Tube diameter {d_mm:.0f} mm is assumed."
        )
    info["text"] = text
    return info


def analyse_batch(t, ec, C0, lam0, lam_inf, t_lo, t_hi, smooth_window=7):
    """Integral and differential fits. ec in the same units as lam0 and lam_inf."""
    t = np.asarray(t, dtype=float)
    ec = np.asarray(ec, dtype=float)
    mask = np.isfinite(t) & np.isfinite(ec)
    t, ec = t[mask], ec[mask]
    order = np.argsort(t)
    t, ec = t[order], ec[order]
    C = C0 * (ec - lam_inf) / (lam0 - lam_inf)
    C = np.clip(C, 1e-8, None)
    X = 1.0 - C / C0
    win = (t >= t_lo) & (t <= t_hi) & (C < 0.95 * C0)
    tw, Cw = t[win], C[win]
    # integral: 1/C = 1/C0 + k t   but the intercept is free, because the clock
    # does not start at a perfect unreacted mixture. Slope is k.
    if len(tw) >= 5:
        inv = 1.0 / Cw
        m, c, r2, se = free_line(tw, inv)
        k_int = float(m)  # per second if t is seconds
    else:
        k_int, r2, se, c = np.nan, np.nan, np.nan, np.nan
    # differential
    n_order, k_diff, r2_d = np.nan, np.nan, np.nan
    if len(tw) >= smooth_window + 2:
        w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        w = min(w, len(tw) - (1 - len(tw) % 2))
        if w >= 5 and w % 2 == 1:
            Cs = savgol_filter(Cw, w, 2)
        else:
            Cs = Cw
        dC = np.gradient(Cs, tw)
        rate = -dC
        ok = rate > 0
        if ok.sum() >= 5:
            slope, intercept, r2_d, _ = free_line(np.log10(Cs[ok]), np.log10(rate[ok]))
            n_order = float(slope)
            # k from enforcing n = 2: rate = k C^2, so k = rate/C^2, geometric mean
            k_each = rate[ok] / Cs[ok] ** 2
            k_diff = float(np.exp(np.mean(np.log(k_each))))
    X_model = None
    if np.isfinite(k_int):
        # time unit is whatever t is. batch_X expects k and t consistent, CA0, equimolar.
        X_model = batch_X_of_time(t, k_int, C0, 1.0)
    notes = []
    if np.isfinite(n_order):
        if abs(n_order - 2) > 0.3:
            notes.append(
                {
                    "level": "warn",
                    "text": (
                        f"The log-log plot of rate against concentration has slope n = {n_order:.2f} "
                        f"(R² = {r2_d:.3f}) on {t_lo:.0f}–{t_hi:.0f}. "
                        f"That is not 2. The integral k assumes second order anyway, which is what the manual asks, "
                        f"but the differential plot is saying the assumption is weak on this window. "
                        f"Early points are mixing as well as reaction, and late points are almost flat, so the window matters."
                    ),
                }
            )
        else:
            notes.append(
                {
                    "level": "good",
                    "text": f"Differential order n = {n_order:.2f}, close to 2, on {t_lo:.0f}–{t_hi:.0f}.",
                }
            )
    if np.isfinite(k_int) and np.isfinite(k_diff) and k_int != 0:
        pct = abs(k_int - k_diff) / abs(k_int) * 100.0
        notes.append(
            {
                "level": "info",
                "text": (
                    f"Integral k = {k_int:.4f} (from the slope of 1/C against t, R² = {r2:.3f}). "
                    f"Differential k with n fixed at 2 is {k_diff:.4f}. They differ by {pct:.1f}%."
                ),
            }
        )
    return {
        "t": t,
        "ec": ec,
        "C": C,
        "X": X,
        "k_integral": k_int,
        "k_differential": k_diff,
        "n": n_order,
        "r2_integral": r2,
        "r2_diff": r2_d,
        "intercept": c,
        "X_model": X_model,
        "notes": notes,
        "window": (t_lo, t_hi),
    }


def analyse_rtd(t, signal, background, Q_Lph, V_L, kind):
    m = moments(t, signal, background)
    if m is None:
        return None
    tau = 60.0 * float(V_L) / float(Q_Lph) if Q_Lph else np.nan
    N = tanks_N(m["tm"], m["var"], tau)
    Pe_o = peclet_open(m["var_over_tm2"])
    Pe_c = peclet_closed(m["var_over_tm2"])
    t_line = np.linspace(0, max(m["t"][-1], tau if np.isfinite(tau) else 1), 300)
    if kind == "CSTR":
        N_show = max(int(round(N.get("N_from_tm", 1))), 1)
        E_model = tanks_E(t_line, m["tm"], N_show)
        E_ideal = ideal_cstr_E(t_line, tau if np.isfinite(tau) else m["tm"])
    else:
        E_model = None
        E_ideal = None
        N_show = None
    dead = None
    if np.isfinite(tau) and tau > 0:
        dead = float(1.0 - m["tm"] / tau)
    notes = diagnose_rtd(kind, m, tau, N, Pe_o, dead)
    return {
        "moments": m,
        "tau": tau,
        "N": N,
        "N_show": N_show,
        "Pe_open": Pe_o,
        "Pe_closed": Pe_c,
        "t_line": t_line,
        "E_model": E_model,
        "E_ideal": E_ideal,
        "dead_fraction": dead,
        "notes": notes,
        "kind": kind,
    }


def diagnose_rtd(kind, m, tau, N, Pe, dead):
    notes = []
    if kind == "CSTR":
        n_tm = N.get("N_from_tm", np.nan)
        n_tau = N.get("N_from_tau", np.nan)
        notes.append(
            {
                "level": "info",
                "text": (
                    f"Mean residence time tm = {m['tm']:.2f} min. "
                    + (f"The space time τ = V/Q = {tau:.2f} min. " if np.isfinite(tau) else "")
                    + f"Tanks in series from the measured RTD: N = tm²/σ² = {n_tm:.2f}. "
                    + (f"The report's version, N = τ²/σ², is {n_tau:.2f}. " if np.isfinite(n_tau) else "")
                    + "Those differ when tm and τ differ. tm²/σ² is the one that belongs to the curve you measured."
                ),
            }
        )
        if dead is not None and dead > 0.08:
            notes.append(
                {
                    "level": "warn",
                    "text": (
                        f"tm is {dead*100:.0f}% shorter than τ. "
                        f"Either some of the volume is not in the flowing path, or the tail of the curve was cut off before it got back to the baseline. "
                        f"Both make the tank look smaller than the ruler says."
                    ),
                }
            )
        if np.isfinite(n_tm) and n_tm > 1.3:
            notes.append(
                {
                    "level": "info",
                    "text": (
                        f"N around {n_tm:.1f} means this vessel is not one perfect mixer. "
                        f"The kinetic CSTR formula assumes N = 1. Using N from this RTD in the sandbox changes the conversion you would predict for the saponification."
                    ),
                }
            )
    else:
        notes.append(
            {
                "level": "info",
                "text": (
                    f"tm = {m['tm']:.2f} min, σ²/tm² = {m['var_over_tm2']:.4f}. "
                    f"Open-open Peclet number ≈ {Pe:.1f}. "
                    f"A large Pe means a narrow peak, close to plug flow. "
                    f"The closed-closed formula gives a different Pe from the same variance; both are shown."
                ),
            }
        )
        if dead is not None and abs(dead) > 0.05:
            notes.append(
                {
                    "level": "info",
                    "text": (
                        f"τ was computed with the volume you typed ({tau:.2f} min). "
                        f"tm/τ = {m['tm']/tau:.2f}. The Group 21 report used 2.4 L for the packed bed as well as the CSTR. "
                        f"If the bed is not 2.4 L, change the volume. Pe comes from the shape of the peak, so it does not need that volume."
                    ),
                }
            )
    return notes


def fit_bypass(taus, Xs, CA0):
    """Fit k and a bypass fraction. Returns k, beta, sse_bypass, sse_ideal."""
    taus = np.asarray(taus, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    def X_of(k, beta, tau):
        if beta >= 0.9:
            return 0.0
        Da = k * CA0 * tau / (1.0 - beta)
        return (1.0 - beta) * float(cstr_X(Da, 1.0))

    def resid(p):
        k = np.exp(p[0])
        beta = p[1]
        return np.array([X_of(k, beta, t) for t in taus]) - Xs

    # ideal k from origin on y, as a start
    y = cstr_design_y(np.clip(Xs, 1e-4, 0.95))
    m, _, _ = origin_slope(taus, y)
    k0 = max(m / CA0, 1e-4) if np.isfinite(m) else 1.0
    ideal = np.array([float(cstr_X(k0 * CA0 * t, 1.0)) for t in taus])
    sse_i = float(np.sum((ideal - Xs) ** 2))
    try:
        res = least_squares(resid, x0=[np.log(k0), 0.05], bounds=([np.log(1e-4), 0.0], [np.log(1e4), 0.8]))
        k = float(np.exp(res.x[0]))
        beta = float(res.x[1])
        sse_b = float(np.sum(res.fun**2))
    except Exception:
        k, beta, sse_b = k0, 0.0, sse_i
    return {"k": k, "beta": beta, "sse_bypass": sse_b, "sse_ideal": sse_i, "k_ideal": float(k0)}


def mixing_bounds(k, CA0, tau, t_measure=None, E_measure=None):
    """Segregation versus an ideal CSTR, for a second-order reaction.

    Segregation (each droplet is its own batch reactor) converts more than a
    perfect mixer when the order is above 1. The ideal CSTR is that mixed end.
    """
    tau = float(tau)
    t = np.linspace(0.0, tau * 12.0, 500)
    E = ideal_cstr_E(t, tau)
    X_seg = segregated_X(t, E, k, CA0, 1.0)
    X_ideal = float(cstr_X(k * CA0 * tau, 1.0))
    out = {"X_seg_ideal_cstr": X_seg, "X_cstr": X_ideal}
    if t_measure is not None and E_measure is not None:
        out["X_seg_measured"] = segregated_X(t_measure, E_measure, k, CA0, 1.0)
    return out
