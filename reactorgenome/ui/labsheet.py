"""Lab-sheet tables, fits and plots shared by the tank and tube pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..engine import labdata
from ..engine.units import c_to_k
from ..kinetics import arrhenius
from . import charts as ch
from . import widgets as w
from .style import AMBER, CYAN, SERIES

R_GAS = 8.314462618
ORDERS = (1.0, 1.5, 2.0, 2.5, 3.0)
LAM0_COLS = [
    {"key": "T_C", "label": "T (°C)", "help": "Temperature.", "fmt": "%.1f"},
    {"key": "lam0", "label": "Λ0 (mS/cm)", "help": "Conductivity of the mixed feed before any reaction at this temperature "
                                                  "(NaOH at CA0, read before the ester meets it).", "fmt": "%.3f"},
]


# ------------------------------------------------------------------ tables

def columns(n_read: int, reading: str, times: bool = True) -> list[dict]:
    cols = [
        {"key": "T_C", "label": "T (°C)", "help": "Reactor temperature for this run.", "fmt": "%.1f"},
        {"key": "Q_LPH", "label": "Total flow (LPH)", "help": "Total flow through the reactor, litres per hour (both feed streams together). "
                                                              "Leave blank if you type τ instead.", "fmt": "%.2f"},
        {"key": "tau_min", "label": "τ (min)", "help": "Space time V/Q in minutes. Blank: worked out from the volume and the flow. "
                                                       "If typed, the typed value is used.", "fmt": "%.3f"},
    ]
    for i in range(1, n_read + 1):
        if times:
            cols.append({"key": f"t{i}", "label": f"t{i} (min)", "help": f"Minutes after the flow was set when reading {i} was taken. Optional.", "fmt": "%.2f"})
        cols.append({"key": f"r{i}", "label": f"{reading}{i} (mS/cm)", "help": f"Outlet conductivity, reading {i}. Type the individual readings, "
                                                                                 "not their average.", "fmt": "%.3f"})
    return cols


def migrate(store: str, lam0_store: str):
    """Older tables: separate NaOH and ester flows, and Λ0 typed on each run."""
    df = st.session_state.get(store)
    if not isinstance(df, pd.DataFrame):
        return
    changed = False
    if "Q_LPH" not in df.columns and {"qA_LPH", "qB_LPH"} <= set(df.columns):
        df = df.copy()
        df.insert(1, "Q_LPH", df["qA_LPH"] + df["qB_LPH"])
        df = df.drop(columns=["qA_LPH", "qB_LPH"])
        changed = True
    if "lam0" in df.columns:
        pairs = df[["T_C", "lam0"]].apply(pd.to_numeric, errors="coerce").dropna().drop_duplicates("T_C")
        if len(pairs):
            old = st.session_state.get(lam0_store)
            old = old[["T_C", "lam0"]].dropna() if isinstance(old, pd.DataFrame) and {"T_C", "lam0"} <= set(old.columns) else pd.DataFrame(columns=["T_C", "lam0"])
            both = pd.concat([old, pairs[~pairs["T_C"].isin(old["T_C"])]], ignore_index=True).astype("float64")
            st.session_state[lam0_store] = both
            st.session_state[f"{lam0_store}__v"] = st.session_state.get(f"{lam0_store}__v", 0) + 1
        df = df.drop(columns=["lam0"])
        changed = True
    if changed:
        st.session_state[store] = df
        st.session_state[f"{store}__v"] = st.session_state.get(f"{store}__v", 0) + 1


def readings_section(store, lam0_store, cols, V, file_name, hidden=()):
    """Readings table plus the Λ0-per-temperature table. Returns readings with Λ0 attached, and the problems found."""
    migrate(store, lam0_store)
    c1, c2 = st.columns([3.2, 1], gap="medium")
    with c1:
        st.markdown("**Runs**")
        df = w.table_editor(store, cols, n_blank=9, file_name=file_name, hidden=hidden)
    with c2:
        st.markdown("**Λ0 for each temperature**")
        lt = w.table_editor(lam0_store, LAM0_COLS, n_blank=3, file_name=f"lam0_{file_name}", compact=True)
        st.caption("Conductivity of the feed before reaction, one row per temperature. It must be higher than every outlet reading.")
    out = labdata.attach_lam0(df, lt) if len(df) else df
    issues = labdata.row_problems(out, V) if len(out) else []
    sus = labdata.lam0_suspect(out) if len(out) else None
    if sus:
        issues.append(sus)
    return out, issues


# ------------------------------------------------------------------ settings and analysis

def inputs_used(df, V, CA0, CB0, lmode, linf, average):
    """One line naming every setting the fit depends on, so a mismatch with the lab sheet is visible."""
    typed = "tau_min" in df and pd.to_numeric(df["tau_min"], errors="coerce").notna().any() if len(df) else False
    lam = "Λ∞ = 0.366 Λ0 (ion estimate)" if lmode == "ion" else f"Λ∞ = {linf:g} mS/cm (typed)"
    tau = "τ as typed in the table" if typed else f"τ = V/Q with V = {V:g} L"
    feed = f"CA0 = {CA0:g} mol/L" + ("" if abs(CB0 - CA0) < 1e-12 else f", CB0 = {CB0:g} mol/L")
    return f"{feed} · {lam} · {tau} · outlet Λ from the {average.lower()}"


def k_unit(order: float) -> str:
    if abs(order - 2) < 1e-12:
        return "L/mol/min"
    if abs(order - 1) < 1e-12:
        return "1/min"
    return f"(L/mol)^{order - 1:g}/min"


def fit_settings(prefix, extra=None):
    """Fitting choices, shown in the left panel."""
    opts = dict(labdata.FITS)
    if extra:
        opts.update(extra)
    how = w.choose(f"{prefix}_fit", "How k is fitted", opts, "y_tau",
                   help="y is the conversion function that equals k·CA0^(n−1)·τ for this reactor. The lab reports fit y against τ through the origin. "
                        "The manual plots τ against y; the two give slightly different k when the points scatter. "
                        "A free intercept reads a constant offset as extra reaction time.")
    order = w.choose(f"{prefix}_order", "Rate law used for the fit", {2.0: "Second order, −rA = k CA CB (manual)", 1.0: "First order, −rA = k CA",
                                                                      1.5: "Order 1.5, −rA = k CA^1.5", 2.5: "Order 2.5", 3.0: "Order 3"}, 2.0,
                     help="Saponification is second order (first in each reactant). The other orders are there to see how strongly the answer depends "
                          "on that assumption; the 'Fit details' tab compares how well each order explains your readings.")
    ax = w.choose(f"{prefix}_arrax", "Arrhenius plot", {"lnk": "ln k against 1/T", "neg": "−ln k against 1/T"}, "lnk",
                  help="The same fit drawn two ways. Some reports plot −ln k so the slope Ea/R is positive.")
    return {"how": how, "order": float(order), "ax": ax}


def _arr(kt):
    ok = kt[np.isfinite(kt["k"]) & (kt["k"] > 0)] if len(kt) else kt
    return (arrhenius(ok["T_C"].values, ok["k"].values) if len(ok) >= 2 else None), ok


def analyse(runs, kind, how, order, model_fit=None):
    """Everything the result tabs show, computed once."""
    res = {"kind": kind, "how": how, "order": order, "runs": runs, "kt": pd.DataFrame(), "arr": None, "notes": [], "orders": None, "per_order": {}}
    if runs.empty or not np.isfinite(runs["X"]).any():
        return res
    equi = bool(np.allclose(runs["M"], 1.0))
    if not equi and abs(order - 2) > 1e-12:
        res["notes"].append({"level": "info", "text": "Orders other than 2 assume CA = CB, which needs an equimolar feed. Second order is used."})
        order = res["order"] = 2.0
    if how == "model":
        if abs(order - 2) > 1e-12:
            res["notes"].append({"level": "info", "text": "Least squares with the mixing model uses the second-order rate law."})
            order = res["order"] = 2.0
        r = labdata.with_order(runs, kind, 2.0)
        kt, r2, notes = model_fit(r)
        kt = kt.assign(slope=kt["k"] * r.groupby("T_C")["scale"].mean().reindex(kt["T_C"]).values, intercept=0.0)
        r = r2.assign(order=2.0, scale=r["scale"].values, y=r["y"].values, k_point=r["k_point"].values) if len(r2) == len(r) else r2
        res["notes"] += [{"level": "info", "text": n} for n in notes]
    else:
        r = labdata.with_order(runs, kind, order)
        kt = labdata.fit_lines(r, how)
        r = r.assign(X_model=predicted_X(r, kt, kind, order))
    res.update(runs=r, kt=kt)
    res["arr"], _ = _arr(kt)
    res["notes"] += fit_notes(r, kt, how)
    bad = r[~np.isfinite(r["y"])]
    if len(bad):
        res["notes"].insert(0, {"level": "warn", "text": f"Run(s) {', '.join(str(int(v)) for v in bad['run'])} give no usable conversion (X ≤ 0 or X ≥ 1), so they are left "
                                                          "out of the fit. X ≤ 0 means the outlet reading is not below Λ0: check Λ0 for that temperature."})
    if equi:
        rows = []
        for n in ORDERS:
            ro = labdata.with_order(runs, kind, n)
            kto = labdata.fit_lines(ro, how if how != "model" else "y_tau")
            Xm = predicted_X(ro, kto, kind, n)
            ok = np.isfinite(Xm) & np.isfinite(ro["X"].values)
            rmse = float(np.sqrt(np.mean((Xm[ok] - ro["X"].values[ok]) ** 2))) if ok.any() else np.nan
            a, _ = _arr(kto)
            res["per_order"][n] = {"kt": kto, "arr": a}
            rows.append({"order": n, "rate law": "k CA CB" if n == 2 else f"k CA^{n:g}", "RMS error in X": rmse,
                         "Ea (kJ/mol)": a["Ea_kJ"] if a else np.nan, "Arrhenius R²": a["r2"] if a else np.nan,
                         "k units": k_unit(n)})
        res["orders"] = pd.DataFrame(rows)
    return res


def predicted_X(runs, kt, kind, order=2.0):
    if kt.empty:
        return np.full(len(runs), np.nan)
    out = []
    for rec in runs.itertuples():
        r = kt[kt["T_C"] == rec.T_C]
        if not len(r) or not np.isfinite(r["k"].iloc[0]):
            out.append(np.nan)
            continue
        r = r.iloc[0]
        scale = getattr(rec, "scale", rec.CA0)
        icpt = r["intercept"] if "intercept" in r and np.isfinite(r["intercept"]) else 0.0
        out.append(labdata.X_of_y(r["k"] * scale * rec.tau + icpt, rec.M, kind, order))
    return np.array(out, dtype=float)


def k_at(res, T, order=None):
    """k at temperature T from the Arrhenius fit (or the single measured k)."""
    order = res["order"] if order is None else order
    src = res if order == res["order"] else res["per_order"].get(order, {})
    arr, kt = src.get("arr"), src.get("kt", pd.DataFrame())
    if arr is not None:
        return float(arr["A"] * np.exp(-arr["Ea_J"] / (R_GAS * c_to_k(T))))
    ok = kt[np.isfinite(kt["k"])] if len(kt) else kt
    return float(ok["k"].iloc[0]) if len(ok) else np.nan


def publish(res, source, detail):
    if abs(res["order"] - 2) > 1e-12 or res["kt"].empty:
        return
    arr, ok = _arr(res["kt"])
    if arr is not None:
        w.publish(source, k_at(res, 30.0), 30.0, arr["Ea_J"], f"{detail}, {len(ok)} temperatures")
    elif len(ok) == 1:
        w.publish(source, float(ok["k"].iloc[0]), float(ok["T_C"].iloc[0]), 0.0, f"{detail}, one temperature (Ea not measured)")


# ------------------------------------------------------------------ views

def _top(*arrs, default=1.0):
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrs]) if arrs else np.array([])
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) and v.max() > 0 else default


def y_label(kind, order, M_equal=True):
    if abs(order - 2) < 1e-12:
        if kind == "cstr":
            return "X/(1−X)²" if M_equal else "X/((1−X)(M−X))"
        return "X/(1−X)" if M_equal else "ln((M−X)/(M(1−X)))/(M−1)"
    if kind == "cstr":
        return f"X/(1−X)^{order:g}"
    return "−ln(1−X)" if abs(order - 1) < 1e-12 else f"((1−X)^{1 - order:g} − 1)/{order - 1:g}"


def observations(runs: pd.DataFrame, kind: str, order: float = 2.0) -> pd.DataFrame:
    yl = y_label(kind, order, bool(np.allclose(runs["M"], 1.0)))
    return pd.DataFrame({
        "Run": runs["run"], "T (°C)": runs["T_C"], "Total flow (LPH)": runs["Q_LPH"], "τ (min)": runs["tau"],
        "Λ0": runs["lam0"], "Λ∞": runs["lam_inf"], "Λ average": runs["lam_avg"], "Λ steady (used)": runs["lam_ss"],
        "X": runs["X"], f"y = {yl}": runs["y"], f"k from this run ({k_unit(order)})": runs["k_point"],
    })


def show_observations(res, key):
    runs = res["runs"]
    show = observations(runs, res["kind"], res["order"])
    st.dataframe(show, hide_index=True, width="stretch", key=key,
                 column_config={c: st.column_config.NumberColumn(format="%.5g") for c in show.columns[1:]})
    st.caption("A blank y or k means that run's conversion is not between 0 and 1, so it cannot be used. "
               "k from this run = y/(CA0^(n−1) τ): what one run alone says.")


def summary_kpis(res):
    """Headline numbers from the readings."""
    kt, arr, order = res["kt"], res["arr"], res["order"]
    unit = k_unit(order)
    items = []
    if arr is not None:
        items += [{"label": "Activation energy Ea", "value": w.fmt(arr["Ea_kJ"], ".2f"), "unit": "kJ/mol", "hi": True,
                   "detail": f"95%: {arr['Ea_kJ_low']:.1f} to {arr['Ea_kJ_high']:.1f}" if arr["n"] > 2 else "two temperatures: no interval"},
                  {"label": "Pre-exponential A", "value": f"{arr['A']:.4g}", "unit": unit, "detail": f"Arrhenius R² = {arr['r2']:.4f}"}]
    for r in kt.itertuples():
        items.append({"label": f"k at {r.T_C:g} °C", "value": w.fmt(r.k, ".4g"), "unit": unit,
                      "detail": f"95%: {w.fmt(r.k_lo, '.3g')} to {w.fmt(r.k_hi, '.3g')}" if np.isfinite(r.k_lo) else f"{int(r.n)} run(s)"})
    if not items:
        return False
    w.kpis(items)
    return True


def k_table(res):
    kt, how = res["kt"], res["how"]
    unit = k_unit(res["order"])
    cols = {"T_C": "T (°C)", "n": "runs", "slope": "slope", "intercept": "intercept", "delay_min": "implied delay (min)",
            "k": f"k ({unit})", "k_lo": "k low (95%)", "k_hi": "k high (95%)", "r2": "R²", "beta": "β"}
    keep = ["T_C", "n", "slope"] + (["intercept", "delay_min"] if how == "free" else []) + ["k", "k_lo", "k_hi", "r2"]
    if how == "model":
        keep = ["T_C", "n", "k", "k_lo", "k_hi", "beta"]
    show = kt[[c for c in keep if c in kt]].rename(columns=cols)
    st.dataframe(show, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.5g") for c in show.columns})
    if how in ("y_tau", "tau_y"):
        st.caption("R² for a line forced through the origin is measured from zero, as a spreadsheet trendline reports it. "
                   "It is always high; read the scatter on the plots too.")


def _design(res, temps, title, height):
    runs, kt, kind, how, order = res["runs"], res["kt"], res["kind"], res["how"], res["order"]
    yl = y_label(kind, order, bool(np.allclose(runs["M"], 1.0)))
    flip = how == "tau_y"
    xt, yt = (f"y = {yl}", "τ (min)") if flip else ("τ (min)", f"y = {yl}")
    f = ch.fig(title, xt, yt, height=height)
    for T in temps:
        i = sorted(runs["T_C"].unique()).index(T)
        col = SERIES[i % len(SERIES)]
        g = runs[(runs["T_C"] == T) & np.isfinite(runs["y"])]
        if g.empty:
            continue
        x, y = g["tau"].values, g["y"].values
        if flip:
            ch.experiment(f, y, x, f"{T:g} °C", color=col)
        else:
            ch.experiment(f, x, y, f"{T:g} °C", color=col)
        r = kt[kt["T_C"] == T]
        if len(r) and np.isfinite(r["slope"].iloc[0]):
            s, c = float(r["slope"].iloc[0]), float(r["intercept"].iloc[0]) if np.isfinite(r["intercept"].iloc[0]) else 0.0
            if flip:
                yy = np.linspace(0, _top(y) * 1.15, 20)
                ch.model(f, yy, s * yy, f"τ = {s:.4g}·y", color=col, dash="dot", width=2)
            else:
                tt = np.linspace(0, _top(x) * 1.15, 20)
                ch.model(f, tt, s * tt + c, f"y = {s:.4g}·τ" + (f" {c:+.4g}" if how == "free" else ""), color=col, dash="dot", width=2)
    return f


def _arrhenius_fig(res, ax):
    arr, kt = res["arr"], res["kt"]
    ok = kt[np.isfinite(kt["k"]) & (kt["k"] > 0)]
    sgn = -1.0 if ax == "neg" else 1.0
    inv = np.asarray(arr["invT"])
    f = ch.fig(f"Arrhenius: Ea = {arr['Ea_kJ']:.2f} kJ/mol, A = {arr['A']:.4g}", "1/T (1/K)", "−ln k" if sgn < 0 else "ln k", height=330)
    ch.experiment(f, inv, sgn * np.asarray(arr["lnk"]), "k per temperature", text=[f"{T:g} °C" for T in ok["T_C"]])
    xx = np.linspace(min(inv), max(inv), 20)
    ch.model(f, xx, sgn * (arr["intercept"] + arr["slope"] * xx), f"fit, R² = {arr['r2']:.4f}")
    f.update_xaxes(tickformat=".5f")
    return f


def _joined(f, x, y, name, col, err=None):
    """Points joined by a dotted line, one legend entry."""
    ch.model(f, x, y, name, color=col, width=1.6, dash="dot")
    f.data[-1].update(showlegend=False, legendgroup=name)
    ch.experiment(f, x, y, name, color=col, err=err)
    f.data[-1].update(legendgroup=name)


def _trend_figs(runs):
    out = {}
    f = ch.fig("Conversion against total flow", "total flow (LPH)", "X", height=320)
    for i, (T, g) in enumerate(runs.groupby("T_C")):
        g = g.sort_values("Q_LPH")
        _joined(f, g["Q_LPH"], g["X"], f"{T:g} °C", SERIES[i % len(SERIES)], err=g["se_X"].fillna(0).values)
    out["Conversion against flow"] = f
    f = ch.fig("Conversion against temperature", "T (°C)", "X", height=320)
    for i, (Q, g) in enumerate(runs.groupby(runs["Q_LPH"].round(3))):
        g = g.sort_values("T_C")
        _joined(f, g["T_C"], g["X"], f"{Q:g} LPH", SERIES[i % len(SERIES)])
    out["Conversion against temperature"] = f
    f = ch.fig("Rate constant from each run alone", "τ (min)", "k = y/(CA0^(n−1) τ)", height=320)
    for i, (T, g) in enumerate(runs.groupby("T_C")):
        g = g.sort_values("tau")
        _joined(f, g["tau"], g["k_point"], f"{T:g} °C", SERIES[i % len(SERIES)])
    out["Single-run k"] = f
    return out


def _parity_figs(runs):
    out = {}
    f = ch.fig("Model against measurement", "measured X", "model X", height=320)
    lim = [0, _top(runs["X"], runs["X_model"], default=0.1) * 1.08]
    ch.ideal(f, lim, lim, "Perfect agreement")
    ch.experiment(f, runs["X"], runs["X_model"], "Runs", text=[f"{T:g} °C, {Q:g} LPH" for T, Q in zip(runs["T_C"], runs["Q_LPH"])])
    out["Parity plot"] = f
    f = ch.fig("Residuals (measured − model)", "run: T (°C) / flow (LPH)", "ΔX", height=320)
    ch.residuals(f, [f"{T:g}/{Q:g}" for T, Q in zip(runs["T_C"], runs["Q_LPH"])], np.nan_to_num((runs["X"] - runs["X_model"]).values))
    f.update_xaxes(type="category")
    out["Residuals"] = f
    return out


def _grid(figs_list, prefix, per_row=2):
    for start in range(0, len(figs_list), per_row):
        cols = st.columns(per_row)
        for j, (name, f) in enumerate(figs_list[start:start + per_row]):
            with cols[j]:
                rows = (sum(1 for t in f.data if t.showlegend is not False) + 1) // 2
                f.update_layout(height=(f.layout.height or 320) + 20 * rows,
                                legend=dict(orientation="h", yref="container", y=0.01, yanchor="bottom", xanchor="left", x=0, font=dict(size=11)),
                                margin=dict(b=58 + 20 * rows))
                ch.show(f, key=f"{prefix}_g_{start + j}")


def results_tab(res, prefix, ax, figs, reactor, issues=(), inputs=""):
    """Headline numbers, every plot, and what the fitted kinetics predict."""
    runs, kt = res["runs"], res["kt"]
    if inputs:
        w.equation(f"Inputs used: {inputs}")
    if issues:
        w.note(f"{len(issues)} problem(s) in the readings table. The results below use the table as it is, so fix these first:", "err")
        for msg in issues:
            w.note(msg, "warn")
    if runs.empty:
        w.empty("No runs yet", "Type the runs and the Λ0 for each temperature below. Ea, k at every temperature and all plots appear here.")
        return
    if kt.empty or not np.isfinite(kt["k"]).any():
        w.empty("No rate constant yet", "None of the runs gives a conversion between 0 and 1. Check Λ0 for each temperature and the Λ∞ setting.")
        w.notes(res["notes"])
        return
    law = "second order, −rA = k CA CB" if res["order"] == 2 else f"order {res['order']:g}, −rA = k CA^{res['order']:g}"
    w.panel_title(f"Kinetics from your readings · {law} · {labdata.FITS.get(res['how'], 'least squares with the mixing model')}")
    summary_kpis(res)
    w.notes([n for n in res["notes"] if n["level"] == "warn"])
    temps = sorted(runs["T_C"].unique())
    plots = []
    for T in temps:
        r = kt[kt["T_C"] == T]
        k = float(r["k"].iloc[0]) if len(r) else np.nan
        r2 = float(r["r2"].iloc[0]) if len(r) and np.isfinite(r["r2"].iloc[0]) else np.nan
        plots.append((f"Design plot {T:g} °C", _design(res, [T], f"{T:g} °C: k = {w.fmt(k, '.4g')}, R² = {w.fmt(r2, '.4f')}", 300)))
    plots.append(("Design plot", _design(res, temps, "Design plot, all temperatures", 300)))
    if res["arr"] is not None:
        plots.append(("Arrhenius plot", _arrhenius_fig(res, ax)))
    plots += list(_trend_figs(runs).items()) + list(_parity_figs(runs).items())
    for name, f in plots:
        figs[name] = f
    _grid(plots, f"{prefix}_res", per_row=3 if len(plots) > 6 else 2)
    if res["arr"] is None:
        w.note("Add runs at a second temperature to get Ea and A.", "info")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    predictions(res, prefix, figs, reactor)


def predictions(res, prefix, figs, reactor):
    """Use the fitted k(T) to show how conversion moves with τ, temperature, concentration and the rate law."""
    runs, kind, order = res["runs"], res["kind"], res["order"]
    w.panel_title("What your fitted kinetics predict")
    st.caption(f"Ideal {reactor} with the k(T) fitted above. Change temperature, concentration and τ to see each effect on its own.")
    temps = sorted(runs["T_C"].unique())
    c1, c2, c3 = st.columns(3)
    with c1:
        T = w.num(f"{prefix}_prT", "Temperature (°C)", 0.0, 100.0, float(round(np.mean(temps))), 1.0, help="Temperature for the predictions.")
    with c2:
        CA0 = w.num(f"{prefix}_prC", "CA0 (mol/L)", 0.001, 2.0, float(runs["CA0"].mean()), 0.005, fmt="%.4f",
                    help="NaOH after mixing. For a second-order reaction a higher concentration reacts faster; for first order it makes no difference to X.")
    with c3:
        tau = w.num(f"{prefix}_prTau", "τ (min)", 0.01, 500.0, float(np.median(runs["tau"])), 0.1, help="Residence time for the concentration and order plots.")
    M = float(runs["M"].mean())
    k = k_at(res, T)

    def X_of(kk, c, tt, n=order):
        return labdata.X_of_y(kk * c ** (n - 1) * tt, M, kind, n)

    w.kpis([{"label": f"k at {T:g} °C", "value": w.fmt(k, ".4g"), "unit": k_unit(order), "hi": True,
             "detail": "from the Arrhenius fit" if res["arr"] is not None else "single temperature, so the same k at every T"},
            {"label": "Predicted X", "value": w.fmt(X_of(k, CA0, tau), ".3f"), "detail": f"τ = {tau:g} min, CA0 = {CA0:g} mol/L"},
            {"label": "Damköhler number", "value": w.fmt(k * CA0 ** (order - 1) * tau, ".3g"), "detail": "k·CA0^(n−1)·τ"}])
    tmax = max(float(runs["tau"].max()) * 2.5, tau * 1.5)
    tt = np.linspace(tmax / 200, tmax, 120)
    f1 = ch.fig("Conversion against τ", "τ (min)", "X", height=330)
    for i, Tm in enumerate(temps):
        col = SERIES[i % len(SERIES)]
        km = k_at(res, Tm)
        c_run = float(runs.loc[runs["T_C"] == Tm, "CA0"].mean())
        ch.model(f1, tt, [X_of(km, c_run, t) for t in tt], f"{Tm:g} °C (fit)", color=col, width=2)
        g = runs[runs["T_C"] == Tm]
        ch.experiment(f1, g["tau"], g["X"], f"{Tm:g} °C readings", color=col)
    ch.model(f1, tt, [X_of(k, CA0, t) for t in tt], f"{T:g} °C, CA0 {CA0:g}", color=CYAN, dash="dash", width=3)
    figs["Predicted conversion against τ"] = f1
    lo, hi = min(temps) - 15, max(temps) + 15
    TT = np.linspace(lo, hi, 60)
    f2 = ch.fig("Rate constant against temperature", "T (°C)", f"k ({k_unit(order)})", height=330)
    ch.model(f2, TT, [k_at(res, x) for x in TT], "Arrhenius fit" if res["arr"] is not None else "single k")
    ok = res["kt"][np.isfinite(res["kt"]["k"])]
    ch.experiment(f2, ok["T_C"], ok["k"], "fitted at each temperature",
                  err=[(h - l) / 2 if np.isfinite(h) and np.isfinite(l) else 0 for l, h in zip(ok["k_lo"], ok["k_hi"])])
    ch.point(f2, T, k, f"{T:g} °C", color=AMBER)
    figs["k against temperature"] = f2
    CC = np.linspace(CA0 * 0.1, CA0 * 4, 60)
    f3 = ch.fig(f"Conversion against CA0 at {T:g} °C, τ = {tau:g} min", "CA0 (mol/L)", "X", height=330)
    if res["per_order"]:
        for i, n in enumerate(ORDERS):
            kn = k_at(res, T, n)
            if np.isfinite(kn):
                ch.model(f3, CC, [X_of(kn, c, tau, n) for c in CC], f"order {n:g}" + (" (in use)" if n == order else ""),
                         color=SERIES[i % len(SERIES)], width=3 if n == order else 1.6, dash=None if n == order else "dot")
    else:
        ch.model(f3, CC, [X_of(k, c, tau) for c in CC], f"order {order:g}")
    ch.vline(f3, float(runs["CA0"].mean()), "CA0 of your runs", color=AMBER)
    figs["Conversion against CA0"] = f3
    _grid([("a", f1), ("b", f2), ("c", f3)], f"{prefix}_pred", per_row=3)
    st.caption("Each order is fitted to your own readings, so all of them pass close to your runs at the CA0 you used. "
               "Away from it they separate: first order does not depend on CA0 at all, higher orders depend on it more and more. "
               "Running at a second concentration is how the order is told apart.")


def details_tab(res, prefix):
    runs = res["runs"]
    if runs.empty or res["kt"].empty:
        w.empty("Nothing fitted yet", "Fit details appear once at least one run gives a usable conversion.")
        return
    k_table(res)
    w.notes(res["notes"])
    if res["orders"] is not None:
        w.panel_title("Which rate law fits the readings best")
        o = res["orders"]
        st.dataframe(o, hide_index=True, width="stretch", key=f"{prefix}_orders",
                     column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ("order", "RMS error in X", "Ea (kJ/mol)", "Arrhenius R²")})
        ok = o.dropna(subset=["RMS error in X"])
        if len(ok):
            best = ok.loc[ok["RMS error in X"].idxmin()]
            msg = (f"Smallest error: order {best['order']:g}. With every run at one CA0 the orders differ only through how X changes with τ, "
                   "so the difference between them is usually small; a run at a second concentration separates them clearly.")
            if best["order"] != 2 and runs["CA0"].nunique() == 1:
                msg += (" Anything that makes the single-run k drift with τ (a Λ∞ that is off, a delay before reading, runs not yet steady) "
                        "also favours a different order here, so this is not evidence against second order on its own.")
            st.caption(msg)
    lam_inf_check(runs, res["kind"], f"{prefix}_lchk")


def fit_notes(runs, kt, how):
    out = []
    for T, g in runs.groupby("T_C"):
        g = g[np.isfinite(g["k_point"])].sort_values("tau")
        if len(g) >= 3:
            lo, hi = g["k_point"].min(), g["k_point"].max()
            if hi > 1.3 * lo:
                trend = "falls" if g["k_point"].iloc[-1] < g["k_point"].iloc[0] else "rises"
                out.append({"level": "info", "text": f"At {T:g} °C the single-run k {trend} from {g['k_point'].iloc[0]:.3g} to {g['k_point'].iloc[-1]:.3g} "
                                                     "as τ grows. For ideal kinetics it would stay flat. A Λ∞ that is off, a delay before reading, "
                                                     "or runs that had not settled all do this."})
    if how == "free" and len(kt):
        for r in kt[~np.isfinite(kt["k"])].itertuples():
            out.append({"level": "warn", "text": f"At {r.T_C:g} °C the free line has a negative slope: conversion does not rise with τ in these readings. "
                                                 "k cannot be read from the slope; use a line through the origin, or check those runs."})
        for r in kt[np.isfinite(kt["delay_min"])].itertuples():
            if abs(r.delay_min) > 0.5 * float(runs.loc[runs["T_C"] == r.T_C, "tau"].min()):
                out.append({"level": "info", "text": f"At {r.T_C:g} °C the intercept is equivalent to {r.delay_min:.2f} min of extra reaction time, "
                                                     "as large as the shortest τ. With three flows the intercept is poorly fixed."})
    if "spread" in runs:
        wide = runs[runs["spread"] > 0.08 * runs["lam_avg"]]
        if len(wide):
            each = "; ".join(f"run {r.run} ({r.T_C:g} °C, {r.Q_LPH:g} LPH) {r.spread:.2f} mS/cm, {r.spread / r.lam_avg:.0%}" for r in wide.itertuples())
            out.append({"level": "info", "text": f"The readings within a run scatter by more than 8% of their mean in {len(wide)} of {len(runs)} runs: {each}. "
                                                 "Those points are the least certain; more readings per run, or waiting longer before reading, would help."})
    return out


def lam_inf_check(runs, kind, key):
    """Λ∞ read from the readings, against the ion estimate and the value in use."""
    with st.expander("Check Λ∞ against the readings", expanded=False):
        st.caption("Every run at one temperature should give the same k. Fitting one Λ∞/Λ0 shared by all runs, and one k per temperature, "
                   "finds the Λ∞ that makes the readings at different flows agree with each other. It needs at least two flows per temperature.")
        res = labdata.fit_lam_inf(runs, kind)
        if res is None:
            w.empty("Not enough runs", "Needs at least two flows at every temperature and more runs than unknowns.")
            return None
        used = float(np.nanmean(runs["lam_inf"] / runs["lam0"]))
        w.kpis([{"label": "Λ∞/Λ0 from the readings", "value": w.fmt(res["ratio"], ".3f"),
                 "detail": f"95%: {res['lo']:.3f} to {res['hi']:.3f}" if np.isfinite(res["lo"]) else "no interval", "hi": True},
                {"label": "From ion conductivities", "value": "0.366", "detail": "Na⁺ + acetate⁻ over Na⁺ + OH⁻ at 25 °C"},
                {"label": "In use now", "value": w.fmt(used, ".3f"), "detail": "average Λ∞/Λ0 over the runs"},
                {"label": "Fit residual", "value": w.fmt(res["rms_mScm"], ".3f"), "unit": "mS/cm", "detail": f"{res['n']} runs, {res['dof']} spare"}])
        kk = pd.DataFrame({"T (°C)": list(res["k"]), "k with this Λ∞ (L/mol/min)": list(res["k"].values())})
        st.dataframe(kk, hide_index=True, width="stretch", key=f"{key}_tab", column_config={c: st.column_config.NumberColumn(format="%.4g") for c in kk.columns})
        width = res["hi"] - res["lo"] if np.isfinite(res["lo"]) else np.inf
        trusted = not res["at_bound"] and width <= 0.4 and res["snr"] >= 3
        if not trusted:
            w.note(f"The readings do not pin Λ∞ down. Between flows the steady readings change by {res['signal_mScm']:.2f} mS/cm on average, "
                   f"only {res['snr']:.1f} times the {res['rms_mScm']:.2f} mS/cm the fit cannot explain. Keep the ion estimate or a measured Λ∞; "
                   "the k values above are not meaningful.", "warn")
        elif res["lo"] <= 0.366 <= res["hi"]:
            w.note("The readings agree with the ion-conductivity estimate of Λ∞.", "good")
        else:
            w.note(f"The readings put Λ∞ near {res['ratio']:.2f} Λ0, against 0.366 from ion conductivities at 25 °C.", "info")
        if trusted and not (res["lo"] <= used <= res["hi"]):
            w.note(f"The Λ∞ in use ({used:.3f} Λ0) lies outside what the readings support. k and Ea depend strongly on it.", "warn")
        return res


def right_summary(res, issues=(), inputs=""):
    """Compact results from the readings, for the right-hand panel."""
    if res["kt"].empty or not np.isfinite(res["kt"]["k"]).any():
        return
    w.panel_title("From your readings")
    if issues:
        w.note(f"{len(issues)} problem(s) in the readings table; see the Results tab before trusting these numbers.", "warn")
    summary_kpis(res)
    if inputs:
        st.caption(f"Using {inputs}.")
