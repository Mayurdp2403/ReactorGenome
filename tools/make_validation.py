"""Write docs/VALIDATION.md from numbers computed now by the engine.

    python tools/make_validation.py

Part A checks the engine against closed-form results. Part B runs the engine
on the report readings (kept in reactorgenome/data_lab.py, used only here and
in tests) and lists where the reports and the app use different conventions.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from reactorgenome import data_lab as D  # noqa: E402
from reactorgenome.analysis import analyse_pfr  # noqa: E402
from reactorgenome.engine import batch, cstr, flowsheet, labdata, pfr, rtd  # noqa: E402
from reactorgenome.engine.rate import Kinetics  # noqa: E402
from reactorgenome.kinetics import cstr_X, pfr_X  # noqa: E402
import synthetic  # noqa: E402

KIN = Kinetics(k_ref=6.0, T_ref_C=30.0, Ea=45e3)
out: list[str] = []


def h(text, level=2):
    out.append(f"\n{'#' * level} {text}\n")


def p(text):
    out.append(text + "\n")


def table(head, rows, fmt=None):
    fmt = fmt or {}
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            f = fmt.get(i, ".4g")
            cells.append(f"{v:{f}}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("–" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")


# ---------------------------------------------------------------------------- A. analytic
def part_a():
    h("A. Engine against closed-form results")
    p("Every row is recomputed when this file is generated. *Error* is |engine − exact|.")

    h("A1. Ideal stirred tank", 3)
    rows = []
    for Da, M in [(0.5, 1.0), (1.0, 1.0), (3.0, 1.0), (1.0, 2.0)]:
        CA0 = 0.05
        tau = Da / (KIN.k_C(30) * CA0)
        m = cstr.CSTRModel(V=tau * 0.4, qA=0.2, qB=0.2, cA_feed=2 * CA0, cB_feed=2 * CA0 * M, T_C=30, kin=KIN)
        X = cstr.steady_isothermal(m)["X"]
        exact = float(cstr_X(Da, M))
        rows.append([Da, M, X, exact, abs(X - exact)])
    table(["Da", "M", "engine X", "closed form X", "error"], rows, {4: ".1e"})

    h("A2. Transient settles on the algebraic steady state", 3)
    rows = []
    for name, kw in [("one tank", {}), ("3 tanks", {"n_tanks": 3}), ("bypass 0.2", {"bypass": 0.2}),
                     ("dead volume 0.25", {"stagnant": 0.25}), ("stagnant zone with exchange", {"stagnant": 0.2, "exchange": 0.3})]:
        m = cstr.CSTRModel(V=2.15, qA=0.2, qB=0.2, cA_feed=0.1, cB_feed=0.1, T_C=30, kin=KIN, **kw)
        ss = cstr.steady_state(m)
        sim = cstr.simulate(m, 60 * m.tau, 401, init="water")
        rows.append([name, ss["X"], sim["X_out"][-1], abs(ss["X"] - sim["X_out"][-1])])
    table(["mixing model", "steady X", "X after 60 τ", "error"], rows, {3: ".1e"})

    h("A3. Tanks in series approach plug flow", 3)
    Da = 1.2
    rows = []
    for N in (1, 2, 5, 10, 40):
        m = cstr.CSTRModel(V=Da / (KIN.k_C(30) * 0.05) * 0.4, qA=0.2, qB=0.2, cA_feed=0.1, cB_feed=0.1, T_C=30, kin=KIN, n_tanks=N)
        X = cstr.steady_isothermal(m)["X"]
        rows.append([N, X, float(pfr_X(Da)) - X])
    table(["N", "X", "plug-flow X − X"], rows, {2: ".4f"})
    p(f"Plug flow at Da = {Da}: X = {float(pfr_X(Da)):.4f}. The gap falls as 1/N (10 → 40 tanks divides it by about 4).")

    h("A4. Tubular reactor", 3)
    rows = []
    for flow, Pe in [("plug", 0), ("dispersion", 1000), ("dispersion", 0.05), ("laminar", 0)]:
        m = pfr.PFRModel(V=0.5, Q=0.5, CA0=0.05, CB0=0.05, T_C=30, kin=KIN, flow=flow, Pe=Pe or 50)
        X = pfr.solve(m)["X_out"]
        Da = KIN.k_C(30) * 0.05 * 1.0
        ref = {"plug": float(pfr_X(Da)), "laminar": np.nan}.get(flow, float(pfr_X(Da)) if Pe > 100 else float(cstr_X(Da)))
        lab = {"plug": "plug, Da/(1+Da)", "laminar": "laminar (below plug)"}.get(flow, f"dispersion Pe = {Pe:g} → {'plug' if Pe > 100 else 'stirred tank'}")
        rows.append([lab, X, ref, abs(X - ref) if np.isfinite(ref) else np.nan])
    table(["case", "engine X", "limit", "error"], rows, {3: ".1e"})

    h("A5. Flowsheet solver", 3)
    feed = {"id": "f", "kind": "feed", "Q": 0.4, "CA": 0.05, "CB": 0.05, "T": 30}
    prod = {"id": "o", "kind": "product"}
    rows = []
    series = {"nodes": [feed, {"id": "c", "kind": "cstr", "V": 2.0, "T": 30}, {"id": "p", "kind": "pfr", "V": 0.5, "T": 30}, prod],
              "edges": [{"from": "f", "to": "c"}, {"from": "c", "to": "p"}, {"from": "p", "to": "o"}]}
    r = flowsheet.solve(series, KIN)
    m1 = cstr.CSTRModel(V=2.0, qA=0.2, qB=0.2, cA_feed=0.1, cB_feed=0.1, T_C=30, kin=KIN)
    c1 = cstr.steady_isothermal(m1)["c_out"]
    X2 = pfr.solve(pfr.PFRModel(V=0.5, Q=0.4, CA0=c1[0], CB0=c1[1], T_C=30, kin=KIN))
    X_seq = 1 - X2["c_out"][0] / 0.05
    rows.append(["CSTR → PFR vs sequential", r["X"], X_seq, abs(r["X"] - X_seq), r["balance_error"], r["iterations"]])
    rec = {"nodes": [feed, {"id": "c", "kind": "cstr", "V": 2.0, "T": 30}, {"id": "s", "kind": "split", "f": 0.5}, prod],
           "edges": [{"from": "f", "to": "c"}, {"from": "c", "to": "s"}, {"from": "s", "to": "o", "port": "out"}, {"from": "s", "to": "c", "port": "side"}]}
    r2 = flowsheet.solve(rec, KIN)
    X1 = cstr.steady_isothermal(m1)["X"]
    rows.append(["recycle round one CSTR (no change expected)", r2["X"], X1, abs(r2["X"] - X1), r2["balance_error"], r2["iterations"]])
    table(["network", "flowsheet X", "reference X", "error", "mass balance error", "passes"], rows, {3: ".1e", 4: ".1e"})

    h("A6. Residence time distribution", 3)
    t = np.linspace(0, 80, 1601)
    r = rtd.analyse(t, rtd.E_cstr(t, 4.0), 0.0, V=1.0, Q=0.25)
    rt = rtd.analyse(t, rtd.E_tanks(t, 4.0, 3.0), 0.0)
    table(["curve", "tm (exact)", "tm", "σ² (exact)", "σ²", "N from moments"],
          [["ideal tank, τ = 4", 4.0, r["tm"], 16.0, r["var"], rtd.N_moments(r)], ["3 tanks, tm = 4", 4.0, rt["tm"], 16 / 3, rt["var"], rtd.N_moments(rt)]])

    h("A7. Steady value from four early readings", 3)
    p("Readings at 5, 10, 15 and 20 min after starting a tank full of water, generated by the engine with k = 6 L/mol/min at 30 °C, "
      "Ea = 45 kJ/mol and 0.3% noise. The k fitted from the resulting steady conversions (manual design plot) is compared with the k used.")
    df = synthetic.cstr_table()
    rows = []
    k40 = KIN.k_C(40)
    for meth in ("model_shape", "model_tau", "free_tau", "last", "mean_after_first", "mean_all"):
        rr = labdata.cstr_runs(df, V=2.0, cA=0.1, cB=0.1, kin=KIN, method=meth, start="water")
        kd = labdata.fit_k_design(rr).set_index("T_C")["k"]
        rows.append([labdata.steady.METHODS[meth], kd[30.0], 100 * (kd[30.0] / 6 - 1), kd[40.0], 100 * (kd[40.0] / k40 - 1)])
    table(["steady value from", "k 30 °C", "error %", "k 40 °C", "error %"], rows, {2: "+.1f", 4: "+.1f"})
    p("A start from water excites two modes: total sodium washing in on τ and the faster reaction mode. One exponential absorbs part of the "
      "fast mode into its plateau; the simulated start-up shape carries both, which is why it is the default.")


# ---------------------------------------------------------------------------- B. reports
def part_b():
    h("B. Engine on the report readings")
    p("The report readings are used here and in `tests/` only; the app never shows them.")

    h("B1. CSTR, Group 8 (manual conventions: drop R0, average R1–R3, Λ∞ = 3.75 mS/cm)", 3)
    df = D.cstr_group8_frame()
    kw = dict(V=2.15, cA=0.07, cB=0.07, kin=Kinetics(k_ref=37.0, T_ref_C=31.0, Ea=11e3), spacing=1.0)
    man = labdata.cstr_runs(df, lam_mode="fixed", lam_fixed=3.75, method="mean_after_first", **kw)
    ion = labdata.cstr_runs(df, lam_mode="ion", lam_fixed=3.75, method="mean_after_first", **kw)
    rows = []
    for (_, a), (_, b) in zip(man.iterrows(), ion.iterrows()):
        key = (int(a["T_C"]), int(round(a["qA_LPH"] + a["qB_LPH"])))
        rows.append([f"{key[0]} °C, {key[1]} LPH", D.CSTR_GROUP8_PUBLISHED["X"][key], a["X"], abs(a["X"] - D.CSTR_GROUP8_PUBLISHED["X"][key]), b["X"]])
    table(["run", "report X", "engine X (same conventions)", "difference", "engine X, Λ∞ from ion conductivities"], rows, {3: ".1e"})
    kd = labdata.fit_k_design(man).set_index("T_C")
    table(["T (°C)", "report k", "engine k (design plot, through origin)"],
          [[f"{T}", D.CSTR_GROUP8_PUBLISHED["k"][T], kd.loc[float(T), "k"]] for T in (31, 41, 50)], {1: ".4f", 2: ".4f"})
    p("k does not rise steadily with temperature and the Arrhenius fit in the report has R² = 0.39. The reading times were not recorded, "
      "so 1 min spacing is assumed below. The app's default rule (simulated start-up shape, last reading within 0.01 of the plateau) "
      "gives these verdicts and k values depending on how each run was started:")
    rows = []
    for start, label in (("previous", "after a flow change"), ("water", "from a tank of water")):
        rr = labdata.cstr_runs(df, lam_mode="fixed", lam_fixed=3.75, method="model_shape", start=start, **kw)
        kk = labdata.fit_k_design(rr).set_index("T_C")["k"]
        bad = [f"{int(r.T_C)} °C {int(round(r.qA_LPH + r.qB_LPH))} LPH" for r in rr.itertuples() if r.verdict != "steady"]
        rows.append([label, ", ".join(bad) or "none", kk[31.0], kk[41.0], kk[50.0]])
    table(["start", "runs not steady", "k 31 °C", "k 41 °C", "k 50 °C"], rows, {2: ".2f", 3: ".2f", 4: ".2f"})
    p("Either way k at 41 °C stays below k at 31 °C, so the plateau treatment does not explain the scatter. The ion-based Λ∞ "
      "(last column of the first table) lowers X at 31 °C and raises it at 50 °C, because Λ0 changes with temperature while 3.75 mS/cm does not.")

    h("B2. Timed CSTR (readings at τ, τ+1, τ+2 min, Λ∞ = 0.1 mS/cm)", 3)
    df = D.cstr_timed_frame()
    kw = dict(V=2.15, cA=0.07, cB=0.07, kin=Kinetics(k_ref=5.2, T_ref_C=28.0, Ea=15e3), lam_mode="fixed", lam_fixed=0.1)
    rows = []
    for meth in ("mean_after_first", "last", "free_tau", "model_shape"):
        rr = labdata.cstr_runs(df, method=meth, start="water", **kw)
        X15 = {int(r.T_C): r.X for r in rr.itertuples() if abs(r.qA_LPH + r.qB_LPH - 15) < 1e-9}
        rows.append([labdata.steady.METHODS[meth], X15[28], X15[40], X15[50]])
    rows.insert(0, ["report (mean of all three)", *[D.CSTR_TIMED_PUBLISHED["X15"][T] for T in (28, 40, 50)]])
    table(["steady value from", "X 28 °C", "X 40 °C", "X 50 °C"], rows)
    p("At 15 LPH the 28 °C run was still climbing (4.11 → 4.29 mS/cm). Averaging those readings makes 28 °C look better than 40 °C. "
      "Extrapolating to the plateau removes most of that inversion. This is a steady-state judgement, not a kinetic effect.")

    h("B3. PFR, Group 8 (free line: τ against X/(1−X), intercept = extra reaction time)", 3)
    out_p = analyse_pfr(D.pfr_group8_frame(), V=0.55, cA=0.07, cB=0.07, lam_inf_mode="manual", lam_inf_fixed=0.1, line_mode="free")
    got = {int(r.T_C): r for r in out_p["kinetics"].itertuples()}
    table(["T (°C)", "report k", "app k", "report delay (min)", "app delay (min)"],
          [[f"{T}", D.PFR_GROUP8_PUBLISHED["k"][T], got[T].k, D.PFR_GROUP8_PUBLISHED["delta_min"][T], got[T].delta_min] for T in (40, 50, 60)])
    p(f"Ea: report {D.PFR_GROUP8_PUBLISHED['Ea_kJ']} kJ/mol, app {out_p['arrhenius']['Ea_kJ']:.1f} kJ/mol. "
      "A free intercept of 7–16 min is far longer than τ (0.8–1.7 min). Physically it is reaction continuing in the sample beaker before "
      "the conductivity is read; the PFR page models it as an explicit delay instead of hiding it in the intercept.")

    h("B4. Batch, Group 21, 50.2 °C", 3)
    b = D.BATCH_50
    t = np.array(b["t_s"], float)
    C = batch.conc_from_kappa(np.array(b["ec"], float), b["lam0"], b["lam_inf"], b["C0"])
    w = (t >= b["window"][0]) & (t <= b["window"][1])
    rows = []
    for order in ("first", "second", "nth"):
        f = batch.fit(t[w] / 60, C[w], order, b["C0"], 1.0, fit_t0=True)
        npar = len(f["names"])
        aic = w.sum() * np.log(f["sse"] / w.sum()) + 2 * npar
        k = f["k"] / 60 if order != "first" else f["k"] / 60
        rows.append([order, k, f.get("n", np.nan) if order == "nth" else (1 if order == "first" else 2), f["r2"], aic])
    table(["rate law", "k (per s; L/mol/s for 2nd order)", "order", "R²", "AIC"], rows, {1: ".4g", 3: ".4f", 4: ".1f"})
    p(f"Report: k = {b['k_integral_reported']} L/mol/s (integral method, smoothed curve), order {b['n_reported']} by the differential method. "
      "The trace after the first 10 s fits first order better than second order. The second reading drops from 15430 to 10370 μS/cm in 5 s, "
      "so mixing and the first-reading κ0 dominate the early points; the app fits a clock offset and lets the user choose κ0.")

    h("B5. RTD, Group 21", 3)
    rows = []
    for run in D.RTD_RUNS:
        tt = [q[0] for q in run["points"]]
        ss = [q[1] for q in run["points"]]
        r = rtd.analyse(tt, ss, run["bg"], V=run["V"], Q=run["Q"] / 60)
        pub = D.RTD_PUBLISHED[run["name"]]
        tau = run["V"] / (run["Q"] / 60)
        rows.append([run["name"], pub["tm"], r["tm"], pub["var"], r["var"], rtd.N_moments(r), tau**2 / r["var"],
                     pub.get("Pe", np.nan), rtd.Pe_open_tm(r["v"]), rtd.Pe_open_exact(r["v"]), rtd.Pe_closed(r["v"])])
    table(["run", "report tm", "app tm", "report σ²", "app σ²", "N = tm²/σ²", "N = τ²/σ²", "report Pe", "Pe 2/Pe+8/Pe²", "Pe exact open", "Pe closed"],
          rows, {1: ".3f", 2: ".3f", 3: ".3f", 4: ".3f", 5: ".2f", 6: ".2f", 7: ".1f", 8: ".1f", 9: ".1f", 10: ".1f"})


# ---------------------------------------------------------------------------- E. 2026 reports
def part_e():
    import report_data as R

    from reactorgenome.kinetics import arrhenius

    h("D. The 2026 lab reports (CSTR, PFR, RTD)")
    p("Readings typed exactly as on each report's sheet (total flow, τ, Λ0 once per temperature, every reading) and analysed with the "
      "report's own choices: average of all readings, the report's Λ∞, y against τ through the origin. The data live in "
      "`tests/report_data.py` and are checked by `tests/test_lab_reports.py`; the app never shows them. `lab_report.pdf` is a "
      "CSTR template with no readings; its method (τ against X/(1−X)², Λ∞ = 3.75 mS/cm) is the 'τ against y' fit option.")

    h("D1. CSTR (cstr_lab.pdf: V = 2.15 L, CA0 = 0.05 mol/L, Λ∞ = 0.10 mS/cm)", 3)
    runs = labdata.cstr_runs(R.cstr_frame(), V=R.CSTR_V, CA0=R.CSTR_CA0, kin=KIN, lam_mode="fixed", lam_fixed=R.CSTR_LINF, method="mean_all")
    rows = [[f"{int(r.T_C)} °C, {int(r.Q_LPH)} LPH", r.tau, R.CSTR_X[i], r.X, R.CSTR_KPOINT[i], r.k_point] for i, r in enumerate(runs.itertuples())]
    table(["run", "τ (min)", "report X", "app X", "report k (run)", "app k (run)"], rows, {1: ".2f", 2: ".5f", 3: ".5f", 4: ".3f", 5: ".3f"})
    p("The 31 °C, 20 LPH entry differs in the fourth decimal: its three readings (5.05, 5.05, 5.04) average to X = 0.39305; the report prints "
      "0.39289 but its own k for that temperature matches the app to four figures, so the table entry is a transcription slip.")
    kt = labdata.fit_lines(runs, "y_tau").set_index("T_C")
    arr = arrhenius(kt.index.values, kt["k"].values)
    table(["T (°C)", "report k", "app k", "report R² (through origin)", "app R²"],
          [[f"{int(T)}", R.CSTR_K[T], kt.loc[T, "k"], R.CSTR_R2[T], kt.loc[T, "r2"]] for T in R.CSTR_K], {1: ".4f", 2: ".4f", 3: ".4f", 4: ".4f"})
    p(f"Arrhenius: report Ea = {R.CSTR_EA_KJ} kJ/mol, A = {R.CSTR_A:.3g}; app Ea = {arr['Ea_kJ']:.2f} kJ/mol, A = {arr['A']:.3g} L/mol/min.")

    h("D2. PFR (PFR_Report.pdf: V = 0.55 L, CA0 = 0.035 mol/L, Λ∞ = 0.1 mS/cm, four readings averaged)", 3)
    pr = labdata.pfr_runs(R.pfr_frame(), V=R.PFR_V, CA0=R.PFR_CA0, lam_mode="fixed", lam_fixed=R.PFR_LINF)
    table(["run", "τ (min)", "app X", "report k (run)", "app k (run)"],
          [[f"{int(r.T_C)} °C, {int(r.Q_LPH)} LPH", r.tau, r.X, R.PFR_KPOINT[i], r.k_point] for i, r in enumerate(pr.itertuples())],
          {1: ".3f", 2: ".5f", 3: ".4f", 4: ".4f"})
    fits = {how: labdata.fit_lines(pr, how).set_index("T_C") for how in ("y_tau", "tau_y", "free")}
    table(["T (°C)", "report k", "app k, y on τ (report)", "app k, τ on y (manual)", "app k, free line"],
          [[f"{int(T)}", R.PFR_K[T], fits["y_tau"].loc[T, "k"], fits["tau_y"].loc[T, "k"], fits["free"].loc[T, "k"]] for T in R.PFR_K],
          {1: ".5f", 2: ".5f", 3: ".4f", 4: ".3f"})
    arr = arrhenius(fits["y_tau"].index.values, fits["y_tau"]["k"].values)
    p(f"Arrhenius: report Ea = {R.PFR_EA_KJ} kJ/mol, A = {R.PFR_A}; app Ea = {arr['Ea_kJ']:.3f} kJ/mol, A = {arr['A']:.2f} L/mol/min. "
      "The report's k values carry a rounding in the last digit, which moves A in the fourth figure.")
    p("Regressing τ on y instead of y on τ gives 5–10% larger k from the same points, because the points scatter. A free line fails outright "
      "at 50 °C (conversion does not rise with τ in those readings), so the app flags it rather than report a negative k.")

    h("D3. RTD (RTD_Report.pdf: readings every 0.5 min, baseline = first reading, no tail correction)", 3)
    rows = []
    for name, s in R.RTD_RUNS.items():
        s = np.asarray(s, float)
        c = R.RTD_CAL[0] * s + R.RTD_CAL[1]
        r = rtd.analyse(np.arange(len(s)) * R.RTD_DT, c, c[0])
        sm = rtd.summary(r)
        want = R.RTD_RESULTS[name]
        param = (f"N = {want['N']}", f"N = {sm['N']:.3f} (whole: {sm['N_whole']})") if "N" in want else \
            (f"Pe = {want['Pe']}, τ = {want['tau']}", f"Pe = {sm['Pe']:.2f}, τ = {sm['tau_Pe']:.3f}")
        rows.append([name, want["tm"], sm["tm"], want["var"], sm["var"], param[0], param[1], sm["end_frac"]])
    table(["run", "report tm", "app tm", "report σ²", "app σ²", "report", "app", "last reading / peak"], rows,
          {1: ".3f", 2: ".3f", 3: ".3f", 4: ".3f", 7: ".2f"})
    p("Identical with or without the TDS calibration: a straight-line calibration cancels in E(t) = C/∫C dt. "
      "Both stirred-tank runs give N ≈ 1.5, not 1. The curves stop at about 4% of the peak; an exponential tail holds only 3% of the tracer but raises "
      "σ² by half and brings N to 1.14 and 1.20, close to one ideal tank. The app shows both and warns when the tail moves σ² by more than 10%.")

    h("D4. What the readings say about Λ∞", 3)
    rows = []
    for label, rr, kind in (("CSTR", runs, "cstr"), ("PFR", pr, "pfr")):
        res = labdata.fit_lam_inf(rr, kind)
        rows.append([label, res["ratio"], f"{res['lo']:.3f} to {res['hi']:.3f}", res["rms_mScm"], res["snr"],
                     ", ".join(f"{T:g} °C: {k:.3g}" for T, k in res["k"].items())])
    table(["set", "Λ∞/Λ0 fitted", "95% interval", "residual (mS/cm)", "change between flows ÷ residual", "k with that Λ∞"], rows,
          {1: ".3f", 3: ".3f", 4: ".1f"})
    p("If the reactor is ideal, every run at one temperature shares one k. Fitting a single Λ∞/Λ0 and one k per temperature to all nine "
      "CSTR runs gives Λ∞ ≈ 0.30 Λ0, close to the 0.366 Λ0 expected from ion conductivities (sodium acetate replaces sodium hydroxide) "
      "and far from the 0.1 mS/cm (0.01 Λ0) the report uses. With that Λ∞, k at 31 °C is about 8.5 L/mol/min, in the range usually quoted "
      "for this reaction near 30 °C, where the report's 3.16 is low. The single-run k falling as τ grows in the report (3.38 → 3.00 at 31 °C) "
      "is what a Λ∞ that is too low produces; with Λ∞ = 0.30 Λ0 the same three runs give 8.8, 8.7 and 8.1. The PFR readings scatter by up to 1 mS/cm between repeats, more than they change "
      "between flows, so they cannot fix Λ∞; the app says so instead of reporting the fit.")

    h("D5. Plot coverage", 3)
    table(["report plot", "in the app"], [
        ["CSTR design plot X/(1−X)² against τ, one per temperature", "Kinetics fit → one design plot per temperature, with slope, k and R²"],
        ["CSTR combined design plot", "Kinetics fit → design plot, all temperatures"],
        ["CSTR Arrhenius ln k against 1/T", "Kinetics fit → Arrhenius, ln k or −ln k, 1/T or 1000/T"],
        ["Template: τ against X/(1−X)² (slope 1/(k CA0))", "Fit option 'τ against y through the origin' redraws the design plots that way"],
        ["PFR conversion against total flow", "Trends → conversion against total flow"],
        ["PFR conversion against temperature", "Trends → conversion against temperature"],
        ["PFR X/(1−X) against τ with lines through the origin", "Kinetics fit → design plots"],
        ["PFR Arrhenius −ln k against 1/T", "Kinetics fit → Arrhenius, '−ln k' option"],
        ["RTD calibration TDS against NaCl", "Calibration tab (standards table, fitted line, residuals)"],
        ["RTD curves, stirred tank at two flows", "All runs → E(t), stirred tank"],
        ["RTD curves, packed bed at two flows", "All runs → E(t), packed bed"],
        ["RTD comparison of all four runs", "All runs → tracer curves, E(θ) and F(t) for every run"],
    ])

    h("D6. What the app adds beyond the reports", 3)
    for line in [
        "Observation table for every run (Λ average, steady Λ used, X, y, single-run k), as in each report's Table 2, built as the readings are typed.",
        "Three ways to fit k (y on τ, τ on y, free line) with 95% intervals on k and on Ea; parity and residual plots.",
        "Single-run k against τ: a flat line means the data are consistent; a slope points to Λ∞, a delay before reading, or unsteady runs.",
        "Λ∞ estimated from the readings, with an interval and a check of whether the data can support it.",
        "Steady-state check for CSTR runs when reading times are typed: plateau extrapolation from the simulated start-up and a per-run verdict.",
        "A non-ideal model of each reactor (bypass, dead and stagnant volume, tanks in series, dispersion, laminar flow, recycle, sample delay, "
        "reversible reaction, energy balance) fitted or compared against the same readings.",
        "RTD: all four runs on one sheet, E(θ) overlay, N rounded and unrounded, three Peclet conventions, τ from Pe, D_L from bed geometry, "
        "tail correction with the tail's share of the tracer, model fits, and the conversion each vessel would give.",
        "k and Ea fitted on one rig can be used to predict another rig or a designed network (Design Studio).",
        "Every page exports a report (HTML), the tables (CSV) and every plot, and the whole session saves as a project file.",
    ]:
        p(f"- {line}")


# ---------------------------------------------------------------------------- C. conventions
def part_c():
    h("C. Conventions that differ between the reports, the manuals and the app")
    items = [
        ("Λ∞ (conductivity at complete conversion)", "CSTR manual 3.75 mS/cm; PFR manual 0.1 mS/cm.",
         "Default Λ∞ = 0.366 Λ0 from limiting ionic conductivities (Na⁺ 50.1, OH⁻ 198, OAc⁻ 40.9 S·cm²/mol); a typed constant is available.",
         "X = (Λ0 − Λ)/(Λ0 − Λ∞), so a lower Λ∞ gives a lower X. 0.1 mS/cm is essentially pure water and understates conversion; "
         "3.75 mS/cm matches the ion estimate only near Λ0 ≈ 10 mS/cm (B1)."),
        ("Steady state in the CSTR", "Average of readings after the first, or the last reading.",
         "Plateau fitted from the readings with the simulated start-up shape (default), a single exponential, or the manual average; the user picks the rule that decides 'steady'.",
         "Averages of readings that are still moving bias X toward the start value (B2)."),
        ("Tanks in series from an RTD", "N = τ²/σ² (report) with τ = V/Q.", "N = tm²/σ² by default; τ²/σ² shown next to it.",
         "They differ whenever tm ≠ V/Q, i.e. whenever there is dead volume or a cut-off tail."),
        ("Open-vessel Peclet number", "σ²/tm² = 2/Pe + 8/Pe² (report).",
         "Shown, together with the exact open–open form σ²/tm² = (2/Pe + 8/Pe²)/(1 + 2/Pe)² and the closed form.",
         "The short form treats the measured tm as τ; the exact form accounts for tm = τ(1 + 2/Pe)."),
        ("Closed-vessel Peclet number", "–", "σ²/τ² = 2/Pe − 2(1 − e^−Pe)/Pe².", "An earlier version of the app had + in place of −; corrected and tested."),
        ("PFR intercept", "Free intercept in τ against X/(1−X).", "Modelled as reaction time after the outlet (sample delay).", "See B3."),
        ("Batch order", "Second order assumed; differential method gave n ≈ 1.", "Order chosen by the user, compared by AIC, with a clock offset.", "See B4."),
        ("Settling time", "Not modelled; readings at fixed times (for example τ, τ+1, τ+2 min in the timed CSTR set).",
         "Simulated for the chosen start: from water the slow mode is τ; after a flow change it is τ(1−X)/(1+X).",
         "After a flow change the tank settles faster than τ suggests; from water it needs several τ."),
    ]
    for title, report, app, why in items:
        p(f"**{title}**\n\n- Reports / manuals: {report}\n- App: {app}\n- Why it matters: {why}\n")


def main():
    out.append("# ReactorGenome validation\n")
    p(f"Generated {date.today().isoformat()} by `tools/make_validation.py`. Do not edit by hand; rerun the script.")
    p("Test suite: `python -m pytest tests` (engine analytic cases, units and bounds, invalid inputs, solver failures, networks, exports, "
      "project save/restore, every page rendered with and without readings).")
    part_a()
    part_b()
    part_c()
    part_e()
    h("E. What was not validated")
    p("- No COMSOL, Aspen or Simulink models were run for comparison; only their modelling principles were followed.")
    p("- The jacket energy balance is checked against its isothermal limit and for multiple steady states, not against measured temperatures.")
    p("- Coil Reynolds and Dean numbers use the water viscosity correlation and the coil geometry typed by the user.")
    doc = ROOT / "docs" / "VALIDATION.md"
    doc.parent.mkdir(exist_ok=True)
    doc.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {doc}")


if __name__ == "__main__":
    main()
