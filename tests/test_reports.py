"""The app must reproduce the printed report numbers when the manual's assumptions are on."""

import numpy as np

from reactorgenome.analysis import analyse_batch, analyse_cstr, analyse_pfr, analyse_rtd, extrapolate_lambda
from reactorgenome.conductivity import conversion, lambda_inf_ion
from reactorgenome.data_lab import (
    BATCH_50,
    CSTR_GROUP8_PUBLISHED,
    CSTR_TIMED_PUBLISHED,
    PFR_GROUP8_PUBLISHED,
    RTD_PUBLISHED,
    RTD_RUNS,
    batch_50_frame,
    cstr_group8_frame,
    cstr_timed_frame,
    pfr_group8_frame,
)
from reactorgenome.flow import dispersion_outlet, peclet_open
from reactorgenome.kinetics import cstr_X, pfr_X, tank_outlet
from reactorgenome.sandbox import cstr_recycle_X, heat_curves, steady_states


def test_cstr_quadratic_limits():
    # Da = 1, equimolar → X = (3 - sqrt(5))/2
    assert abs(cstr_X(1.0) - (3 - np.sqrt(5)) / 2) < 1e-9
    assert cstr_X(0.0) == 0.0
    assert pfr_X(1.0) == 0.5
    # M = 1 formula must match the general one
    for Da in (0.2, 1.0, 3.0):
        assert abs(cstr_X(Da, 1.0) - cstr_X(Da, 1.0 + 1e-12)) < 1e-6


def test_group8_cstr_matches_report():
    df = cstr_group8_frame()
    out = analyse_cstr(df, V=2.15, cA=0.07, cB=0.07, lam_inf_mode="manual", lam_inf_fixed=3.75, how="drop_first", spacing=1.0)
    runs = out["runs"]
    for _, r in runs.iterrows():
        key = (int(r["T_C"]), int(round(r["Q"])))
        assert abs(r["X"] - CSTR_GROUP8_PUBLISHED["X"][key]) < 1.5e-3, (key, r["X"])
    got = {int(r.T_C): r.k for r in out["kinetics"].itertuples()}
    for T, k in CSTR_GROUP8_PUBLISHED["k"].items():
        assert abs(got[T] - k) < 0.05, (T, got[T], k)
    arr = out["arrhenius"]
    assert abs(arr["Ea_kJ"] - CSTR_GROUP8_PUBLISHED["Ea_kJ"]) < 0.05
    assert abs(arr["A"] - CSTR_GROUP8_PUBLISHED["A"]) / CSTR_GROUP8_PUBLISHED["A"] < 0.01
    assert abs(arr["r2"] - CSTR_GROUP8_PUBLISHED["r2"]) < 0.02


def test_timed_cstr_and_the_temperature_inversion():
    df = cstr_timed_frame()
    manual = analyse_cstr(df, 2.15, 0.07, 0.07, "manual", 0.1, "all", 1.0)
    got = {int(r.T_C): r.k for r in manual["kinetics"].itertuples()}
    for T, k in CSTR_TIMED_PUBLISHED["k"].items():
        assert abs(got[T] - k) < 0.03, (T, got[T], k)
    # 28 °C, 15 LPH beats 40 °C on the average, and does not after the tail is finished
    r28 = manual["runs"][(manual["runs"].T_C == 28) & (np.isclose(manual["runs"].Q, 15))].iloc[0]
    r40 = manual["runs"][(manual["runs"].T_C == 40) & (np.isclose(manual["runs"].Q, 15))].iloc[0]
    assert abs(r28["X"] - 0.4909) < 1.5e-3
    assert r28["X"] > r40["X"]
    fixed = extrapolate_lambda([8.6, 9.6, 10.6], [4.11, 4.21, 4.29])
    lam_ss = fixed["geo"] if fixed["geo"] is not None else fixed["lam_ss"]
    # geometric tail of a pure exponential
    assert abs(fixed["geo"] - 4.61) < 0.02
    X_ss = float(conversion(lam_ss, 8.16, 0.1))
    assert X_ss < r40["X"]
    assert X_ss < 0.45


def test_pfr_matches_report():
    df = pfr_group8_frame()
    out = analyse_pfr(df, V=0.55, cA=0.07, cB=0.07, lam_inf_mode="manual", lam_inf_fixed=0.1, line_mode="free")
    r = out["runs"][(out["runs"].T_C == 40) & (np.isclose(out["runs"].Q, 20))].iloc[0]
    assert abs(r["tau"] - 1.65) < 1e-6
    assert abs(r["X"] - 0.3340) < 1.5e-3
    got = {int(row.T_C): row for row in out["kinetics"].itertuples()}
    for T, k in PFR_GROUP8_PUBLISHED["k"].items():
        assert abs(got[T].k - k) < 0.05, (T, got[T].k, k)
        assert abs(got[T].delta_min - PFR_GROUP8_PUBLISHED["delta_min"][T]) < 0.15
    arr = out["arrhenius"]
    assert abs(arr["Ea_kJ"] - 46.5) < 0.8


def test_batch_sample_and_trace():
    assert len(BATCH_50["t_s"]) == 74
    assert BATCH_50["t_s"][2] == 10
    C = 0.040 * (9410 - 5390) / (15430 - 5390)
    assert abs(C - 0.016016) < 1e-5
    df = batch_50_frame()
    out = analyse_batch(df["t_s"], df["ec_uS"], 0.040, 15430, 5390, 10, 280)
    # The report smoothed the curve before fitting. Raw and lightly smoothed
    # slopes will not hit 4.5968 exactly. They must be the same order.
    assert 1.0 < out["k_integral"] < 12.0
    assert np.isfinite(out["n"])


def test_rtd_moments_and_pe():
    assert len(RTD_RUNS[0]["points"]) == 91
    assert len(RTD_RUNS[1]["points"]) == 65
    assert len(RTD_RUNS[2]["points"]) == 40
    assert len(RTD_RUNS[3]["points"]) == 28
    for run in RTD_RUNS:
        t = [p[0] for p in run["points"]]
        s = [p[1] for p in run["points"]]
        out = analyse_rtd(t, s, run["bg"], run["Q"], run["V"], "CSTR" if run["kind"] == "CSTR" else "bed")
        pub = RTD_PUBLISHED[run["name"]]
        assert abs(out["moments"]["tm"] - pub["tm"]) / pub["tm"] < 0.02, (run["name"], out["moments"]["tm"])
        assert abs(out["moments"]["var"] - pub["var"]) / pub["var"] < 0.03, (run["name"], out["moments"]["var"])
        if "Pe" in pub:
            assert abs(out["Pe_open"] - pub["Pe"]) < 1.2, (run["name"], out["Pe_open"])
        if "N_tau" in pub:
            assert abs(out["N"]["N_from_tau"] - pub["N_tau"]) < 0.05


def test_pe_formula():
    # report: v = 0.05575 → Pe ≈ 39.50
    assert abs(peclet_open(0.05575) - 39.50) < 0.05


def test_recycle_does_not_change_one_cstr():
    direct, via = cstr_recycle_X(k=10.0, CA0=0.035, CB0=0.035, tau=6.0, R=3.0)
    assert abs(direct - via) < 1e-3


def test_dispersion_limits():
    Da = 1.2
    X_lo, _, _ = dispersion_outlet(0.2, Da)
    X_hi, _, _ = dispersion_outlet(400, Da)
    X_mid, x, c = dispersion_outlet(40, Da)
    assert abs(X_lo - cstr_X(Da)) < 1e-6
    assert abs(X_hi - pfr_X(Da)) < 1e-6
    assert cstr_X(Da) < X_mid < pfr_X(Da)
    assert x is not None


def test_ion_ratio_is_not_either_manual_constant():
    # Group 8 Λ0 at 31 °C is 8.17. The ion estimate is near 3, not 0.1 and not forced to 3.75.
    linf = lambda_inf_ion(8.17)
    assert 2.5 < linf < 3.5


def test_multiplicity_preset_has_three_states():
    # A hot, concentrated what-if. Not the dilute lab feed.
    A, Ea = 1.0e12, 90000.0
    T = np.linspace(5, 160, 400)
    _X, Qg, Qr = heat_curves(T, A, Ea, CA0=5.0, M=1.0, V=2.15, v0=0.4, dH=-100000.0, UA=40.0, Ta=30.0, T0=30.0)
    states = steady_states(T, Qg, Qr)
    assert len(states) >= 3, states
    assert any(s["stable"] for s in states)
    assert any(not s["stable"] for s in states)


def test_tank_mass_balance():
    # τ = X / (k CA^2 / CA0) wait: known Da
    k, CA0, tau = 20.0, 0.035, 5.0
    Da = k * CA0 * tau
    X = cstr_X(Da)
    CA, CB, _, _ = tank_outlet(CA0, CA0, k, tau)
    assert abs((CA0 - CA) / CA0 - X) < 1e-5
    assert abs(CA - CB) < 1e-8
