"""Engine checks: analytic limits, internal consistency, bad input, balances."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from reactorgenome.engine import batch, cstr, flowsheet, pfr, rtd, steady, units
from reactorgenome.engine.rate import Kinetics, equilibrium_conversion
from reactorgenome.engine.units import EngineError

KIN = Kinetics(k_ref=6.0, T_ref_C=30.0, Ea=45e3)


def tank(**kw):
    base = dict(V=2.15, qA=0.2, qB=0.2, cA_feed=0.1, cB_feed=0.1, T_C=30.0, kin=KIN)
    base.update(kw)
    return cstr.CSTRModel(**base)


# ------------------------------------------------------------------ units


def test_units_round_trip():
    assert units.lph_to_lpm(60.0) == pytest.approx(1.0)
    assert units.lpm_to_lph(units.lph_to_lpm(37.0)) == pytest.approx(37.0)
    assert units.c_to_k(25.0) == pytest.approx(298.15)
    assert units.k_per_s_to_per_min(0.1) == pytest.approx(6.0)
    assert units.arrhenius_k(6.0, 30.0, 45e3, 30.0) == pytest.approx(6.0)
    assert units.arrhenius_k(6.0, 30.0, 45e3, 40.0) > 6.0


def test_bounds_catch_bad_values():
    assert units.check("V", -1) is not None
    assert units.check("V", 0) is not None
    assert units.check("V", float("nan")) is not None
    assert units.check("V", "abc") is not None
    assert units.check("V", 2.0) is None
    assert units.check("bypass", 0.99) is not None


# ------------------------------------------------------------------ CSTR


@pytest.mark.parametrize("cB", [0.1, 0.15, 0.07])
def test_cstr_ideal_matches_analytic(cB):
    m = tank(cB_feed=cB)
    Da = m.kin.k_C(m.T_C) * m.CA0 * m.tau
    assert cstr.steady_state(m)["X"] == pytest.approx(cstr.ideal_X(Da, m.M), abs=1e-10)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"n_tanks": 3},
        {"bypass": 0.2},
        {"stagnant": 0.25, "exchange": 0.4},
        {"stagnant": 0.25, "exchange": 0.0},
        {"n_tanks": 2, "bypass": 0.1, "stagnant": 0.2, "exchange": 0.3},
        {"kin": Kinetics(k_ref=6.0, k2_ratio=0.2)},
    ],
)
def test_cstr_transient_settles_on_algebraic_steady_state(kw):
    m = tank(**kw)
    ss = cstr.steady_state(m)
    sim = cstr.simulate(m, t_end=80 * m.tau, n_out=50, init="water")
    assert sim["X_out"][-1] == pytest.approx(ss["X"], abs=1e-7)


def test_dead_volume_is_a_smaller_tank():
    a = cstr.steady_state(tank(stagnant=0.3, exchange=0.0))["X"]
    b = cstr.steady_state(tank(V=2.15 * 0.7))["X"]
    assert a == pytest.approx(b, abs=1e-12)


def test_bypass_mass_balance():
    beta = 0.25
    m = tank(bypass=beta)
    Da = m.kin.k_C(30) * m.CA0 * m.tau / (1 - beta)
    assert cstr.steady_state(m)["X"] == pytest.approx((1 - beta) * cstr.ideal_X(Da), abs=1e-10)


def test_many_tanks_approach_plug_flow():
    Da = 6.0 * 0.05 * tank().tau
    gaps = [pfr.ideal_X(Da) - cstr.steady_state(tank(n_tanks=n))["X"] for n in (10, 20, 40)]
    assert gaps[0] > gaps[1] > gaps[2] > 0
    assert gaps[0] / gaps[2] == pytest.approx(4.0, rel=0.1)


def test_settling_time_single_tank():
    m = tank()
    ss = cstr.steady_state(m)
    X = ss["X"]
    # flow change with the same feed: only the reaction mode is disturbed
    prev = cstr.steady_state(tank(qA=0.4, qB=0.4))["state"]
    fast = cstr.settling(m, ss, y0=prev)["tau_r"]
    assert fast == pytest.approx(m.tau * (1 - X) / (1 + X), rel=1e-2)
    # start from water: sodium washes in on τ
    assert cstr.settling(m, ss, init="water")["tau_r"] == pytest.approx(m.tau, rel=1e-2)
    assert cstr.settling(m, ss)["tau_slow"] == pytest.approx(m.tau, rel=1e-4)


def test_reversible_limit_is_equilibrium():
    kin = Kinetics(k_ref=500.0, k2_ratio=0.3)
    m = tank(kin=kin, V=50.0)
    assert cstr.steady_state(m)["X"] == pytest.approx(equilibrium_conversion(m.CA0, m.CB0, 0.3), abs=2e-3)


def test_jacket_with_no_heat_is_isothermal():
    kin = Kinetics(k_ref=6.0, Ea=45e3, dH=0.0)
    iso = cstr.steady_state(tank(kin=kin))["X"]
    jac = cstr.steady_state(tank(kin=kin, jacket=True, UA=500.0, T_jacket_C=30.0))
    assert jac["X"] == pytest.approx(iso, abs=1e-7)
    assert jac["T_out_C"] == pytest.approx(30.0, abs=1e-6)


def test_jacket_multiple_steady_states():
    A, Ea = 1e12, 90e3
    k30 = A * np.exp(-Ea / (units.R_GAS * 303.15))
    kin = Kinetics(k_ref=k30, T_ref_C=30.0, Ea=Ea, dH=-100e3)
    m = cstr.CSTRModel(V=2.15, qA=0.2, qB=0.2, cA_feed=10.0, cB_feed=10.0, T_C=30.0, kin=kin,
                       jacket=True, UA=40.0, T_jacket_C=30.0)
    cold = cstr.steady_state(m, init="feed", T_init_C=30.0)
    hot = cstr.steady_state(m, init="feed", T_init_C=150.0)
    assert hot["T_out_C"] - cold["T_out_C"] > 50.0
    assert cstr.settling(m, cold)["stable"] and cstr.settling(m, hot)["stable"]


def test_conductivity_mapping_matches_manual_formula():
    m = tank()
    ss = cstr.steady_state(m)
    lam0, linf = 10.0, 3.66
    lam = cstr.conductivity(ss["c_out"][0], ss["c_out"][2], m.CA0, lam0, linf)
    assert (lam0 - lam) / (lam0 - linf) == pytest.approx(ss["X"], abs=1e-12)
    assert cstr.conductivity(0.0, 0.0, m.CA0, lam0, linf) == 0.0


def test_cstr_rejects_zero_flow():
    with pytest.raises(EngineError):
        cstr.steady_state(tank(qA=0.0, qB=0.0))
    with pytest.raises(EngineError):
        cstr.simulate(tank(), t_end=-1)


# ------------------------------------------------------------------ steady-state assessment


def test_plateau_fit_recovers_known_value():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    lam = 5.0 + 2.0 * np.exp(-t / 3.0)
    a = steady.assess(t, lam, 10.0, 3.66, method="model_tau", tau_r=3.0)
    assert a["fit"]["lam_ss"] == pytest.approx(5.0, abs=1e-10)
    b = steady.assess(t, lam, 10.0, 3.66, method="free_tau", tau_r=1.0)
    assert b["fit"]["lam_ss"] == pytest.approx(5.0, abs=1e-6)
    assert b["fit"]["tau_r"] == pytest.approx(3.0, rel=1e-5)


def test_steady_verdicts():
    t = np.array([2.0, 4.0, 6.0, 8.0])
    rising = 5.0 + 2.0 * np.exp(-t / 3.0)
    a = steady.assess(t, rising, 10.0, 3.66, tau_r=3.0, criterion="approach", tol_X=0.01)
    assert a["verdict"] == "not_steady"
    assert a["t_ready"] > 8.0
    t2 = np.array([20.0, 22.0, 24.0, 26.0])
    flat = 5.0 + 2.0 * np.exp(-t2 / 3.0)
    assert steady.assess(t2, flat, 10.0, 3.66, tau_r=3.0, criterion="approach")["verdict"] == "steady"
    assert steady.assess(t2, flat, 10.0, 3.66, tau_r=3.0, criterion="elapsed", n_relax=4)["verdict"] == "steady"
    one = steady.assess([5.0], [6.0], 10.0, 3.66, tau_r=3.0, criterion="drift")
    assert one["verdict"] == "unknown"


def test_plateau_uncertainty_with_noise():
    rng = np.random.default_rng(1)
    t = np.linspace(1, 12, 8)
    lam = 5.0 + 2.0 * np.exp(-t / 3.0) + rng.normal(0, 0.01, t.size)
    f = steady.fit_plateau(t, lam, "model_tau", 3.0)
    lo, hi = f["ci"]
    assert lo < 5.0 < hi


def test_model_shape_recovers_plateau_after_start_from_water():
    # Early readings after a water start carry two modes; one exponential is biased, the simulated shape is not.
    m = tank()
    ss = cstr.steady_state(m)
    sim = cstr.simulate(m, 12.0, 241, init="water")
    j = [np.argmin(np.abs(sim["t"] - t)) for t in (3.0, 6.0, 9.0, 12.0)]
    lam0, linf = 10.0, 3.66
    lam = cstr.conductivity(sim["c_out"][j, 0], sim["c_out"][j, 2], m.CA0, lam0, linf)
    lam_ss = cstr.conductivity(ss["c_out"][0], ss["c_out"][2], m.CA0, lam0, linf)
    sh = cstr.settling(m, ss, init="water")
    shape = (sh["shape"]["t"], lam0 * sh["shape"]["dA"] + linf * sh["shape"]["dC"])
    good = steady.fit_plateau(sim["t"][j], lam, "model_shape", shape=shape)
    one = steady.fit_plateau(sim["t"][j], lam, "model_tau", sh["tau_r"])
    assert good["lam_ss"] == pytest.approx(lam_ss, abs=5e-5)
    assert abs(one["lam_ss"] - lam_ss) > 20 * abs(good["lam_ss"] - lam_ss)
    a = steady.assess(sim["t"][j], lam, lam0, linf, method="model_shape", shape=shape)
    assert np.isfinite(a["t_ready"]) or a["verdict"] == "steady"


# ------------------------------------------------------------------ batch


def test_batch_second_order_forms_agree():
    t = np.linspace(0, 30, 31)
    C1 = batch.model_C(t, "second", 6.0, 0.05, 1.0)
    assert np.allclose(1 / C1 - 1 / 0.05, 6.0 * t)
    C2 = batch.model_C(t, "second", 6.0, 0.05, 1.6)
    C3 = batch.model_C(t, "reversible", 6.0, 0.05, 1.6, K=0.0)
    assert np.allclose(C2, C3, atol=1e-8)


def test_batch_fit_recovers_k_and_clock_offset():
    t = np.linspace(0.5, 20, 25)
    C = batch.model_C(t + 0.3, "second", 6.0, 0.05, 1.0)
    f = batch.fit(t, C, "second", 0.05, 1.0, fit_t0=True)
    assert f["k"] == pytest.approx(6.0, rel=1e-5)
    assert f["t0"] == pytest.approx(0.3, abs=1e-5)


def test_batch_power_law_finds_order():
    t = np.linspace(0, 20, 30)
    C = batch.model_C(t, "second", 6.0, 0.05, 1.0)
    f = batch.fit(t, C, "nth", 0.05, fit_t0=False)
    assert f["n"] == pytest.approx(2.0, abs=1e-4)


def test_batch_integral_transform_is_linear():
    t = np.linspace(0, 20, 20)
    for M in (1.0, 1.5):
        C = batch.model_C(t, "second", 4.0, 0.05, M)
        y = batch.integral_y(C, 0.05, "second", M)
        assert np.allclose(y, 4.0 * t, rtol=1e-6, atol=1e-8)


def test_batch_needs_three_points():
    with pytest.raises(EngineError):
        batch.fit([1, 2], [0.04, 0.03], "second", 0.05)


# ------------------------------------------------------------------ PFR


@pytest.mark.parametrize("M", [1.0, 2.0, 0.6])
def test_plug_matches_analytic(M):
    m = pfr.PFRModel(V=0.5, Q=0.4, CA0=0.05, CB0=0.05 * M, T_C=30.0, kin=KIN)
    Da = 6.0 * 0.05 * m.tau
    assert pfr.solve(m)["X_out"] == pytest.approx(pfr.ideal_X(Da, M), abs=1e-7)


def test_plug_delay_equals_longer_tube():
    m = pfr.PFRModel(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30.0, kin=KIN, delay=2.0)
    assert pfr.solve(m)["X_measured"] == pytest.approx(pfr.ideal_X(6.0 * 0.05 * (m.tau + 2.0)), abs=1e-7)


def test_adiabatic_without_heat_is_isothermal():
    kin = Kinetics(k_ref=6.0, Ea=45e3, dH=0.0)
    a = pfr.solve(pfr.PFRModel(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30, kin=kin, adiabatic=True))
    b = pfr.solve(pfr.PFRModel(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30, kin=kin))
    assert a["X_out"] == pytest.approx(b["X_out"], abs=1e-9)


def test_recycle_limits():
    base = dict(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30.0, kin=KIN)
    plug = pfr.solve(pfr.PFRModel(**base))["X_out"]
    assert pfr.solve(pfr.PFRModel(**base, recycle=1e-9))["X_out"] == pytest.approx(plug, abs=1e-7)
    big = pfr.solve(pfr.PFRModel(**base, recycle=400.0))["X_out"]
    tau = 0.5 / 0.4
    assert big == pytest.approx(cstr.ideal_X(6.0 * 0.05 * tau), abs=2e-3)


def test_dispersion_limits():
    base = dict(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30.0, kin=Kinetics(k_ref=30.0))
    Da = 30.0 * 0.05 * 0.5 / 0.4
    hi = pfr.solve(pfr.PFRModel(**base, flow="dispersion", Pe=2000))["X_out"]
    lo = pfr.solve(pfr.PFRModel(**base, flow="dispersion", Pe=0.05))["X_out"]
    assert hi == pytest.approx(pfr.ideal_X(Da), abs=2e-3)
    assert lo == pytest.approx(cstr.ideal_X(Da), abs=5e-3)


def test_laminar_below_plug_for_second_order():
    base = dict(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30.0, kin=Kinetics(k_ref=30.0))
    lam = pfr.solve(pfr.PFRModel(**base, flow="laminar"))["X_out"]
    plug = pfr.solve(pfr.PFRModel(**base))["X_out"]
    from reactorgenome.flow import laminar_pipe_E, segregated_X

    tau = 0.5 / 0.4
    t = np.linspace(tau / 2 + 1e-6, 400 * tau, 400000)
    ref = segregated_X(t, laminar_pipe_E(t, tau), 30.0, 0.05, 1.0)
    assert lam < plug
    assert lam == pytest.approx(ref, abs=2e-3)


def test_pfr_rejects_bad_combination():
    with pytest.raises(EngineError):
        pfr.solve(pfr.PFRModel(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30, flow="laminar", recycle=1.0))


# ------------------------------------------------------------------ RTD


def test_rtd_ideal_cstr_moments():
    tau = 4.0
    t = np.linspace(0, 60, 1201)
    r = rtd.analyse(t, np.exp(-t / tau))
    assert r["tm"] == pytest.approx(tau, rel=2e-3)
    assert r["var"] == pytest.approx(tau**2, rel=5e-3)
    assert r["F"][-1] == pytest.approx(1.0, abs=1e-9)
    assert trapezoid(r["E"], r["t"]) == pytest.approx(1.0, abs=1e-12)


def test_rtd_tail_correction():
    tau = 4.0
    t = np.linspace(0, 12, 241)
    cut = rtd.analyse(t, np.exp(-t / tau))
    fixed = rtd.analyse(t, np.exp(-t / tau), tail=True)
    assert abs(fixed["tm"] - tau) < 0.02 * tau < abs(cut["tm"] - tau)
    assert fixed["tail"]["area_frac"] > 0.03


def test_rtd_tanks_fit():
    t = np.linspace(0, 40, 801)
    E = rtd.E_tanks(t, 6.0, 3.0)
    r = rtd.analyse(t, E)
    assert rtd.N_moments(r) == pytest.approx(3.0, rel=1e-2)
    fit = {row["model"]: row for row in rtd.fit_models(r)}
    assert fit["Tanks in series (fitted)"]["params"]["N"] == pytest.approx(3.0, rel=1e-3)


def test_rtd_dispersion_fit_and_peclet_forms():
    t = np.linspace(0.01, 30, 3000)
    E = rtd.E_dispersion(t, 5.0, 20.0)
    r = rtd.analyse(t, E)
    fit = {row["model"]: row for row in rtd.fit_models(r)}
    assert fit["Axial dispersion, open (fitted)"]["params"]["Pe"] == pytest.approx(20.0, rel=1e-3)
    assert rtd.Pe_open_exact(r["v"]) == pytest.approx(20.0, rel=1e-2)
    assert rtd.Pe_open_tm(r["v"]) != pytest.approx(20.0, rel=1e-2)
    v = 2 / 7 - 2 * (1 - np.exp(-7)) / 49
    assert rtd.Pe_closed(v) == pytest.approx(7.0, rel=1e-8)


def test_rtd_rejects_empty_signal():
    with pytest.raises(EngineError):
        rtd.analyse([0, 1, 2, 3, 4], [5, 5, 5, 5, 5], background=5)


# ------------------------------------------------------------------ flowsheet


def node(i, kind, **p):
    return dict(id=i, kind=kind, x=0, y=0, **p)


def test_flowsheet_series_equals_sequential():
    plant = {
        "nodes": [node("f", "feed", Q=0.4, CA=0.05, CB=0.05, T=30), node("a", "cstr", V=2.0, T=30),
                  node("b", "pfr", V=0.5, T=30), node("p", "product")],
        "edges": [{"from": "f", "to": "a"}, {"from": "a", "to": "b"}, {"from": "b", "to": "p"}],
    }
    res = flowsheet.solve(plant, KIN)
    Xa = cstr.ideal_X(6.0 * 0.05 * 5.0)
    CA1 = 0.05 * (1 - Xa)
    Xb = pfr.ideal_X(6.0 * CA1 * 1.25)
    assert res["X"] == pytest.approx(1 - (1 - Xa) * (1 - Xb), abs=1e-7)
    assert res["balance_error"] < 1e-9


def test_flowsheet_recycle_around_cstr_changes_nothing():
    plant = {
        "nodes": [node("f", "feed", Q=0.4, CA=0.05, CB=0.05, T=30), node("a", "cstr", V=2.0, T=30),
                  node("s", "split", f=0.6), node("p", "product")],
        "edges": [{"from": "f", "to": "a"}, {"from": "a", "to": "s"}, {"from": "s", "to": "p", "port": "out"},
                  {"from": "s", "to": "a", "port": "side"}],
    }
    res = flowsheet.solve(plant, KIN)
    assert res["X"] == pytest.approx(cstr.ideal_X(6.0 * 0.05 * 5.0), abs=1e-8)
    assert res["balance_error"] < 1e-8


def test_flowsheet_bypass_matches_cstr_model():
    f = 0.25
    plant = {
        "nodes": [node("f", "feed", Q=0.4, CA=0.05, CB=0.05, T=30), node("s", "split", f=f),
                  node("a", "cstr", V=2.15, T=30), node("p", "product")],
        "edges": [{"from": "f", "to": "s"}, {"from": "s", "to": "a", "port": "out"},
                  {"from": "s", "to": "p", "port": "side"}, {"from": "a", "to": "p"}],
    }
    res = flowsheet.solve(plant, KIN)
    ref = cstr.steady_state(tank(bypass=f))["X"]
    assert res["X"] == pytest.approx(ref, abs=1e-9)


def test_flowsheet_recycle_pfr_matches_pfr_model():
    plant = {
        "nodes": [node("f", "feed", Q=0.4, CA=0.05, CB=0.05, T=30), node("b", "pfr", V=0.5, T=30),
                  node("s", "split", f=2.0 / 3.0), node("p", "product")],
        "edges": [{"from": "f", "to": "b"}, {"from": "b", "to": "s"}, {"from": "s", "to": "p", "port": "out"},
                  {"from": "s", "to": "b", "port": "side"}],
    }
    res = flowsheet.solve(plant, KIN)
    ref = pfr.solve(pfr.PFRModel(V=0.5, Q=0.4, CA0=0.05, CB0=0.05, T_C=30, kin=KIN, recycle=2.0))["X_out"]
    assert res["X"] == pytest.approx(ref, abs=1e-7)


def test_flowsheet_two_feeds_mix():
    plant = {
        "nodes": [node("f1", "feed", Q=0.2, CA=0.1, CB=0.0, T=30), node("f2", "feed", Q=0.2, CA=0.0, CB=0.1, T=30),
                  node("a", "cstr", V=2.15, T=30), node("p", "product")],
        "edges": [{"from": "f1", "to": "a"}, {"from": "f2", "to": "a"}, {"from": "a", "to": "p"}],
    }
    res = flowsheet.solve(plant, KIN)
    assert res["X"] == pytest.approx(cstr.steady_state(tank())["X"], abs=1e-9)


def test_flowsheet_validation_messages():
    bad = {"nodes": [node("a", "cstr", V=2.0, T=30)], "edges": []}
    msgs = flowsheet.validate(bad)
    assert any("Feed" in m for m in msgs) and any("Product" in m for m in msgs)
    with pytest.raises(EngineError):
        flowsheet.solve(bad, KIN)


def test_flowsheet_trapped_loop_detected():
    plant = {
        "nodes": [node("f", "feed", Q=0.4, CA=0.05, CB=0.05, T=30), node("a", "cstr", V=2.0, T=30),
                  node("s", "split", f=0.0), node("p", "product")],
        "edges": [{"from": "f", "to": "a"}, {"from": "a", "to": "s"}, {"from": "s", "to": "a", "port": "out"},
                  {"from": "s", "to": "p", "port": "side"}],
    }
    with pytest.raises(EngineError):
        flowsheet.solve(plant, KIN)
