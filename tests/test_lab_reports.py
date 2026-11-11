"""Given a report's own conventions, the engine reproduces the report's numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import report_data as R  # noqa: E402

from reactorgenome.engine import labdata, rtd  # noqa: E402
from reactorgenome.engine.rate import Kinetics  # noqa: E402
from reactorgenome.kinetics import arrhenius  # noqa: E402

KIN = Kinetics(k_ref=6.0, T_ref_C=30.0, Ea=45e3)


def _cstr_runs(lam_mode="fixed"):
    return labdata.cstr_runs(R.cstr_frame(), V=R.CSTR_V, CA0=R.CSTR_CA0, kin=KIN, lam_mode=lam_mode, lam_fixed=R.CSTR_LINF, method="mean_all")


def test_cstr_sheet_has_no_problems():
    assert labdata.row_problems(R.cstr_frame(), R.CSTR_V) == []


def test_cstr_conversions_and_point_k():
    runs = _cstr_runs()
    assert np.allclose(runs["tau"], [R.CSTR_TAU[int(q)] for q in runs["Q_LPH"]])
    assert np.max(np.abs(runs["X"].values - R.CSTR_X)) < 2e-4
    assert np.allclose(runs["k_point"], R.CSTR_KPOINT, rtol=2e-3)


def test_cstr_rate_constants_r2_and_arrhenius():
    kt = labdata.fit_lines(_cstr_runs(), "y_tau").set_index("T_C")
    for T, k in R.CSTR_K.items():
        assert kt.loc[T, "k"] == pytest.approx(k, rel=2e-5)
        assert kt.loc[T, "r2"] == pytest.approx(R.CSTR_R2[T], abs=1e-4)
    arr = arrhenius(kt.index.values, kt["k"].values)
    assert arr["Ea_kJ"] == pytest.approx(R.CSTR_EA_KJ, abs=0.01)
    assert arr["A"] == pytest.approx(R.CSTR_A, rel=5e-3)


def test_cstr_readings_fix_lam_inf_near_ion_value():
    res = labdata.fit_lam_inf(_cstr_runs(), "cstr")
    assert 0.25 < res["ratio"] < 0.35 and res["snr"] > 10
    assert res["lo"] > 0.1 / 8.25


def test_pfr_rate_constants_and_arrhenius():
    runs = labdata.pfr_runs(R.pfr_frame(), V=R.PFR_V, CA0=R.PFR_CA0, lam_mode="fixed", lam_fixed=R.PFR_LINF)
    assert np.allclose(runs["k_point"], R.PFR_KPOINT, rtol=1e-4)
    kt = labdata.fit_lines(runs, "y_tau").set_index("T_C")
    for T, k in R.PFR_K.items():
        assert kt.loc[T, "k"] == pytest.approx(k, rel=2e-5)
    arr = arrhenius(kt.index.values, kt["k"].values)
    assert arr["Ea_kJ"] == pytest.approx(R.PFR_EA_KJ, abs=0.002)
    assert arr["A"] == pytest.approx(R.PFR_A, rel=5e-4)


def test_pfr_free_line_and_lam_inf_are_flagged():
    runs = labdata.pfr_runs(R.pfr_frame(), V=R.PFR_V, CA0=R.PFR_CA0, lam_mode="fixed", lam_fixed=R.PFR_LINF)
    free = labdata.fit_lines(runs, "free").set_index("T_C")
    assert not np.isfinite(free.loc[50.0, "k"])
    assert labdata.fit_lam_inf(runs, "pfr")["snr"] < 3


@pytest.mark.parametrize("name", list(R.RTD_RUNS))
@pytest.mark.parametrize("calibrated", [False, True])
def test_rtd_moments(name, calibrated):
    s = np.asarray(R.RTD_RUNS[name], float)
    t = np.arange(len(s)) * R.RTD_DT
    c = R.RTD_CAL[0] * s + R.RTD_CAL[1] if calibrated else s
    r = rtd.analyse(t, c, c[0])
    out = rtd.summary(r)
    want = R.RTD_RESULTS[name]
    assert out["tm"] == pytest.approx(want["tm"], abs=1e-3)
    assert out["var"] == pytest.approx(want["var"], abs=2e-3)
    assert out["sigma"] == pytest.approx(want["sigma"], abs=1e-3)
    if "N" in want:
        assert out["N"] == pytest.approx(want["N"], abs=1e-3)
        assert out["N_whole"] == 2
    else:
        assert out["Pe"] == pytest.approx(want["Pe"], abs=0.01)
        assert out["tau_Pe"] == pytest.approx(want["tau"], abs=1e-3)


def test_rtd_dispersion_coefficient():
    s = np.asarray(R.RTD_RUNS["Packed bed 15 LPH"], float)
    r = rtd.analyse(np.arange(len(s)) * R.RTD_DT, s, s[0])
    out = rtd.summary(r, Q_Lmin=15 / 60, bed={"L_m": 0.5, "D_m": 0.025, "eps": 0.4})
    u = 15 / 1000 / 3600 / (np.pi * 0.025**2 / 4)
    assert out["D_L"] == pytest.approx(u * 0.5 / (0.4 * out["Pe"]), rel=1e-12)


def test_rtd_calibration_line():
    conc = np.array([0.0, 0.001, 0.002, 0.004])
    sig = (conc - R.RTD_CAL[1]) / R.RTD_CAL[0]
    cal = rtd.calibration(conc, sig)
    assert cal["a"] == pytest.approx(R.RTD_CAL[0], rel=1e-9) and cal["b"] == pytest.approx(R.RTD_CAL[1], abs=1e-12)
