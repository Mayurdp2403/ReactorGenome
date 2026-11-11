"""Every page renders without an exception, empty and with readings."""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent))
import synthetic  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TABLES = {"Batch": ("ws_batch", synthetic.batch_table), "CSTR": ("ws_cstr", synthetic.cstr_table),
          "PFR": ("ws_pfr", synthetic.pfr_table), "RTD": ("ws_rtd", synthetic.rtd_table)}


def _run(page, data=False, **state):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["nav"] = page
    if data and page in TABLES:
        key, make = TABLES[page]
        at.session_state[key] = make()
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    level, text = at.session_state["_status"]
    return level, text


@pytest.mark.parametrize("page", ["Home", "Batch", "CSTR", "PFR", "RTD", "Design Studio"])
def test_page_empty(page):
    _ok(_run(page))


@pytest.mark.parametrize("page", ["Batch", "CSTR", "PFR", "RTD"])
def test_page_with_readings(page):
    at = _run(page, data=True)
    level, text = _ok(at)
    assert level != "err", text
    z = zipfile.ZipFile(io.BytesIO(at.session_state["_export_builders"][page]()))
    names = z.namelist()
    assert "report.html" in names and "tables/readings.csv" in names
    assert any(n.startswith("plots/") for n in names)
    text = re.sub(r"<script.*?</script>", "", z.read("report.html").decode(), flags=re.S)
    assert ">None<" not in text and ">nan<" not in text.lower()


def test_design_export():
    at = _run("Design Studio")
    z = zipfile.ZipFile(io.BytesIO(at.session_state["_export_builders"]["Design Studio"]()))
    assert {"tables/streams.csv", "tables/blocks.csv", "parameters.json"} <= set(z.namelist())


def test_cstr_variants():
    for state in ({"c_noni": True, "c_beta": 0.1, "c_alpha": 0.1}, {"c_zones": True, "c_N": 3}, {"c_jacket": True}, {"c_mode": "steady"}, {"c_rev": True},
                  {"c_times": True, "c_ssm": "model_shape", "c_runstart": "water"}, {"c_fit": "tau_y"}, {"c_fit": "free"},
                  {"c_fit": "model", "c_fitb": True}, {"c_lmode": "fixed", "c_linf": 0.1}):
        _ok(_run("CSTR", data=True, **state))


def test_cstr_old_table_layout():
    old = synthetic.cstr_table().rename(columns={"Q_LPH": "qA_LPH"})
    old["qA_LPH"] = old["qA_LPH"] / 2
    old.insert(2, "qB_LPH", old["qA_LPH"])
    at = _run("CSTR", ws_cstr=old)
    level, text = _ok(at)
    assert "run(s) analysed" in text, text


def test_pfr_flow_models():
    for flow in ("plug", "dispersion", "laminar"):
        _ok(_run("PFR", data=True, p_flow=flow))
    _ok(_run("PFR", data=True, p_rec=True, p_R=1.0))
    _ok(_run("PFR", data=True, p_adia=True))


def test_pfr_fits_and_tau_column():
    df = synthetic.pfr_table()
    df["tau_min"] = 60 * 0.5 / df["Q_LPH"]
    df.loc[df.duplicated("T_C"), "lam0"] = float("nan")
    for fit in ("y_tau", "tau_y", "free"):
        level, text = _ok(_run("PFR", ws_pfr=df, p_fit=fit, p_eq=False, p_CB0=0.06))
        assert level == "ok", text


def _pfr_report(**state):
    import report_data as rd

    base = dict(p_V=rd.PFR_V, p_CA0=rd.PFR_CA0, p_lm="fixed", p_linf=rd.PFR_LINF)
    base.update(state)
    return base


def _markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_pfr_readings_shifted_into_lam0_do_not_crash():
    """Readings typed one column to the left (Λ1 in the Λ0 column) gave X ≤ 0 at whole temperatures."""
    import numpy as np
    import report_data as rd

    df = rd.pfr_frame()
    shifted = df.assign(lam0=df["r1"], r1=df["r2"], r2=df["r3"], r3=df["r4"], r4=df[["r1", "r2", "r3", "r4"]].mean(axis=1),
                        Q_LPH=60 * 0.5 / df["tau_min"])
    at = _run("PFR", **_pfr_report(ws_pfr=shifted))
    level, text = _ok(at)
    assert level == "warn", text
    assert "no larger than the outlet readings" in _markdown(at)
    allbad = df.assign(lam0=np.full(len(df), 5.0))
    _ok(_run("PFR", **_pfr_report(ws_pfr=allbad)))
    _ok(_run("CSTR", ws_cstr=synthetic.cstr_table().assign(lam0=1.0)))


def test_lam0_table_and_results():
    import report_data as rd

    df = rd.pfr_frame()
    lam0 = df[["T_C", "lam0"]].drop_duplicates().reset_index(drop=True)
    at = _run("PFR", **_pfr_report(ws_pfr=df.drop(columns=["lam0"]), ws_pfr_lam0=lam0))
    level, text = _ok(at)
    assert level == "ok", text
    md = _markdown(at)
    assert "Activation energy Ea" in md and "12.70" in md
    for T in (40, 50, 60):
        assert f"k at {T} °C" in md
    at = _run("PFR", **_pfr_report(ws_pfr=df.drop(columns=["lam0"])))
    level, text = _ok(at)
    assert level == "warn" and "add 40 °C to the Λ0 table" in _markdown(at)


def test_cstr_results_show_ea_and_k():
    import report_data as rd

    df = rd.cstr_frame()
    lam0 = df[["T_C", "lam0"]].dropna().reset_index(drop=True)
    at = _run("CSTR", ws_cstr=df.drop(columns=["lam0"]), ws_cstr_lam0=lam0, c_V=rd.CSTR_V, c_CA0=rd.CSTR_CA0, c_lmode="fixed", c_linf=rd.CSTR_LINF)
    _ok(at)
    md = _markdown(at)
    assert "25.07" in md and "3.163" in md and "k at 60 °C" in md


def test_batch_readings_below_kappa_inf_are_left_out():
    """With the ion estimate κ∞ = 0.366 κ0 the last readings of this run sit below κ∞ (CA < 0); they used to wreck the fit."""
    import numpy as np
    import pandas as pd
    from reactorgenome.data_lab import BATCH_50 as b

    df = pd.DataFrame({"T_C": b["T"], "t_s": b["t_s"], "kappa": np.asarray(b["ec"], float) / 1000})
    at = _run("Batch", ws_batch=df, b_C0=b["C0"], b_T=b["T"], b_tlo=b["window"][0], b_thi=b["window"][1])
    level, text = _ok(at)
    assert level != "err", text
    md = _markdown(at)
    assert "at or below κ∞" in md
    assert not re.search(r"\de\+(0[5-9]|[1-9]\d)", md), "a fitted k blew up"


def test_old_lam0_column_moves_to_its_table():
    at = _run("PFR", data=True)
    _ok(at)
    assert "lam0" not in at.session_state["ws_pfr"].columns
    assert len(at.session_state["ws_pfr_lam0"]) == synthetic.pfr_table()["T_C"].nunique()


@pytest.mark.parametrize("order", [1.0, 1.5, 2.5, 3.0])
def test_reaction_orders(order):
    for page, key in (("PFR", "p_order"), ("CSTR", "c_order")):
        level, text = _ok(_run(page, data=True, **{key: order}))
        assert level != "err", text
    _ok(_run("PFR", data=True, p_order=order, p_fit="free", p_arrax="neg"))
    _ok(_run("CSTR", data=True, c_order=order, c_fit="model"))


def test_batch_orders():
    for order in ("first", "second", "nth", "reversible"):
        _ok(_run("Batch", data=True, b_order=order))


def test_rtd_several_runs_bed_and_tail():
    at = _run("RTD", data=True, ws_rtdruns=synthetic.rtd_runs(), r_tail=True, r_pick="Bed 30 LPH", r_L=50.0, r_D=2.5, r_eps=0.4)
    level, text = _ok(at)
    assert "2 run(s)" in text, text


def test_rtd_calibration_standards():
    import pandas as pd

    cal = pd.DataFrame({"conc": [0.0, 0.001, 0.002, 0.004], "signal": [10.0, 68.0, 127.0, 244.0]})
    _ok(_run("RTD", data=True, r_cal="standards", ws_rtdcal=cal))
    _ok(_run("RTD", data=True, r_cal="typed", r_cal_a=1.7e-5, r_cal_b=-1.7e-4))


def test_rtd_old_single_run_table():
    import pandas as pd

    old = synthetic.rtd_table()[["t_min", "s1"]].rename(columns={"s1": "signal"})
    level, text = _ok(_run("RTD", ws_rtd=pd.DataFrame(old)))
    assert "1 run(s)" in text, text


def test_design_presets():
    at = _run("Design Studio")
    for key in ("series", "recycle", "bypass", "twofeeds", "jacket"):
        at.button(key=f"btn_preset_{key}").click().run()
        level, text = _ok(at)
        assert level == "ok", (key, text)


def test_navigation_keeps_page():
    at = _run("CSTR")
    at.session_state["nav"] = None
    at.run()
    assert at.session_state["nav"] == "CSTR"
