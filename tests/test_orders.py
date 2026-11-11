"""Design functions for a general reaction order."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).parent))
import report_data as rd  # noqa: E402

from reactorgenome.engine import labdata  # noqa: E402


@pytest.mark.parametrize("kind", ["cstr", "pfr"])
@pytest.mark.parametrize("order", [1.0, 1.5, 2.0, 2.5, 3.0])
def test_inverse(kind, order):
    for X in (0.05, 0.3, 0.6, 0.9):
        y = labdata.y_of_X(X, 1.0, kind, order)
        assert labdata.X_of_y(y, 1.0, kind, order) == pytest.approx(X, abs=1e-10)


@pytest.mark.parametrize("order", [1.0, 1.5, 2.5, 3.0])
def test_pfr_y_is_k_tau_from_the_ode(order):
    k, C0, tau = 3.0, 0.05, 2.0
    sol = solve_ivp(lambda t, c: [-k * c[0] ** order], (0, tau), [C0], rtol=1e-11, atol=1e-14)
    X = 1 - sol.y[0, -1] / C0
    assert labdata.y_of_X(X, 1.0, "pfr", order) == pytest.approx(k * C0 ** (order - 1) * tau, rel=1e-7)


@pytest.mark.parametrize("order", [1.0, 1.5, 2.5, 3.0])
def test_cstr_y_is_k_tau_from_the_balance(order):
    k, C0, tau = 3.0, 0.05, 2.0
    X = labdata.X_of_y(k * C0 ** (order - 1) * tau, 1.0, "cstr", order)
    C = C0 * (1 - X)
    assert C0 - C == pytest.approx(k * C**order * tau, rel=1e-9)


def test_order_two_keeps_the_report_numbers():
    runs = labdata.pfr_runs(rd.pfr_frame(), V=rd.PFR_V, CA0=rd.PFR_CA0, lam_mode="fixed", lam_fixed=rd.PFR_LINF)
    kt = labdata.fit_lines(labdata.with_order(runs, "pfr", 2.0), "y_tau")
    for T, k in rd.PFR_K.items():
        assert kt.loc[kt["T_C"] == T, "k"].iloc[0] == pytest.approx(k, rel=1e-4)


def test_first_order_k_has_no_concentration():
    runs = labdata.pfr_runs(rd.pfr_frame(), V=rd.PFR_V, CA0=rd.PFR_CA0, lam_mode="fixed", lam_fixed=rd.PFR_LINF)
    r1 = labdata.with_order(runs, "pfr", 1.0)
    assert np.allclose(r1["scale"], 1.0)
    ok = np.isfinite(r1["y"])
    assert np.allclose(r1.loc[ok, "y"], -np.log(1 - r1.loc[ok, "X"]))


def test_lam0_table_and_suspect():
    df = rd.pfr_frame()
    table = df[["T_C", "lam0"]].drop_duplicates()
    out = labdata.attach_lam0(df.drop(columns=["lam0"]), table)
    assert np.allclose(out["lam0"], df["lam0"])
    assert labdata.lam0_suspect(df) is None
    shifted = df.assign(lam0=df["r1"], r1=df["r2"])
    assert labdata.lam0_suspect(shifted) is not None
