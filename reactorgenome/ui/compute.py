"""Cached calls into the engine. Streamlit reruns the script on every click;
these keep the solvers from repeating work whose inputs have not changed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import streamlit as st

from ..engine import batch, cstr, flowsheet, labdata, pfr, rtd
from ..engine.rate import Kinetics

_C = dict(show_spinner=False, max_entries=96)


@st.cache_data(**_C)
def cstr_ss(m: cstr.CSTRModel, init="feed", T_init_C=None):
    return cstr.steady_state(m, init=init, T_init_C=T_init_C)


@st.cache_data(**_C)
def cstr_sim(m: cstr.CSTRModel, t_end: float, n: int, init: str, y0: tuple | None, T_init_C=None):
    return cstr.simulate(m, t_end, n, init=init, y0=np.array(y0) if y0 is not None else None, T_init_C=T_init_C)


@st.cache_data(**_C)
def cstr_settle(m: cstr.CSTRModel, init: str, y0: tuple | None):
    ss = cstr.steady_state(m)
    return cstr.settling(m, ss, init=init, y0=np.array(y0) if y0 is not None else None)


@st.cache_data(**_C)
def cstr_curve(m: cstr.CSTRModel, Q_values: tuple):
    """Steady X against τ by scaling both flows together."""
    out = []
    for Q in Q_values:
        f = Q / m.Q
        try:
            mm = replace(m, qA=m.qA * f, qB=m.qB * f)
            out.append(cstr.steady_state(mm)["X"])
        except Exception:  # noqa: BLE001 - a failed point is left as a gap
            out.append(np.nan)
    return np.array(out)


@st.cache_data(**_C)
def cstr_sweep(m: cstr.CSTRModel, field: str, values: tuple):
    out = []
    for v in values:
        try:
            if field == "k2_ratio":
                mm = replace(m, kin=replace(m.kin, k2_ratio=float(v)))
            elif field == "n_tanks":
                mm = replace(m, n_tanks=int(v))
            elif field == "M":
                mm = replace(m, cB_feed=m.cA_feed * m.qA / m.qB * float(v))
            else:
                mm = replace(m, **{field: float(v)})
            out.append(cstr.steady_state(mm)["X"])
        except Exception:  # noqa: BLE001
            out.append(np.nan)
    return np.array(out)


@st.cache_data(**_C)
def cstr_runs(df, **kw):
    return labdata.cstr_runs(df, **kw)


@st.cache_data(**_C)
def cstr_fit_model(runs, **kw):
    return labdata.fit_k_model(runs, **kw)


@st.cache_data(**_C)
def pfr_solve(m: pfr.PFRModel, n: int = 101):
    return pfr.solve(m, n)


@st.cache_data(**_C)
def pfr_curve(m: pfr.PFRModel, Q_values: tuple):
    out = []
    for Q in Q_values:
        try:
            out.append(pfr.solve(replace(m, Q=float(Q)), n=21)["X_out"])
        except Exception:  # noqa: BLE001
            out.append(np.nan)
    return np.array(out)


@st.cache_data(**_C)
def batch_fit(t: tuple, C: tuple, order: str, CA0: float, M: float, fit_t0: bool):
    return batch.fit(np.array(t), np.array(C), order, CA0, M, fit_t0=fit_t0)


@st.cache_data(**_C)
def rtd_analyse(t: tuple, s: tuple, bg: float, tail: bool, V, Q):
    return rtd.analyse(np.array(t), np.array(s), bg, tail=tail, V=V, Q=Q)


@st.cache_data(**_C)
def rtd_models(r: dict):
    return rtd.fit_models(r)


@st.cache_data(**_C)
def plant_solve(plant_json: str, kin: Kinetics):
    import json

    return flowsheet.solve(json.loads(plant_json), kin)
