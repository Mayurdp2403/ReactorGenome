"""Small building blocks shared by every page."""

from __future__ import annotations

import html
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from ..engine.rate import Kinetics
from .style import AMBER, BORDER, CYAN, MUTED, PANEL2, TEXT


# ------------------------------------------------------------------ inputs


def _default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def num(key, label, lo, hi, default, step, help=None, fmt=None, disabled=False):
    _default(key, default)
    return st.number_input(label, min_value=lo, max_value=hi, step=step, key=key, help=help, format=fmt, disabled=disabled)


def slide(key, label, lo, hi, default, step, help=None, fmt=None, disabled=False):
    _default(key, default)
    return st.slider(label, min_value=lo, max_value=hi, step=step, key=key, help=help, format=fmt, disabled=disabled)


def toggle(key, label, default=False, help=None):
    _default(key, default)
    return st.toggle(label, key=key, help=help)


def choose(key, label, options: dict, default, help=None, horizontal=False, radio=False):
    _default(key, default)
    keys = list(options)
    if radio:
        return st.radio(label, keys, format_func=lambda k: options[k], key=key, help=help, horizontal=horizontal)
    return st.selectbox(label, keys, format_func=lambda k: options[k], key=key, help=help)


# ------------------------------------------------------------------ text blocks


def page_title(title, text):
    st.markdown(f"<div class='rg-title'><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>", unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f"<div class='rg-section'><h3>{html.escape(title)}</h3><span>{html.escape(sub)}</span></div>", unsafe_allow_html=True)


def panel_title(text):
    st.markdown(f"<div class='rg-panel-title'>{html.escape(text)}</div>", unsafe_allow_html=True)


def note(text, level="info"):
    st.markdown(f"<div class='rg-note {level}'>{html.escape(text)}</div>", unsafe_allow_html=True)


def notes(items):
    for it in items:
        if isinstance(it, dict):
            note(it["text"], it.get("level", "info"))
        else:
            note(str(it))


def empty(title, text):
    st.markdown(f"<div class='rg-empty'><b>{html.escape(title)}</b>{html.escape(text)}</div>", unsafe_allow_html=True)


def equation(text):
    st.markdown(f"<div class='rg-eq'>{html.escape(text)}</div>", unsafe_allow_html=True)


def fmt(v, spec=".3f", unit=""):
    try:
        if v is None or not np.isfinite(float(v)):
            return "–"
    except (TypeError, ValueError):
        return "–"
    return f"{float(v):{spec}}{(' ' + unit) if unit else ''}"


def kpis(items):
    """items: dicts with label, value (already formatted), unit, detail, hi."""
    cells = []
    for it in items:
        unit = f"<small>{html.escape(it.get('unit', ''))}</small>" if it.get("unit") else ""
        det = f"<div class='d'>{html.escape(it['detail'])}</div>" if it.get("detail") else ""
        cls = "rg-kpi hi" if it.get("hi") else "rg-kpi"
        cells.append(f"<div class='{cls}'><div class='l'>{html.escape(it['label'])}</div><div class='v'>{html.escape(str(it['value']))}{unit}</div>{det}</div>")
    st.markdown("<div class='rg-kpis'>" + "".join(cells) + "</div>", unsafe_allow_html=True)


def set_status(level: str, text: str, model: str = ""):
    st.session_state["_status"] = (level, text)
    st.session_state["_model_id"] = model


# ------------------------------------------------------------------ observation tables


def table_editor(store: str, columns: list[dict], n_blank: int = 6, file_name: str = "readings.csv", hidden=(), compact=False):
    """Editable table that starts empty. columns: dicts with key, label, help, fmt, min, max.

    Returns the rows that have at least one value. CSV import accepts either the
    column keys or the labels as headers. Hidden columns keep their values but
    are not shown.
    """
    keys = [c["key"] for c in columns]
    ver_key = f"{store}__v"
    _default(ver_key, 0)
    if store not in st.session_state:
        st.session_state[store] = pd.DataFrame({k: pd.Series([np.nan] * n_blank, dtype="float64") for k in keys})
    else:
        df = st.session_state[store]
        missing = [k for k in keys if k not in df.columns]
        if missing or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)
            for k in missing:
                df[k] = np.nan
            st.session_state[store] = df[keys + [c for c in df.columns if c not in keys]]
            st.session_state[ver_key] += 1
    cfg = {
        c["key"]: st.column_config.NumberColumn(c["label"], help=c.get("help"), format=c.get("fmt", "%.4g"),
                                                min_value=c.get("min"), max_value=c.get("max"))
        for c in columns
    }
    edited = st.data_editor(st.session_state[store], column_config=cfg, num_rows="dynamic", hide_index=True,
                            column_order=[k for k in keys if k not in hidden],
                            width="stretch", key=f"{store}__ed{st.session_state[ver_key]}")
    remember_table(store, edited)
    if compact:
        c2, c3, c4 = st.columns(3)
        up = None
    else:
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.2])
        with c1:
            up = st.file_uploader("Import CSV", type=["csv"], key=f"{store}__up{st.session_state[ver_key]}",
                                  label_visibility="collapsed", help="Headers may be the column keys or the labels shown above.")
    with c2:
        if st.button("+5" if compact else "Add 5 rows", key=f"{store}__add", width="stretch", help="Add 5 empty rows."):
            more = pd.DataFrame({k: pd.Series([np.nan] * 5, dtype="float64") for k in keys})
            st.session_state[store] = pd.concat([edited, more], ignore_index=True)
            st.session_state[ver_key] += 1
            st.rerun()
    with c3:
        if st.button("Clear" if compact else "Clear table", key=f"{store}__clr", width="stretch", help="Empty this table."):
            st.session_state[store] = pd.DataFrame({k: pd.Series([np.nan] * n_blank, dtype="float64") for k in keys})
            st.session_state[ver_key] += 1
            st.rerun()
    filled = edited.dropna(how="all")
    with c4:
        st.download_button("CSV" if compact else "Download CSV", filled.to_csv(index=False), file_name=file_name, mime="text/csv",
                           key=f"{store}__dl", width="stretch", disabled=filled.empty, help="Download this table as a CSV file.")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            lab2key = {c["label"]: c["key"] for c in columns}
            raw = raw.rename(columns=lambda h: lab2key.get(str(h).strip(), str(h).strip()))
            missing = [k for k in keys if k not in raw.columns]
            if len(missing) == len(keys):
                note("None of the CSV headers match this table, so nothing was imported.", "err")
            else:
                for k in missing:
                    raw[k] = np.nan
                df = raw[keys].apply(pd.to_numeric, errors="coerce").astype("float64")
                st.session_state[store] = df
                st.session_state[ver_key] += 1
                st.rerun()
        except Exception as exc:  # noqa: BLE001 - shown to the user
            note(f"Could not read that CSV: {exc}", "err")
    return filled.reset_index(drop=True)


# ------------------------------------------------------------------ stopwatch


def stopwatch(marks_min: list[float], label="Reading schedule"):
    """Browser-side stopwatch. Keeps running across reruns (localStorage)."""
    marks = json.dumps([round(float(m), 2) for m in marks_min if np.isfinite(m)])
    st.iframe(f"""<!doctype html><html><body style="margin:0;background:transparent">
<div id="sw" style="font-family:'IBM Plex Sans',sans-serif;color:{TEXT};background:{PANEL2};border:1px solid {BORDER};border-radius:12px;padding:10px 12px">
 <div style="display:flex;align-items:center;justify-content:space-between">
  <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:{MUTED}">Stopwatch · {html.escape(label)}</div>
  <div id="nx" style="font-size:12px;color:{MUTED}"></div></div>
 <div style="display:flex;align-items:center;gap:10px;margin-top:6px">
  <div id="tm" style="font-family:'IBM Plex Mono',monospace;font-size:30px;min-width:120px">00:00</div>
  <button id="go" style="background:{CYAN};color:#04202A;border:0;border-radius:8px;padding:6px 12px;font-weight:700;cursor:pointer">Start</button>
  <button id="rs" style="background:transparent;color:{TEXT};border:1px solid {BORDER};border-radius:8px;padding:6px 10px;cursor:pointer">Reset</button>
 </div>
 <div id="mk" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px"></div>
</div>
<script>
const marks={marks};const S=window.localStorage;const k='rg_sw';
function st(){{try{{return JSON.parse(S.getItem(k))||{{run:false,t0:0,acc:0}}}}catch(e){{return {{run:false,t0:0,acc:0}}}}}}
function save(s){{S.setItem(k,JSON.stringify(s))}}
function el(){{const s=st();return s.acc+(s.run?(Date.now()-s.t0)/1000:0)}}
const mk=document.getElementById('mk');
marks.forEach((m,i)=>{{const d=document.createElement('span');d.id='m'+i;d.textContent='t'+(i+1)+' '+m.toFixed(1)+' min';
 d.style.cssText='font-family:IBM Plex Mono,monospace;font-size:12px;padding:3px 8px;border-radius:999px;border:1px solid {BORDER};color:{MUTED}';mk.appendChild(d)}});
function tick(){{const e=el();const mm=Math.floor(e/60),ss=Math.floor(e%60);
 document.getElementById('tm').textContent=String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0');
 let next=-1;marks.forEach((m,i)=>{{const d=document.getElementById('m'+i);const due=e>=m*60;
  d.style.background=due?'rgba(52,211,153,.15)':'transparent';d.style.color=due?'#34D399':'{MUTED}';d.style.borderColor=due?'#34D399':'{BORDER}';
  if(!due&&next<0)next=i;
  if(Math.abs(e-m*60)<5&&st().run){{d.style.background='rgba(245,176,65,.25)';d.style.color='{AMBER}';d.style.borderColor='{AMBER}'}}}});
 document.getElementById('nx').textContent=next>=0?('next reading in '+Math.max(0,marks[next]*60-e).toFixed(0)+' s'):(marks.length?'all readings due':'');
 document.getElementById('go').textContent=st().run?'Pause':'Start';}}
document.getElementById('go').onclick=()=>{{const s=st();if(s.run){{s.acc=el();s.run=false}}else{{s.t0=Date.now();s.run=true}}save(s);tick()}};
document.getElementById('rs').onclick=()=>{{save({{run:false,t0:0,acc:0}});tick()}};
setInterval(tick,250);tick();
</script></body></html>""", height=132)


# ------------------------------------------------------------------ kinetics shared across rigs


def registry() -> dict:
    return st.session_state.setdefault("_kin_registry", {})


def publish(source: str, k_ref: float, T_ref: float, Ea: float, detail: str):
    if np.isfinite(k_ref) and k_ref > 0:
        registry()[source] = {"k_ref": float(k_ref), "T_ref": float(T_ref), "Ea": float(Ea) if np.isfinite(Ea) else 0.0,
                              "detail": detail}


def kinetics_panel(prefix: str, own: str, default_k=6.0, default_T=30.0, default_Ea=45.0, show_rev=True, rev_on=False):
    """Pick where k comes from and return a Kinetics object plus a label."""
    reg = registry()
    sources = {"typed": "Typed below"}
    for name, v in reg.items():
        if name != own:
            sources[name] = f"Fitted on the {name} page"
    if own in reg:
        sources = {own: "Fitted from this page's readings", **sources}
    src_key = f"{prefix}_ksrc"
    if src_key in st.session_state and st.session_state[src_key] not in sources:
        st.session_state[src_key] = "typed"
    src = choose(src_key, "Rate constant comes from", sources, "typed",
                 help="Use a value you type, or a k (and Ea) fitted from readings on this or another experiment page. "
                      "Fitting on one rig and predicting another is how the lab data become a design tool.")
    if src == "typed":
        k = num(f"{prefix}_k", "k at reference temperature (L/mol/min)", 0.0001, 1e6, default_k, 0.1,
                help="Second-order rate constant for NaOH + ethyl acetate. Literature puts it near 6 L/mol/min at 25–30 °C.", fmt="%.5g")
        Tr = num(f"{prefix}_Tr", "Reference temperature (°C)", 0.0, 100.0, default_T, 1.0, help="Temperature at which the k above applies.")
        Ea = num(f"{prefix}_Ea", "Activation energy (kJ/mol)", 0.0, 200.0, default_Ea, 1.0,
                 help="How strongly k rises with temperature. 0 means k does not change with temperature.")
        label = f"typed k = {k:.4g} L/mol/min at {Tr:g} °C, Ea = {Ea:g} kJ/mol"
    else:
        v = reg[src]
        k, Tr, Ea = v["k_ref"], v["T_ref"], v["Ea"] / 1000.0
        st.markdown(f"<div class='rg-eq'>k = {k:.4g} L/mol/min at {Tr:g} °C<br>Ea = {Ea:.1f} kJ/mol<br><span style='color:{MUTED}'>{html.escape(v['detail'])}</span></div>",
                    unsafe_allow_html=True)
        label = f"k from {src}: {k:.4g} L/mol/min at {Tr:g} °C, Ea = {Ea:.1f} kJ/mol"
    k2 = 0.0
    if show_rev and rev_on:
        k2 = slide(f"{prefix}_k2", "Reverse / forward rate constant, k2/k1", 0.0, 1.0, 0.05, 0.01,
                   help="Saponification is practically irreversible. A non-zero value lets acetate and ethanol react back, which caps conversion below 100%.")
    return Kinetics(k_ref=float(k), T_ref_C=float(Tr), Ea=float(Ea) * 1000.0, k2_ratio=float(k2)), label


def download_bundle(label, builder, file_name, key):
    """One click: the ZIP is built only when the button is pressed."""
    st.download_button(label, data=builder, file_name=file_name, mime="application/zip", key=key, type="primary",
                       width="stretch", icon=":material/download:")


_SAVED_PREFIXES = ("c_", "b_", "p_", "r_", "d_", "ws_")


def remember_table(store: str, edited: pd.DataFrame):
    """Edits live in the editor widget's state; keep the current table so a saved project has them."""
    st.session_state.setdefault("_tables", {})[store] = edited.copy()


def project_bytes() -> bytes:
    keep = {}
    tables = st.session_state.get("_tables", {})
    for k, v in st.session_state.items():
        if "__" in k or not (k.startswith(_SAVED_PREFIXES) or k in ("plant_state", "project_name")):
            continue
        if isinstance(v, pd.DataFrame):
            v = tables.get(k, v)
            keep[k] = {"__df__": json.loads(v.to_json(orient="split"))}
        elif isinstance(v, (int, float, str, bool)) or v is None:
            keep[k] = v
        elif isinstance(v, dict) and k == "plant_state":
            keep[k] = v
    keep["_kin_registry"] = registry()
    return json.dumps(keep, indent=1).encode()


def load_project(raw: bytes) -> int:
    data = json.loads(raw)
    n = 0
    for k, v in data.items():
        if k == "_kin_registry":
            st.session_state["_kin_registry"] = v
            continue
        if "__" in k or not (k.startswith(_SAVED_PREFIXES) or k in ("plant_state", "project_name")):
            continue
        if isinstance(v, dict) and "__df__" in v:
            df = pd.read_json(io.StringIO(json.dumps(v["__df__"])), orient="split")
            for col in df.columns:
                num = pd.to_numeric(df[col], errors="coerce")
                if num.notna().sum() == df[col].notna().sum():
                    df[col] = num.astype("float64")
            st.session_state[k] = df
            st.session_state.get("_tables", {}).pop(k, None)
            st.session_state[f"{k}__v"] = st.session_state.get(f"{k}__v", 0) + 1
        else:
            st.session_state[k] = v
        n += 1
    v = st.session_state.get("plant_version", 0) + 1
    st.session_state["plant_version"] = v
    if isinstance(st.session_state.get("plant_state"), dict):
        st.session_state["plant_state"] = {**st.session_state["plant_state"], "version": v}
    return n
