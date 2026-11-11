"""Export: CSV tables, parameter JSON, plot data and a self-contained HTML report.

The report is one HTML file. Plotly is embedded once, inline, so the file opens
without an internet connection. Figures are restyled light for printing.
"""

from __future__ import annotations

import datetime as _dt
import html
import io
import json
import zipfile

import numpy as np
import pandas as pd


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def params_json(params: dict) -> str:
    return json.dumps(_jsonable(params), indent=2, ensure_ascii=False)


def figure_frame(fig) -> pd.DataFrame:
    """Every trace of a Plotly figure as long-format rows: trace, x, y."""
    rows = []
    for tr in fig.data:
        x = getattr(tr, "x", None)
        y = getattr(tr, "y", None)
        if x is None or y is None:
            continue
        name = tr.name or tr.type
        for xi, yi in zip(np.asarray(x).tolist(), np.asarray(y).tolist()):
            rows.append({"trace": name, "x": xi, "y": yi})
    return pd.DataFrame(rows)


def _light(fig):
    f = fig.__class__(fig)
    f.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="#1F2937", family="Arial"), title_font=dict(color="#111827"))
    f.update_xaxes(gridcolor="#E5E7EB", linecolor="#9CA3AF", zerolinecolor="#D1D5DB", tickfont=dict(color="#374151"),
                   title_font=dict(color="#374151"))
    f.update_yaxes(gridcolor="#E5E7EB", linecolor="#9CA3AF", zerolinecolor="#D1D5DB", tickfont=dict(color="#374151"),
                   title_font=dict(color="#374151"))
    f.update_layout(legend=dict(font=dict(color="#1F2937"), bgcolor="rgba(255,255,255,0.8)"))
    return f


_CSS = """
body{font-family:Arial,Helvetica,sans-serif;color:#111827;max-width:1080px;margin:32px auto;padding:0 20px;line-height:1.5}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:30px 0 8px;border-bottom:2px solid #0E7490;padding-bottom:4px}
.sub{color:#4B5563;margin:0 0 18px}.meta{display:grid;grid-template-columns:200px 1fr;gap:4px 14px;font-size:14px}
.meta b{color:#374151}table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #D1D5DB;padding:4px 8px;text-align:right}
th{background:#F3F4F6}td:first-child,th:first-child{text-align:left}ul{margin:6px 0 6px 18px}pre{background:#F9FAFB;padding:10px;border:1px solid #E5E7EB;white-space:pre-wrap}
.warn{color:#92400E}.good{color:#065F46}.foot{margin-top:40px;color:#6B7280;font-size:12px}
"""


def _public(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal columns (leading underscore) and show missing values as a dash."""
    df = df[[c for c in df.columns if not str(c).startswith("_")]]

    def cell(v):
        if v is None:
            return "–"
        if isinstance(v, (float, np.floating)):
            return f"{v:.5g}" if np.isfinite(v) else "–"
        return v

    return df.map(cell)


def _table_html(df: pd.DataFrame) -> str:
    return _public(df).to_html(index=False, na_rep="–", border=0, escape=True)


def report_html(title: str, meta: dict, sections: list[dict]) -> str:
    """sections: dicts with heading and any of text, items, equations, table, figure."""
    from plotly.io import to_html

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             "<p class='sub'>ReactorGenome · One Digital Twin for Batch, CSTR, PFR and RTD Labs</p>",
             "<div class='meta'>"]
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    for k, v in list(meta.items()) + [("Generated", stamp)]:
        parts.append(f"<b>{html.escape(str(k))}</b><span>{html.escape(str(v))}</span>")
    parts.append("</div>")
    first = True
    for s in sections:
        parts.append(f"<h2>{html.escape(s['heading'])}</h2>")
        if s.get("text"):
            parts.append(f"<p>{html.escape(s['text'])}</p>")
        if s.get("equations"):
            parts.append("<pre>" + html.escape("\n".join(s["equations"])) + "</pre>")
        if s.get("items"):
            lis = []
            for it in s["items"]:
                if isinstance(it, dict):
                    lis.append(f"<li class='{it.get('level', '')}'>{html.escape(it['text'])}</li>")
                else:
                    lis.append(f"<li>{html.escape(str(it))}</li>")
            parts.append("<ul>" + "".join(lis) + "</ul>")
        if s.get("table") is not None and len(s["table"]):
            parts.append(_table_html(s["table"]))
        if s.get("figure") is not None:
            parts.append(to_html(_light(s["figure"]), include_plotlyjs=True if first else False, full_html=False,
                                 config={"displaylogo": False}))
            first = False
    parts.append("<p class='foot'>Produced by ReactorGenome. Model results depend on the assumptions listed above; "
                 "experimental values are only as good as the readings typed in.</p></body></html>")
    return "".join(parts)


def bundle(title: str, meta: dict, sections: list[dict], tables: dict, params: dict) -> bytes:
    """ZIP with report.html, tables/*.csv, plots/*.csv and parameters.json."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.html", report_html(title, meta, sections))
        z.writestr("parameters.json", params_json(params))
        for name, df in tables.items():
            if df is not None and len(df):
                z.writestr(f"tables/{name}.csv", df[[c for c in df.columns if not str(c).startswith("_")]].to_csv(index=False))
        for i, s in enumerate(sections):
            if s.get("figure") is not None:
                safe = "".join(ch if ch.isalnum() else "_" for ch in s["heading"]).strip("_")[:60]
                z.writestr(f"plots/{i:02d}_{safe}.csv", figure_frame(s["figure"]).to_csv(index=False))
    return buf.getvalue()
