"""Visual system: one palette, one type scale, used by CSS, Plotly and the JS components."""

BG = "#0B1220"
PANEL = "#111A2B"
PANEL2 = "#16223A"
BORDER = "rgba(148,163,184,0.18)"
TEXT = "#E6EDF7"
MUTED = "#94A3B8"
CYAN = "#22D3EE"
TEAL = "#2DD4BF"
BLUE = "#60A5FA"
AMBER = "#F5B041"
RED = "#F87171"
GREEN = "#34D399"
VIOLET = "#A78BFA"
SLATE = "#94A3B8"

SERIES = [CYAN, AMBER, BLUE, TEAL, VIOLET, GREEN, RED]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, input, textarea, select, button {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}}
html {{ font-size: 16.5px; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, rgba(34,211,238,0.06), transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, rgba(96,165,250,0.05), transparent 60%), {BG}; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding-top: 0.9rem !important; padding-bottom: 3rem; max-width: 1640px; }}
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
p, li, label, .stMarkdown {{ color: {TEXT}; }}
code, .mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }}

/* ---------- app bar ---------- */
.rg-bar {{ display:flex; align-items:center; gap:14px; padding:6px 2px 2px; }}
.rg-logo {{ width:42px; height:42px; flex:none; }}
.rg-brand {{ font-size:1.45rem; font-weight:700; color:{TEXT}; line-height:1.1; }}
.rg-brand span {{ color:{CYAN}; }}
.rg-sub {{ font-size:0.86rem; color:{MUTED}; margin-top:2px; }}
.rg-pill {{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-size:0.85rem;
           border:1px solid {BORDER}; background:{PANEL}; color:{TEXT}; white-space:nowrap; }}
.rg-pill i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.rg-pill.ok i {{ background:{GREEN}; }} .rg-pill.warn i {{ background:{AMBER}; }} .rg-pill.err i {{ background:{RED}; }} .rg-pill.idle i {{ background:{SLATE}; }}
.rg-model {{ font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:{MUTED}; padding:6px 0 2px; }}
.rg-model b {{ color:{TEXT}; font-weight:500; }}

/* ---------- headings inside pages ---------- */
.rg-title {{ margin: 0.3rem 0 0.2rem; }}
.rg-title h1 {{ font-size:1.75rem; margin:0; font-weight:700; }}
.rg-title p {{ color:{MUTED}; margin:0.2rem 0 0; font-size:0.98rem; max-width:980px; }}
.rg-section {{ display:flex; align-items:baseline; gap:10px; margin:1.4rem 0 0.4rem; }}
.rg-section h3 {{ font-size:1.12rem; margin:0; font-weight:600; }}
.rg-section span {{ color:{MUTED}; font-size:0.9rem; }}
.rg-panel-title {{ font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:{MUTED}; margin:0 0 6px; font-weight:600; }}

/* ---------- containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{BORDER} !important; border-radius:14px !important; background:rgba(17,26,43,0.72); }}
[data-testid="stExpander"] details {{ border:1px solid {BORDER} !important; border-radius:12px !important; background:{PANEL}; }}
[data-testid="stExpander"] summary p {{ font-weight:600; font-size:0.98rem; }}

/* ---------- KPI cards ---------- */
.rg-kpis {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.rg-kpi {{ background:{PANEL2}; border:1px solid {BORDER}; border-radius:12px; padding:10px 12px; min-height:74px; }}
.rg-kpi .l {{ font-size:0.76rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em; }}
.rg-kpi .v {{ font-family:'IBM Plex Mono',monospace; font-size:1.32rem; color:{TEXT}; margin-top:2px; }}
.rg-kpi .v small {{ font-size:0.78rem; color:{MUTED}; margin-left:3px; }}
.rg-kpi .d {{ font-size:0.78rem; color:{MUTED}; margin-top:1px; }}
.rg-kpi.hi {{ border-color:rgba(34,211,238,0.45); }}
.rg-kpi.hi .v {{ color:{CYAN}; }}

/* ---------- notes ---------- */
.rg-note {{ border-left:3px solid {SLATE}; background:rgba(148,163,184,0.06); padding:8px 12px; border-radius:0 10px 10px 0; margin:6px 0; font-size:0.93rem; color:{TEXT}; }}
.rg-note.good {{ border-color:{GREEN}; background:rgba(52,211,153,0.07); }}
.rg-note.warn {{ border-color:{AMBER}; background:rgba(245,176,65,0.08); }}
.rg-note.err {{ border-color:{RED}; background:rgba(248,113,113,0.08); }}
.rg-note.info {{ border-color:{BLUE}; background:rgba(96,165,250,0.07); }}
.rg-empty {{ border:1px dashed {BORDER}; border-radius:12px; padding:22px; text-align:center; color:{MUTED}; font-size:0.95rem; }}
.rg-empty b {{ color:{TEXT}; display:block; margin-bottom:4px; font-size:1rem; }}
.rg-eq {{ font-family:'IBM Plex Mono',monospace; font-size:0.86rem; background:{PANEL2}; border:1px solid {BORDER}; border-radius:10px; padding:8px 10px; color:{TEXT}; margin:6px 0; }}

/* ---------- landing ---------- */
.rg-hero {{ display:grid; grid-template-columns: 1.15fr 1fr; gap:28px; align-items:center; padding:18px 4px 6px; }}
.rg-hero h1 {{ font-size:2.35rem; line-height:1.12; margin:0 0 10px; font-weight:700; }}
.rg-hero h1 span {{ color:{CYAN}; }}
.rg-hero p {{ color:{MUTED}; font-size:1.06rem; margin:0 0 6px; max-width:620px; }}
.rg-card {{ background:linear-gradient(180deg, {PANEL2}, {PANEL}); border:1px solid {BORDER}; border-radius:16px; padding:16px 16px 10px; height:100%; }}
.rg-card svg {{ width:100%; height:150px; display:block; }}
.rg-card h3 {{ margin:10px 0 4px; font-size:1.12rem; }}
.rg-card p {{ color:{MUTED}; font-size:0.92rem; margin:0; min-height:66px; }}
.rg-card .tag {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:{CYAN}; letter-spacing:0.06em; text-transform:uppercase; }}
.rg-studio {{ background:linear-gradient(120deg, rgba(34,211,238,0.10), rgba(96,165,250,0.06) 55%, rgba(245,176,65,0.05)); border:1px solid rgba(34,211,238,0.35);
              border-radius:18px; padding:18px 22px; display:grid; grid-template-columns: 1fr 1.35fr; gap:20px; align-items:center; }}
.rg-studio h2 {{ margin:4px 0 6px; font-size:1.5rem; }}
.rg-studio p {{ color:{MUTED}; margin:0 0 6px; }}
.rg-studio svg {{ width:100%; height:180px; }}

.rg-card h3, .rg-studio h2, .rg-hero h1, .rg-title h1, .rg-section h3 {{ padding:0 !important; }}
.rg-card h3 {{ margin-top:6px !important; }}

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button {{ border-radius:10px; font-weight:600; border:1px solid {BORDER}; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background:{CYAN}; color:#04202A; border-color:{CYAN}; }}
.stButton > button[kind="primary"]:hover {{ background:#67E8F9; border-color:#67E8F9; }}
[data-testid="stWidgetLabel"] p {{ font-size:0.93rem !important; color:{TEXT} !important; }}
[data-testid="stTabs"] button p {{ font-size:0.96rem; font-weight:600; }}
[data-testid="stTabs"] [aria-selected="true"] p {{ color:{CYAN}; }}
[data-baseweb="tab-highlight"] {{ background:{CYAN} !important; }}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {{ border:1px solid {BORDER}; border-radius:10px; }}
[data-testid="stMetric"] {{ background:{PANEL2}; border:1px solid {BORDER}; border-radius:12px; padding:10px 12px; }}
[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace; }}
div[data-testid="stSegmentedControl"] button p {{ font-weight:600; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{CYAN}; border-color:{CYAN}; }}
iframe {{ border-radius:14px; }}
</style>
"""

LOGO = f"""<svg class="rg-logo" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{CYAN}"/><stop offset="1" stop-color="{BLUE}"/></linearGradient></defs>
<rect x="2" y="2" width="44" height="44" rx="12" fill="{PANEL2}" stroke="url(#lg)" stroke-width="1.5"/>
<path d="M15 10 C 33 16, 15 22, 33 28 S 15 34, 33 38" fill="none" stroke="url(#lg)" stroke-width="2.6" stroke-linecap="round"/>
<path d="M33 10 C 15 16, 33 22, 15 28 S 33 34, 15 38" fill="none" stroke="{AMBER}" stroke-width="2" stroke-linecap="round" opacity="0.85"/>
<circle cx="24" cy="19" r="2.2" fill="{TEXT}"/><circle cx="24" cy="31" r="2.2" fill="{TEXT}"/>
</svg>"""
