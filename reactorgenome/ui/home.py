"""Landing page: what ReactorGenome is, four experiments, one design studio."""

from __future__ import annotations

import streamlit as st

from . import shell
from .style import AMBER, BLUE, CYAN, MUTED, TEAL, TEXT, VIOLET

_TANK = f"""<svg viewBox="0 0 260 150"><defs><linearGradient id="hl1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{TEAL}" stop-opacity=".55"/><stop offset="1" stop-color="{BLUE}" stop-opacity=".35"/></linearGradient></defs>
<path d="M10 40 H80" stroke="{BLUE}" stroke-width="5" stroke-dasharray="8 8"><animate attributeName="stroke-dashoffset" from="16" to="0" dur="1s" repeatCount="indefinite"/></path>
<path d="M10 70 H80" stroke="{AMBER}" stroke-width="5" stroke-dasharray="8 8"><animate attributeName="stroke-dashoffset" from="16" to="0" dur="1s" repeatCount="indefinite"/></path>
<rect x="80" y="18" width="100" height="120" rx="16" fill="#0E1830" stroke="#3B4B66" stroke-width="2"/>
<rect x="85" y="40" width="90" height="93" rx="12" fill="url(#hl1)"/>
<line x1="130" y1="8" x2="130" y2="108" stroke="#CBD5E1" stroke-width="3"/>
<rect x="110" y="104" width="40" height="7" rx="3.5" fill="#E2E8F0"><animate attributeName="width" values="40;4;40" dur="0.8s" repeatCount="indefinite"/><animate attributeName="x" values="110;128;110" dur="0.8s" repeatCount="indefinite"/></rect>
<path d="M180 118 H250" stroke="{TEAL}" stroke-width="5" stroke-dasharray="8 8"><animate attributeName="stroke-dashoffset" from="16" to="0" dur="1s" repeatCount="indefinite"/></path>
</svg>"""

_COIL = f"""<svg viewBox="0 0 260 150"><path d="M10 75 H40" stroke="{BLUE}" stroke-width="5"/>
<path d="M40 75 C55 20, 65 20, 75 75 S95 130, 110 75 S130 20, 145 75 S165 130, 180 75 S200 20, 215 75" fill="none" stroke="{VIOLET}" stroke-width="9" stroke-linecap="round" opacity=".35"/>
<path d="M40 75 C55 20, 65 20, 75 75 S95 130, 110 75 S130 20, 145 75 S165 130, 180 75 S200 20, 215 75" fill="none" stroke="{CYAN}" stroke-width="3.5" stroke-dasharray="10 10"><animate attributeName="stroke-dashoffset" from="20" to="0" dur="0.9s" repeatCount="indefinite"/></path>
<path d="M215 75 H250" stroke="{TEAL}" stroke-width="5"/></svg>"""

_BEAKER = f"""<svg viewBox="0 0 260 150"><defs><linearGradient id="hl2" x1="0" y1="0" x2="0" y2="1"><stop offset="0"><animate attributeName="stop-color" values="{BLUE};{TEAL};{BLUE}" dur="6s" repeatCount="indefinite"/></stop><stop offset="1" stop-color="#10233B"/></linearGradient></defs>
<path d="M85 18 V122 Q85 136 99 136 H161 Q175 136 175 122 V18" fill="#0E1830" stroke="#3B4B66" stroke-width="3"/>
<path d="M89 50 V120 Q89 132 101 132 H159 Q171 132 171 120 V50 Z" fill="url(#hl2)"/>
<line x1="158" y1="4" x2="158" y2="96" stroke="{CYAN}" stroke-width="3"/><rect x="152" y="92" width="12" height="20" rx="3" fill="#0B1220" stroke="{CYAN}" stroke-width="2"/>
<rect x="112" y="122" width="36" height="6" rx="3" fill="#E2E8F0"><animate attributeName="width" values="36;6;36" dur="0.6s" repeatCount="indefinite"/><animate attributeName="x" values="112;127;112" dur="0.6s" repeatCount="indefinite"/></rect>
<text x="205" y="70" fill="{MUTED}" font-size="12" font-family="IBM Plex Mono">κ(t)</text>
<path d="M195 85 C205 100, 220 108, 250 112" stroke="{CYAN}" stroke-width="2.5" fill="none"/></svg>"""

_PULSE = f"""<svg viewBox="0 0 260 150"><path d="M14 120 H246" stroke="#3B4B66" stroke-width="1.5"/><path d="M20 120 V24" stroke="#3B4B66" stroke-width="1.5"/>
<path d="M20 120 C40 120, 50 30, 70 30 C100 30, 130 105, 246 118" fill="none" stroke="{AMBER}" stroke-width="3"/>
<path d="M20 120 C50 120, 70 60, 100 55 C130 52, 160 100, 246 119" fill="none" stroke="{CYAN}" stroke-width="2.5" stroke-dasharray="6 6"/>
<circle r="5" fill="{AMBER}"><animateMotion dur="3.5s" repeatCount="indefinite" path="M20 120 C40 120, 50 30, 70 30 C100 30, 130 105, 246 118"/></circle>
<text x="150" y="44" fill="{MUTED}" font-size="12" font-family="IBM Plex Mono">E(t)</text></svg>"""

_STUDIO = f"""<svg viewBox="0 0 560 180">
<g font-family="IBM Plex Sans" font-size="12" fill="{TEXT}">
<rect x="10" y="66" width="78" height="44" rx="10" fill="#16223A" stroke="{BLUE}"/><text x="49" y="93" text-anchor="middle">Feed</text>
<rect x="130" y="56" width="96" height="64" rx="12" fill="#16223A" stroke="{CYAN}"/><text x="178" y="93" text-anchor="middle">CSTR</text>
<rect x="268" y="56" width="96" height="64" rx="12" fill="#16223A" stroke="{VIOLET}"/><text x="316" y="93" text-anchor="middle">PFR</text>
<rect x="404" y="66" width="70" height="44" rx="10" fill="#16223A" stroke="{MUTED}"/><text x="439" y="93" text-anchor="middle">Split</text>
<rect x="500" y="66" width="52" height="44" rx="10" fill="#16223A" stroke="{TEAL}"/><text x="526" y="93" text-anchor="middle">Out</text></g>
<g fill="none" stroke-width="3" stroke-dasharray="9 9">
<path d="M88 88 H130" stroke="{CYAN}"><animate attributeName="stroke-dashoffset" from="18" to="0" dur="1s" repeatCount="indefinite"/></path>
<path d="M226 88 H268" stroke="{CYAN}"><animate attributeName="stroke-dashoffset" from="18" to="0" dur="1s" repeatCount="indefinite"/></path>
<path d="M364 88 H404" stroke="{CYAN}"><animate attributeName="stroke-dashoffset" from="18" to="0" dur="1s" repeatCount="indefinite"/></path>
<path d="M474 88 H500" stroke="{CYAN}"><animate attributeName="stroke-dashoffset" from="18" to="0" dur="1s" repeatCount="indefinite"/></path>
<path d="M439 110 C439 165, 316 165, 300 140 C290 125, 200 165, 178 120" stroke="{AMBER}"><animate attributeName="stroke-dashoffset" from="18" to="0" dur="1s" repeatCount="indefinite"/></path></g>
<text x="300" y="172" fill="{AMBER}" font-size="11" font-family="IBM Plex Mono" text-anchor="middle">recycle</text></svg>"""

CARDS = [
    ("Batch", "BATCH · κ(t)", "Batch reactor", "Conductivity against time. Fit first, second, power-law or reversible kinetics with confidence intervals and compare temperatures.", _BEAKER),
    ("CSTR", "CSTR · STEADY STATE", "Stirred tank", "Transient readings fitted to a steady plateau, non-ideal mixing (tanks, bypass, dead zones), jacket energy balance and Arrhenius.", _TANK),
    ("PFR", "PFR · HELICAL COIL", "Tubular reactor", "Profiles along the coil, the beaker-delay correction, plug versus dispersion versus laminar flow, recycle and adiabatic runs.", _COIL),
    ("RTD", "RTD · TRACER", "Residence time", "Tracer pulse to E(t), F(t), moments and skewness. Tanks-in-series and dispersion fits, and what the RTD means for conversion.", _PULSE),
]


def render():
    st.markdown(
        "<div class='rg-hero'><div><h1>Every CHE331 reactor rig, <span>one live model</span>.</h1>"
        "<p>Type the readings you take in the lab. ReactorGenome fits the kinetics, checks whether each run had really settled, "
        "and shows how the answer changes when an assumption changes: mixing, stoichiometry, reversibility, heat.</p>"
        "<p>Then go past what the rig can do: reactors in series, recycle, bypass, jackets with more than one steady state.</p></div>"
        f"<div class='rg-card' style='padding:10px'>{_STUDIO}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='rg-section'><h3>Experiments</h3><span>same layout on every page: parameters · live reactor · results · readings · analysis</span></div>",
                unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, (page, tag, title, text, art) in zip(cols, CARDS):
        with col:
            st.markdown(f"<div class='rg-card'>{art}<div class='tag'>{tag}</div><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)
            st.button(f"Open {title.lower()}", key=f"btn_open_{page}", on_click=shell.go, args=(page,), width="stretch")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='rg-studio'><div><div class='tag' style='font-family:IBM Plex Mono;color:{CYAN};font-size:.75rem;letter-spacing:.08em'>DESIGN STUDIO</div>"
        "<h2>Build a plant the lab does not have</h2><p>Drag feeds, CSTRs, coils, packed beds and splitters onto a canvas, wire them together, "
        "add recycle, bypass or a cooling jacket, and the whole network is solved with the same kinetics you fitted.</p></div>"
        f"<div>{_STUDIO}</div></div>", unsafe_allow_html=True)
    c = st.columns([1, 1, 1])[1]
    c.button("Open the design studio", key="btn_open_studio", type="primary", on_click=shell.go, args=("Design Studio",), width="stretch")
