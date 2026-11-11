"""Screenshots of the running app for the project report.

Starts the app on a spare port with the lab readings loaded as a project
file, drives Microsoft Edge with Playwright, and writes PNGs to figures/.

    python docs/report/make_figures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
PORT = 8513
URL = f"http://localhost:{PORT}"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import report_data as rd  # noqa: E402
from reactorgenome import data_lab  # noqa: E402


def _df(df: pd.DataFrame) -> dict:
    return {"__df__": json.loads(df.to_json(orient="split"))}


def project() -> dict:
    cstr = rd.cstr_frame()
    pfr = rd.pfr_frame()
    b = data_lab.BATCH_50
    batch = pd.DataFrame({"T_C": b["T"], "t_s": b["t_s"], "kappa": np.asarray(b["ec"], float) / 1000})
    runs = pd.DataFrame({"name": list(rd.RTD_RUNS), "vessel": ["Stirred tank", "Stirred tank", "Packed bed", "Packed bed"],
                         "Q_LPH": [15.0, 10.0, 15.0, 10.0], "V_L": [np.nan] * 4})
    return {
        "project_name": "CHE331 lab readings",
        "ws_cstr": _df(cstr.drop(columns=["lam0"])), "ws_cstr_lam0": _df(cstr[["T_C", "lam0"]].dropna().reset_index(drop=True)),
        "c_V": rd.CSTR_V, "c_CA0": rd.CSTR_CA0, "c_lmode": "fixed", "c_linf": rd.CSTR_LINF, "c_Q": 20.0, "c_T": 31.0,
        "ws_pfr": _df(pfr.drop(columns=["lam0"])), "ws_pfr_lam0": _df(pfr[["T_C", "lam0"]].drop_duplicates().reset_index(drop=True)),
        "p_V": rd.PFR_V, "p_CA0": rd.PFR_CA0, "p_lm": "fixed", "p_linf": rd.PFR_LINF, "p_Q": 30.0, "p_T": 50.0,
        "ws_rtd": _df(rd.rtd_sheet()), "ws_rtdruns": _df(runs),
        "ws_batch": _df(batch), "b_C0": b["C0"], "b_T": b["T"], "b_tlo": b["window"][0], "b_thi": b["window"][1],
        "b_lm": "fixed", "b_linf": b["lam_inf"] / 1000, "b_dur": 6.0,
    }


def energy_project() -> dict:
    """The Design Studio's strong-exotherm what-if on the CSTR page: 5 mol/L feed, weak jacket."""
    k30 = 1e12 * np.exp(-90e3 / (8.314462618 * 303.15))
    return {**project(), "c_jacket": True, "c_mode": "steady", "c_V": 2.15, "c_CA0": 5.0, "c_Q": 24.0, "c_T": 30.0,
            "c_ksrc": "typed", "c_k": k30, "c_Tr": 30.0, "c_Ea": 90.0, "c_dH": -100.0, "c_UA": 40.0, "c_Ta": 30.0, "c_Ti": 30.0}


def wait_idle(page, settle=2.5):
    time.sleep(1.5)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180_000)
    page.wait_for_function("() => document.querySelectorAll('.stSpinner').length === 0", timeout=180_000)
    time.sleep(settle)


def box(loc):
    b = loc.bounding_box()
    return b["x"], b["y"], b["width"], b["height"]


def shot(page, name, top, bottom=None, until="bottom", pad=12):
    """Clip from the top of one element to the top or bottom edge of another (or of itself)."""
    top.evaluate("e => e.scrollIntoView({block: 'start'})")
    time.sleep(2.0)
    x0, y0, w0, h0 = box(top)
    y1 = y0 + h0
    if bottom is not None:
        x1, by, w1, bh = box(bottom)
        y1 = by + bh if until == "bottom" else by - 2 * pad
    clip = {"x": 0, "y": max(y0 - pad, 0), "width": page.viewport_size["width"], "height": min(y1 - y0 + 2 * pad, page.viewport_size["height"] - y0)}
    for attempt in range(3):
        try:
            page.screenshot(path=str(OUT / f"{name}.png"), clip=clip, timeout=45_000, animations="disabled")
            break
        except Exception:
            print("retry", name)
            page.mouse.move(5, 5 + attempt)
            time.sleep(5)
    print("wrote", name, int(clip["height"]))


def goto(br, slug):
    """A fresh tab per page: a tab reused after the home page stops producing frames in headless Edge."""
    page = br.new_page(viewport={"width": 1600, "height": 3600}, device_scale_factor=1.25)
    page.goto(f"{URL}/?page={slug}")
    page.wait_for_selector("[data-testid='stMainBlockContainer'] .rg-kpis, .rg-title, .rg-hero, h1", timeout=180_000)
    wait_idle(page)
    return page


def tab(page, name):
    page.get_by_role("tab", name=name, exact=True).click()
    wait_idle(page, 2.0)
    return page.get_by_role("tabpanel").filter(visible=True).first


def title(page, text):
    return page.locator(".rg-panel-title", has_text=text).first


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    if not only or only - {"energy"}:
        session(project(), lambda br: run(br, only - {"energy"}))
    if not only or "energy" in only:
        session(energy_project(), run_energy)


def session(proj, runner):
    pj = Path(tempfile.gettempdir()) / "rg_report_project.reactorgenome.json"
    pj.write_text(json.dumps(proj))
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT), "--server.headless", "true",
                            "--", str(pj)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            br = p.chromium.launch(channel="msedge")
            runner(br)
            br.close()
    finally:
        srv.terminate()
        pj.unlink(missing_ok=True)


def top_region(page, name):
    head = page.locator(".rg-title").first
    block = page.locator("[data-testid='stHorizontalBlock']").filter(has=page.locator(".rg-panel-title", has_text="Assumptions")).first
    shot(page, name, head, block)


def run(br, only):
    want = (lambda s: not only or s in only)
    if want("home"):
        page = goto(br, "home")
        shot(page, "home", page.locator("[data-testid='stMainBlockContainer']").first)
        page.close()
    if want("cstr"):
        _cstr(goto(br, "cstr"))
    if want("pfr"):
        _pfr(goto(br, "pfr"))
    if want("batch"):
        _batch(goto(br, "batch"))
    if want("rtd"):
        _rtd(goto(br, "rtd"))
    if want("design"):
        page = goto(br, "design-studio")
        page.locator(".st-key-btn_preset_series button").click()
        wait_idle(page, 4.0)
        shot(page, "design_studio", page.locator("[data-testid='stMainBlockContainer']").first)


def collapse(page, *labels):
    for lab in labels:
        page.locator("[data-testid='stExpander'] summary", has_text=lab).first.click()
    time.sleep(1.5)


def _cstr(page):
    collapse(page, "Start-up", "Fitting your readings")
    top_region(page, "cstr_overview")
    shot(page, "cstr_readings", page.locator(".rg-section", has_text="Readings").first, page.get_by_role("tablist").first, until="top")
    panel = tab(page, "Results")
    shot(page, "cstr_results", panel, title(page, "What your fitted kinetics predict"), until="top")
    shot(page, "cstr_predictions", title(page, "What your fitted kinetics predict"), panel)
    panel = tab(page, "Fit details")
    shot(page, "cstr_fit_details", panel)
    panel = tab(page, "Assumption effects")
    shot(page, "cstr_assumptions", panel)
    panel = tab(page, "Export")
    shot(page, "export", panel)
    page.close()


def run_energy(br):
    page = goto(br, "cstr")
    panel = tab(page, "Energy balance")
    shot(page, "cstr_energy", panel)
    page.close()


def _pfr(page):
    collapse(page, "Fitting your readings")
    top_region(page, "pfr_overview")
    panel = tab(page, "Results")
    shot(page, "pfr_results", panel, title(page, "What your fitted kinetics predict"), until="top")
    shot(page, "pfr_predictions", title(page, "What your fitted kinetics predict"), panel)
    panel = tab(page, "Flow models")
    shot(page, "pfr_flow_models", panel)
    page.close()


def _batch(page):
    top_region(page, "batch_overview")
    panel = tab(page, page.get_by_role("tab").first.inner_text().strip())
    shot(page, "batch_fit", panel)
    page.close()


def _rtd(page):
    top_region(page, "rtd_overview")
    panel = tab(page, "All runs")
    shot(page, "rtd_all_runs", panel)
    panel = tab(page, "Model fits")
    shot(page, "rtd_model_fits", panel)
    page.close()


if __name__ == "__main__":
    main()
