# ReactorGenome

One Digital Twin for Batch, CSTR, PFR and RTD Labs (CHE331, NaOH + ethyl acetate).

Project report: [`docs/report/report.pdf`](docs/report/report.pdf) (LaTeX source alongside).

Each experiment page has the same layout: assumptions on the left, the reactor animated in the middle, results and diagnostics on the right, your readings table underneath, then analysis tabs (fits, assumption effects, parameter sweeps, comparisons, export). Every control has a **?** with a plain explanation. The Design Studio is a drag-and-connect board where tanks, tubes, packed beds, splitters, several feeds and recycles are solved as one network.

The readings tables start empty and follow the lab sheets. Type what you measured, or import a CSV.

- **CSTR and PFR:** one row per run with temperature, total flow, τ (optional; worked out from V/Q when blank) and up to four outlet readings, plus a small table with Λ0 for each temperature. In the CSTR you can also note the time of each reading, which lets the app check whether each run had settled. The Results tab shows Ea, A and k at every temperature with all the plots, and what the fitted kinetics predict at any temperature, concentration and τ. The rate law used for the fit (order 1 to 3) is a setting in the left panel.
- **RTD:** one time column and up to four runs side by side, each with its own name, vessel, flow and volume, plus an optional calibration (typed line or a fit to standards).
- **Batch:** time and conductivity at each temperature.

## Run it

```bat
run.bat
```

Or:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the address it prints. On the lab Wi-Fi, other laptops can use the same address. A page can be opened directly with `?page=cstr` (or `batch`, `pfr`, `rtd`, `design-studio`).

**Project → Save** writes every table, setting and the design board to one `.reactorgenome.json` file; **Project → Load** restores it. `python -m streamlit run app.py -- my.reactorgenome.json` opens a saved project at start.

## Layout

```
app.py                      routing only
reactorgenome/engine/       models and solvers, no Streamlit (units, rate, cstr, steady, batch, pfr, rtd, flowsheet, labdata, report)
reactorgenome/ui/           one module per page, plus shell, widgets, charts, compute (cached engine calls), export
reactorgenome/components/   the animation player and the design board (plain HTML/JS)
tests/                      pytest suite
tools/make_validation.py    regenerates docs/VALIDATION.md
```

## Checks

```bash
python -m pytest tests            # engine, exports, project save/restore, every page with and without readings
python tools/make_validation.py   # rewrites docs/VALIDATION.md from fresh numbers
```

`docs/VALIDATION.md` lists the analytic checks, the comparison with the lab reports (used only for checking), and every place where the reports, the manuals and the app use different conventions.
