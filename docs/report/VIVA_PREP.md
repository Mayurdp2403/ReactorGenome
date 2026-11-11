# ReactorGenome — Viva / Presentation Prep

Use this file to study. It is written so you can **speak** the answers, not recite code.
Live app: https://che331.streamlit.app  
Report: `docs/report/report.pdf`  
Repo: https://github.com/Mayurdp2403/ReactorGenome

---

## 1. Thirty-second pitch (memorise this)

> CHE331 runs Batch, CSTR, PFR and RTD under fixed conditions. We write readings down and analyse later, so we never *see* what is going wrong inside the reactor during the lab. ReactorGenome is a web app that sits on a laptop next to the rig. Students type the same readings they already take. The app fits kinetics, checks whether a run really reached steady state, lets you switch assumptions on and off, and then uses your fitted *k* to design plants the lab cannot build — series, recycle, jackets with multiple steady states. No new sensors. One app for all four labs.

---

## 2. The problem (why this project exists)

**What the lab does today**
- One reactor, three flows, three temperatures, equal dilute NaOH + ethyl acetate feeds.
- Readings written on paper / sheet, analysis after the slot.
- Reports answer “why does X fall with flow?” or “what does a dead zone do?” in *words*, never by trying it.

**What went wrong in our own CSTR data**
- Conductivity was still drifting when we called it “steady”.
- 28 °C gave higher conversion than 40 °C — physically suspicious.
- We wrote “unstable readings” and moved on because there was no tool to diagnose it.

**What we promised**
A laptop web app (no new hardware) that:
1. Takes lab-sheet readings  
2. Fits them properly  
3. Updates plots live  
4. Lets you drop textbook assumptions one by one  
5. Tries setups the lab cannot run (series, recycle, non-isothermal scale-up)

---

## 3. What you built (high level only)

| Layer | What it is | What to say if asked |
|--------|------------|----------------------|
| Front end | **Streamlit** (Python web UI) | “Students open a browser; no install of a desktop app.” |
| Maths / solvers | **NumPy + SciPy** | “Same equations as Fogler / Levenspiel; numerical ODE and fitting.” |
| Plots | **Plotly** | “Interactive graphs; camera button saves PNG for the report.” |
| Design board | Small HTML/JS canvas | “Drag feeds, tanks, tubes, splitters; solved as one network.” |
| Deployed at | Streamlit Cloud | https://che331.streamlit.app |

**Architecture in one sentence:**  
An *engine* (models only) + *pages* (Batch, CSTR, PFR, RTD, Design Studio). Same layout on every experiment page.

**Same layout everywhere**
- **Left:** assumptions and run conditions  
- **Middle:** animated reactor + live curves  
- **Right:** results and diagnostics  
- **Below:** readings table (like the lab sheet)  
- **Tabs:** fits, assumption effects, sweeps, export  

---

## 4. Chemistry and lab facts (they will ask)

**Reaction:** NaOH + ethyl acetate → sodium acetate + ethanol (saponification).

**Why conductivity?**  
NaOH (OH⁻) is highly conductive; as it is consumed, conductivity falls. We convert conductivity Λ to conversion *X*:

\[
X = \frac{\Lambda_0 - \Lambda}{\Lambda_0 - \Lambda_\infty}
\]

- **Λ₀** = feed / initial conductivity (before reaction, or at mixing)  
- **Λ∞** = conductivity if reaction went to completion  

**Default Λ∞ in the app:** from limiting ionic conductivities → roughly **Λ∞ ≈ 0.366 Λ₀** (acetate replaces hydroxide). Students can also type the manual’s fixed value (e.g. 0.1 or 3.75 mS/cm).

**Why this matters for viva:**  
Using Λ∞ that is too low (e.g. 0.1 mS/cm when Λ₀ ≈ 10) **inflates X** and **deflates k**. Fitting Λ∞ from the CSTR data itself gave ≈ 0.30 Λ₀ — close to the ion estimate — and *k* at 31 °C rose from ~3.2 to ~8.5 L/mol·min. That is a strong “insight” answer if they ask what the app found that the report missed.

**Rate law the manuals usually assume:** second order, −r_A = *k* C_A C_B (equimolar → −r_A = *k* C_A²).

**Damköhler number (say this cleanly):**  
Da = *k* C_A0^(n−1) τ — dimensionless: how fast reaction is vs how fast fluid moves through.

---

## 5. Core CheE concepts (short, speakable)

### CSTR
- Ideal: outlet = tank contents.  
- Design equation (2nd order, equimolar): τ = X / [*k* C_A0 (1−X)²] → plot **y = X/(1−X)² vs τ**; slope → *k*.  
- **Settling:** after a change, conductivity needs ~several τ_r, not “one τ”. From water start the slow mode is ~τ; after a flow change it settles faster.  
- **Non-ideal:** bypass (fresh feed skips reaction → lower X), dead volume (less active volume → lower X), tanks in series (moves toward PFR → higher X for same total V).

### PFR
- Ideal: plug flow along the coil.  
- Design: y = X/(1−X) vs τ for 2nd order equimolar.  
- **Sample delay:** reaction can continue in the beaker after the outlet → looks like a free intercept of many minutes; the app models that as an explicit delay.  
- Alternatives: axial dispersion (Péclet number Pe), laminar flow, recycle, adiabatic.

### Batch
- κ(t) → C_A(t) → fit *k*.  
- Integral plot: 1/C_A − 1/C_A0 vs t (2nd order) should be straight.  
- Differential method: slope of ln(−dC/dt) vs ln C ≈ order *n*. Our batch data fit **n ≈ 1**, not 2 — same as the lab report’s differential method. Say: “mixing and κ₀ dominate early points; after the window, first order wins by AIC.”

### RTD
- Tracer pulse → E(t), mean time t_m, variance σ².  
- **N = t_m²/σ²** (tanks in series). Ideal CSTR → N = 1.  
- Packed bed → high Pe (closer to plug flow).  
- **Tail cut-off:** if you stop recording while tracer is still leaving, σ² is too small → N too high. Our stirred-tank curves stop at ~4% of peak; adding a tail brings N from ~1.5 toward ~1.2.

### Arrhenius
\[
\ln k = \ln A - E_a/(RT)
\]
Plot ln *k* vs 1/T → slope −E_a/R. App also gives 95% intervals on E_a and *A*.

### Multiple steady states (energy balance)
Heat generation (exothermic kinetics) vs heat removal (jacket UA(T − T_a)). Curves can cross **three** times: two stable, one unstable → **runaway** depending on start-up. Dilute lab feeds barely heat; the app’s what-if uses concentrated feed + weak jacket to show the textbook picture.

---

## 6. How we checked it was correct (say this if asked “how do you know it works?”)

1. **Closed-form checks:** ideal CSTR/PFR X, tanks-in-series → plug flow, RTD moments, recycle flowsheet vs sequential solve (errors like 10⁻¹⁰–10⁻¹⁶).  
2. **Lab reports (2026):** typed the group sheets with *their* conventions; app reproduced report **k** and **E_a** to four figures.  
   - CSTR: E_a = **25.07 kJ/mol**  
   - PFR: E_a = **12.70 kJ/mol**  
   - RTD: t_m, σ², N, Pe match exactly  
3. Those report readings are **not** built into the app — empty tables on start. Used only for testing.

---

## 7. Every figure in the report — what it is and why it is there

Study these against `report.pdf`. If the professor points at a screenshot, this is what you say.

### Home (`home.png`)
**What:** Landing page; cards for each experiment + Design Studio.  
**Why:** Shows one app serves all four labs; entry point for students.

### CSTR overview (`cstr_overview.png`)
**What:** Full CSTR page — assumption switches, animated tank, KPIs, fitted E_a / *k* on the right.  
**Why:** Proves the “digital twin” layout: live model + diagnostics while you type.

### CSTR readings (`cstr_readings.png`)
**What:** Lab-sheet style table (T, flow, τ, Λ1–Λ4) + small Λ₀ table per temperature.  
**Why:** Matching the manual’s observation table was a proposal promise — “enter the readings”.

### CSTR results (`cstr_results.png`)
**What:** One screen with E_a, A, *k*(T), design plots per T, combined design plot, Arrhenius, X vs flow, X vs T, single-run *k* vs τ, parity, residuals.  
**Why:** Everything a lab report needs without Excel. **Single-run *k* vs τ:** flat → consistent data; sloping → bad Λ∞, unsteady runs, or delay.

### CSTR predictions (`cstr_predictions.png`)
**What:** Sliders for T, C_A0, τ → predicted X; plots of X(τ), *k*(T), and X vs C_A0 for orders 1–3.  
**Why:** Shows effect of temperature, concentration and **reaction order**. All orders fit at the lab C_A0; they separate only if you change concentration — that is how you tell order apart.

### CSTR assumptions (`cstr_assumptions.png`)
**What:** Bar chart: ideal X vs bypass / dead volume / 3 tanks / stagnant / reversible.  
**Why:** Proposal point 4 — flip assumptions and *see* the cost (e.g. 10% bypass drops X by ~0.03 in the example).

### CSTR energy (`cstr_energy.png`)
**What:** Heat generation vs removal (three intersections) + phase portrait (runaway basin).  
**Why:** Proposal’s “scale-up / multiple steady states” — what the dilute lab cannot show. Feed 5 mol/L, weak jacket.

### PFR overview / results / flow models
**Overview:** Coil animation, X(z), Re/Dean, fitted E_a = 12.70.  
**Results:** Same family of kinetic plots as CSTR for the tubular data.  
**Flow models:** Same *k* under plug / dispersion / laminar — measured points overlaid.  
**Why:** Shows non-ideal tube models without rebuilding the rig.

### Batch overview / fit
**Overview:** κ → C_A curve and fitted *k* with interval.  
**Fit:** Integral plot, differential (n≈1), residuals, AIC ranking of rate laws.  
**Why:** Order is a *choice*, not an assumption; AIC makes that honest.

### RTD overview / all runs / model fits
**Overview:** E(t) for one run + warning about cut-off tail.  
**All runs:** Four curves, E(θ), F(t).  
**Model fits:** Tanks-in-series vs dispersion ranked by error.  
**Why:** RTD connects “mixing quality” to N, Pe, and conversion.

### Design Studio (`design_studio.png`)
**What:** Drag-and-drop plant (e.g. CSTR then PFR), stream table, unit table, overall X = 0.545.  
**Why:** “What the lab cannot do” — series, recycle, bypass, two feeds, jackets; mass balance &lt; 1e−8.

### Export (`export.png`)
**What:** ZIP (HTML report + CSVs + plots + settings) and HTML-only.  
**Why:** Proposal deliverable — one-click export for the lab report. Also: **Project → Save** writes a `.reactorgenome.json` another student can **Load** (share a Design Studio plant).

---

## 8. How students and TAs use this in the 3-hour lab

### Before the slot (TA)
1. Open https://che331.streamlit.app on the lab Wi‑Fi (or a laptop with `run.bat` locally).  
2. Project name = group number / names.  
3. Remind students: **type what the manual already asks**; no extra sensors.  
4. Optional: load a previous group’s `.reactorgenome.json` if comparing designs.

### Minute-by-minute pattern (any of CSTR / PFR / Batch / RTD)

| Phase | Time | Student | App | TA role |
|--------|------|---------|-----|---------|
| Setup | 0–15 min | Set V, C_A0, Λ∞ mode (ion or typed), temperature | Left panel | Check concentrations match the bottle labels |
| Run 1 | 15–40 min | Start flow / mixing; wait; type readings when due | CSTR: stopwatch for 2–5 settling times; table fills | Don’t let them average a climbing conductivity |
| Live check | during | Watch transient / X / diagnostics | Middle + right panels | Ask: “Has it settled? What does the status say?” |
| More runs | 40–100 min | Change Q or T; repeat | Results tab updates Arrhenius as 2nd and 3rd T appear | Point at *k* vs τ plot if values drift |
| Assumptions | 100–130 min | Toggle bypass / dead volume / dispersion | Assumption effects / flow models | “Which assumption moves X the most?” |
| Beyond the rig | 130–160 min | Open Design Studio; load fitted *k*; try series / recycle | Design Studio presets | “Which arrangement gives higher X for same volume?” |
| Wrap-up | 160–180 min | Export ZIP + Save project JSON | Export / Project | Collect JSON if they want to continue at home |

### CSTR-specific coaching lines for TAs
- If X at low T &gt; X at high T → **look at whether readings were still rising**; use plateau / start-up shape, not blind average.  
- If single-run *k* falls as τ rises → suspect **Λ∞ too low**.  
- Equimolar + isothermal should stay on unless discussing heat effects.

### PFR-specific
- If free-line fit gives nonsense or negative *k* → switch to **y on τ through origin** or turn on **sample delay**.  
- Check Dean / Re so students see the coil is not a straight tube.

### Batch-specific
- Set fitting window to skip mixing (e.g. first 10 s) and the flat noisy tail.  
- If κ readings fall below κ∞ → app warns and drops those points; fix typed κ∞.

### RTD-specific
- Record until tracer is near baseline; if not, turn on **tail correction** and read the warning.  
- Compare N for tank (~1–2) vs bed (large N / high Pe).

### Design Studio in the last half hour
1. After CSTR/PFR fit, kinetics are **published** to the studio (second-order *k*, E_a).  
2. Preset **CSTR then PFR** → see overall X and stream table.  
3. Try **recycle** or **bypass** — conversion changes immediately.  
4. Optional: jacketed strong-exotherm what-if (not the real dilute feed) to show multiple steady states.  
5. Save project → share JSON with another group.

---

## 9. Design Studio — what to say in detail

**Purpose:** Solve a **flowsheet** (network), not one isolated reactor.

**Blocks:** Feed, CSTR, PFR, packed bed, Splitter, Product.  
**Connections:** pipes; splitter fraction → bypass or recycle.

**How it solves (conceptually, no code):**  
- Mass balances on every node.  
- Each unit uses the same −r_A = *k*(T) C_A C_B (optional reverse).  
- Recycle: iterate until compositions stop changing (Wegstein).  
- Report overall X, per-unit X, τ, T, jacket duty, mass-balance error.

**Tabs under the board**
- **Streams / Units** — tables for the report  
- **Profiles** — C_A and T vs cumulative volume along the main path (tank = step drop; tube = gradual)  
- **Levenspiel** — 1/(−r_A) vs X; area related to reactor size  
- **Sweep** — vary one parameter (V, Q, split fraction…)  
- **Compare rigs** — *k*(T) from Batch/CSTR/PFR on one plot + ideal X–Da curves  

**Demo numbers to remember:** CSTR (2 L) then PFR (0.5 L) at the default kinetics → overall **X ≈ 0.545**, balance error &lt; 1e−8.

---

## 10. Likely viva questions and model answers

**Q: What is ReactorGenome?**  
A: A browser digital twin for the four CHE331 reactor labs. Type existing readings; get fits, diagnostics, assumption checks, and a design sandbox.

**Q: Why not Aspen / COMSOL?**  
A: Those are heavy, licensed, and not built around *this* lab sheet and conductivity workflow. We needed something a group opens in minute one of a 3-hour slot on a laptop.

**Q: What is new vs Excel?**  
A: Live transient + steady-state judgement, ion-based Λ∞, non-ideal models, shared *k* across rigs, flowsheet design, export of a full report pack — while typing, not after the lab.

**Q: How is conversion obtained?**  
A: From conductivity using Λ₀ and Λ∞. Default Λ∞ from ion conductivities; optional typed value or fit from data.

**Q: How do you get *k* and E_a?**  
A: From the design plot (y vs τ) at each T, then Arrhenius across temperatures, with 95% intervals.

**Q: What is Damköhler number here?**  
A: Da = *k* C_A0^(n−1) τ. Large Da → high conversion for a given reactor type.

**Q: Your CSTR E_a was 25 kJ/mol and PFR 12.7 — why different?**  
A: Different groups, different Λ∞ conventions, possible sample delay and scatter in PFR. The app matches *each report’s own method*; comparing rigs honestly needs the same Λ∞ and steady-state rules — that is what “Compare rigs” is for.

**Q: How do you know the maths is right?**  
A: Analytic limits + reproducing 2026 lab-report *k* and E_a to four significant figures; 130+ automated tests.

**Q: What did the app reveal that the reports missed?**  
A: (1) Λ∞ ≈ 0.3 Λ₀ fits CSTR better than 0.1 mS/cm. (2) Temperature inversion was largely unsteady averaging. (3) RTD N≈1.5 was mostly a cut-off tail. (4) Batch differential order ≈ 1.

**Q: Bypass vs dead volume?**  
A: Bypass: fraction of feed never enters reactor → dilutes product, lowers X. Dead volume: part of V is stagnant → effective τ smaller → lowers X. Both shown on Assumption effects.

**Q: Multiple steady states — when?**  
A: Strong exotherm + limited cooling (finite UA). Generation and removal curves can intersect three times; middle state unstable.

**Q: Can groups share work?**  
A: Yes — Project Save → `.reactorgenome.json` → other student Load project. Design Studio board and all tables restore.

**Q: Phone?**  
A: Works in a browser; we validated on laptop. Narrow screens clip some tabs — lab use is laptop-first.

**Q: Limitations (be honest)?**  
A: Dead volume / stagnant zones are compared more than fully fitted (bypass can be fitted). Energy balance is a what-if for dilute labs. Only second-order *k* is shared across pages. Chemistry is this saponification system.

**Q: How would a TA use it tomorrow?**  
A: Open the page for today’s experiment, watch the status/diagnostics while groups type, catch unsteady CSTR early, then in the last hour push them into Design Studio with their own *k*.

---

## 11. Numbers worth memorising

| Item | Value |
|------|--------|
| Live URL | https://che331.streamlit.app |
| CSTR report E_a | 25.07 kJ/mol |
| CSTR report *k* (31 / 46 / 60 °C) | 3.16 / 5.85 / 7.44 L/mol·min |
| PFR report E_a | 12.70 kJ/mol |
| PFR report *k* (40 / 50 / 60 °C) | 11.74 / 13.75 / 15.73 L/mol·min |
| Design Studio demo X (CSTR→PFR) | 0.545 |
| Default Λ∞ / Λ₀ (ions) | ≈ 0.366 |
| Reaction | NaOH + EtOAc |

---

## 12. Suggested 5-minute presentation flow

1. **Problem** (45 s) — drifting CSTR readings; answers only in words.  
2. **Demo Home → CSTR** (90 s) — type/show readings table → Results tab (E_a, plots).  
3. **Assumption effects** (30 s) — flip bypass, show bar chart.  
4. **Design Studio** (60 s) — CSTR then PFR, X = 0.545.  
5. **Lab use + share JSON** (45 s) — 3-hour workflow; Save/Load.  
6. **Close** (30 s) — validated against reports; deployed at che331.streamlit.app.

Keep the report PDF open on the figures page as backup if the live Wi‑Fi fails.

---

## 13. One-page cheat sheet (last look before you walk in)

- **What:** Digital twin web app for Batch / CSTR / PFR / RTD + Design Studio.  
- **Input:** Same readings as the manual (conductivity / TDS / time / flow / T).  
- **Output:** X, *k*(T), E_a, diagnostics, assumption costs, plant designs, ZIP export.  
- **Key insight:** Bad Λ∞ and unsteady averages explain weird report numbers better than “bad kinetics”.  
- **Lab use:** Laptop beside rig → type live → check settled → fit → Design Studio → export.  
- **Stack (if asked):** Python, Streamlit, NumPy/SciPy, Plotly; hosted on Streamlit Cloud.  
- **Not claiming:** New hardware, phone-perfect UI, or Aspen replacement for industry plants.

Good luck — speak the pitch, show one Results screen and the Design Studio, and you can defend the rest from this file.
