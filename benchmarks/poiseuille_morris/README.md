# benchmarks/poiseuille_morris — CASE-SPH-POISEUILLE-MORRIS

Plane Poiseuille flow in the configuration of the Morris–Fox–Zhu (1997,
*J. Comput. Phys.* **136**:214–226) low-Reynolds channel: body-force-driven flow
between two no-slip plates, periodic in `x`, run to a steady state and compared
with the exact solution.

| file | role |
|---|---|
| `DESIGN.md` | predeclared design: decision, alternatives, hypothesis, metrics, thresholds, stopping rule, confounders, controls |
| `run.py` | the sweep: resolution ladder, Reynolds ladder, support ladder, periodic-length ladder, both solver routes, controls → `runs/poiseuille_morris/results.json` |
| `make_figs.py` | figures 1–5 from the results JSON |
| `make_doc.py` | renders `docs/benchmarks/poiseuille-morris.md` (numbers injected from the JSON, zero hand copy) |
| `publish_report.py` | posts the progress report to the platform (`/tensor/reports`) |
| `../../tests/test_poiseuille_morris.py` | reference-solution consistency, `Re_c` mapping, gate regression, zero-forcing control, artefact thresholds |

Reference numbers are never typed in: `run.py::reference_profile` evaluates
`u = g y (H − y)/(2ν)`, `u_max = gH²/(8ν)`, `τ_w = ρgH/2`. The 1997 article is
closed access, so its printed table is graded **WITHHELD** rather than copied
(`results.json → reference.authority_access`).

```bash
PYTHONPATH=src:. python benchmarks/poiseuille_morris/run.py --out runs/poiseuille_morris --threads 1
PYTHONPATH=src:. python benchmarks/poiseuille_morris/make_figs.py
PYTHONPATH=src:. python benchmarks/poiseuille_morris/make_doc.py
PYTHONPATH=src:. python -m pytest tests/test_poiseuille_morris.py -q
```

`--threads 1` matters: with these tensor sizes multi-threaded torch is ~7×
slower than a single thread (the same effect is recorded in
`benchmarks/poiseuille/run.py`).