# Paper material bundle: Laplace-law baseline table + uncertainty + field contours

Material for `paper-draft-tensorsph-kernel-verification` (mission `paper-mat-tensorsph-kernel-verification-1`,
TensorSPH branch `agent/m3043`).

## What is here

| path | content |
|---|---|
| `run.py` | the only measurement program: ensemble sweep → `out/runs/*.json`, `out/results.json`, `out/tables.md`, figures |
| `make_doc.py` | renders `docs/paper_laplace_uncertainty_material.md` from `out/results.json` (no hand-typed numbers) |
| `out/runs/run_*.json` | one file per (resolution, ensemble member): raw pressures, curvature, cut force, profile |
| `out/results.json` | aggregate table, three pressure metrics, 95 % CIs, three-state verdicts, contour field data |
| `out/tables.md` | the same table rendered as markdown |
| `out/fig1_pressure_field.png` | pressure-field contour of the static drop |
| `out/fig2_curvature_field.png` | curvature-error field `|kappa| - 1/R` |
| `out/fig3_uncertainty.png` | relative error vs `R/dx` with 95 % CI error bars, gates 3 %/15 % |
| `out/fig4_baseline_compare.png` | radial pressure profile vs analytic, and time development |

## Reproduce

```bash
PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/run.py \
    --dx-list 0.003 0.0025 0.002 0.0015 0.001 --realisations 5
PYTHONPATH=src python benchmarks/paper_laplace_uncertainty/make_doc.py
PYTHONPATH=src pytest tests/test_paper_laplace_uncertainty.py -q
```

Runs are cached in `out/runs/`; delete a file to recompute that member. Each
member takes 12 s (R/dx = 6.7) to 150 s (R/dx = 20).

## Definitions (fixed here so the tables cannot drift)

* theory: `Delta p = sigma/R` (2-D, one principal curvature).
* metric A `band`: `mean(p, r<0.5R) - mean(p, 0.8R<r<0.95R)` — samples the pressure at the interface.
* metric B `core_to_shell`: `mean(p, r<0.5R) - mean(p, 0.95R<r<1.05R)` — uses analytic `p_out = 0`,
  only valid if the interior pressure is uniform.
* metric C `force_implied`: `|F_cut|/(2R)` with `F_cut = sum_{x>0} m a_st,x` (analytic `|F_cut| = 2 sigma`);
  independent of any pressure measurement.
* error = `(measured - theory)/theory * 100 %`; numerator and denominator are tabulated explicitly.
* 95 % CI = `t_{0.975,n-1} * s / sqrt(n)` over the ensemble of lattice placements.
* verdict: `PASS` if `|e| + CI < 3 %`, `FAIL` if `|e| >= 15 %`, else `WITHHELD`.

The ensemble varies the **lattice placement** with the drop centre fixed. Rotating
the lattice is *not* an ensemble axis: for an axisymmetric drop a global rotation is
an exact symmetry and reproduces the same numbers bit for bit (checked; rotation
members were removed after they returned identical values).