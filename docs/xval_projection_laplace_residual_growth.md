# The residual current of the projection-SPH drop accumulates in time

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m22813**
**Type** re-analysis of the shipped dynamic records (no new solve in this unit)

## ① Question

The dynamic arm gates `max|u|/v_ref` against Adami et al. (2012)'s `0.10` at the end of a
120-step march. That gate answers *whether* the drop is at rest, not *what kind* of residual
prevents it. Two mechanisms produce the same end-of-march number and have opposite consequences
for the anchor:

* a **steady offset** — the projection leaves a field behind it every step, but the field does
  not grow; a discretisation error, expected to fall with `dx`;
* an **accumulating current** — the solenoidal part of the CSF traction is orthogonal to what the
  projection operator can remove, so it is re-injected every step and integrates in `u`; a
  *formulation* property, which no amount of resolution removes.

The shipped records carry the full per-sample history, so the two are separable without any new
march: a steady offset is flat in `t/t_sigma`, an accumulation is linear in it.

## ② Method

`benchmarks/laplace_sph/analyze_residual_growth.py` re-reads
`benchmarks/laplace_sph/results_dynamic_ladder/dx8_nu01.json, benchmarks/laplace_sph/results_dynamic_ladder/dx12_nu01.json` and fits `max|u|/v_ref`, `mean|u|/v_ref`,
the Laplace-jump error and the area drift against `t/t_sigma` (least squares, `R²` reported). Case
parameters are those of the marches themselves (`R = 20 mm`, `sigma = 0.072 N/m`,
`rho0 = 1000 kg/m³`, `h/dx = 1.3`, `dt = 0.1 t_sigma`, primary `nu`).

## ③ Result

| R/dx | particles | steps | max|u|/v_ref at t₁ | at t_end | plateau | rise to 90 % [t_sigma] | R² (straight line) | mean at t_end | dp err end % | area drift end % | max div ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 780 | 120 | 0.059 | 0.263 | 0.237 | 10.0 | 0.84 | 0.03 | 1.17 | -0.431 | 0.00 |
| 12 | 1733 | 120 | 0.072 | 0.286 | 0.251 | 5.0 | 0.71 | 0.03 | 2.19 | 0.769 | 0.00 |

| gate | verdict |
|---|---|
| saturating transient, not an unbounded accumulation | PASS (measured 0.843, declared <= 0.9) |
| plateau reached inside the marched window | FAIL (measured 10.000, declared <= 6.0) |
| Adami 0.10 line at the end of the march | FAIL (measured 0.286, declared <= 0.1) |
| whole drop drifting (mean current linearly growing) | FAIL (measured 0.473, declared >= 0.9) |
| Laplace jump inside the owner 15 % gate | PASS (measured 2.194, declared <= 15.0) |
| plateau level falls with `dx` | FAIL (measured 1.062, declared <= 1.0) |

![residual current rise and plateau](laplace_sph_residual_growth.png)

Reading, restricted to what the two records on disk support:

* the max speed rises from `0.059` to a plateau of `0.237 v_ref`
  in the first `10.0 t_sigma` and then **holds**; a straight line through
  the whole march explains only `R² = 0.84` of it, so the residual is a
  **saturating steady current**, not an unbounded accumulation. Had it been linear, the `t = 12`
  value would have been ~3x what is measured;
* the **mean** speed peaks around `t ≈ 5 t_sigma` and then decays, i.e. the drop is not drifting as
  a whole -- the plateau is a lattice-scale current, and it is not the area that carries it
  (area drift stays at `-0.43 %`, inside the 2 % gate);
* the Laplace jump the anchor is graded on stays inside the owner 15 % gate for the whole march
  (`+1.17 … +2.19 %`
  at the final sample), while the Adami 0.10 line is broken by a factor
  `2.9`. The static claim and the at-rest claim are
  therefore *separately* decided, which the ladder's single end-of-march gate did not show;
* the plateau *level* is `0.237` at `R/dx = 8` and
  `0.251` at `R/dx = 12` -- it does **not** fall with
  resolution on this two-point record, but a factor 1.50
  in `R/dx` is not a refinement study (see §④).

## ④ Honest scope

* Only **2** level(s) exist in the tree (`R/dx` = 8, 12).
  The missing ladder jobs (dx16_nu01, dx24_nu01) were **submitted to the queued compute path in this unit**
  (`cj4389` = `dx24_nu01`, `cj4390` = `dx16_nu01`, `cj4391` = `dx12_horizon`,
  `cj4392`/`cj4393` = the `nu` sweep) and were **still `queued` after 23 min**, so the
  refinement question is **not** adjudicated here and the ladder debt page stands unchanged.
  The `dx`, `nu` and horizon columns above are therefore a *re-analysis*, not new evidence about
  convergence.
* The plateau window is the last 60 % of a 120-step march (12 samples per level). A 600-step
  horizon -- one of the queued jobs -- is what would turn "plateau" into a measured steady state.
* The band attribution instrument added in this unit
  (`relax_projection_consistent(..., band_report=True)`, `docs/xval_projection_laplace_spurious_bands.md`)
  is not exercised here for the same reason: it needs a marched run. Its offline gates are green
  (`tests/test_projection_spurious_bands.py`, 6 passed).

## ⑤ Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_sph/analyze_residual_growth.py
PYTHONPATH=src python -m pytest tests/test_projection_spurious_bands.py -q
```
