# CASE-SPH-POISEUILLE-MORRIS — predeclared design

Mission 20084. Written **before** the production sweep was launched (the sweep
`cj2505` was submitted after this file was created; no threshold below was
changed after seeing results).

## 1. Decision

Does TensorSPH reproduce the steady plane-Poiseuille velocity profile of the
Morris low-Reynolds channel case within the case tolerance **< 5 %**, and which
of the two SPH routes in this repository does so at the resolutions the
repository can afford?

## 2. Alternatives

| id | route | code | prior evidence |
|---|---|---|---|
| A (headline) | incompressible SPH, pressure projection | `tensorsph/isph_channel.py::run_case` | `GAP-SPH-ISPH-PROJECTION`; start-up series max L2 5.7 % @ H/dx=8 → 0.63 % @ H/dx=20 |
| B (baseline / negative control) | production weakly-compressible SPH, Morris viscous Laplacian, Tait EOS | `tensorsph/droplet_impact.py::rhs` driven by `benchmarks/poiseuille/run.py` | recorded ν_eff/ν ≈ 1.22 over-diffusion, centre velocity ≈ 0.81·u_max (i.e. −19 %) |

Route B is the incumbent; it is not a straw man but the path every other SPH
benchmark in this repository uses. If A cannot beat B on the *same* metric, the
decision is "no improvement".

## 3. Hypothesis

* **H1** route A converges to the exact profile with an interior L2 error below
  5 % at `H/dx ≥ 12` and keeps decreasing under refinement (observed order
  > 1).
* **H0** the residual error is a boundary artefact that does not shrink under
  refinement, so no affordable resolution reaches 5 %.

H0 is falsified by the resolution ladder, not by a single run.

## 4. Outcome metrics (unit of analysis = one steady-state profile)

All errors are normalised by the analytical peak `u_max = g H² / (8ν)`, which is
finite at the walls; a per-row relative error would divide by zero there.

| symbol | definition | band |
|---|---|---|
| `e_c` | (u(0.5H) − u_ref(0.5H)) / u_ref(0.5H) | single centre row |
| `e_L2` | √(mean[(u−u_ref)²]/u_max²) | interior rows 0.15H ≤ y ≤ 0.85H |
| `e_max` | max |u−u_ref| / u_max | interior rows |
| `e_τ` | (τ_w,measured − ρgH/2)/(ρgH/2), momentum budget | wall |
| `p` | least-squares slope of log e_L2 vs log dx over the ladder | ladder |

## 5. Reference and its access status

The case authority is Morris, Fox & Zhu (1997) *J. Comput. Phys.* 136(1):214–226
(plane-Poiseuille validation section). That article is **closed access**
(Unpaywall `oa_status = closed`, `has_repository_copy = false`, checked
2026-11-14), so the paper's printed profile/table is not readable here.

Per the zero-hand-copy rule, **no number is transcribed from the paper or from
memory**. The reference values are the exact steady solution of the
Navier–Stokes equations for this geometry — which is precisely the curve the
paper validates against — evaluated programmatically inside `run.py`
(`reference_profile`, `case_from_reynolds`). Paper-specific printed values are
graded **WITHHELD**, the analytic reference is graded **REAL**.

## 6. Baseline / control

1. Route B on the identical geometry, forcing and reference metric (route
   comparison control).
2. Physical negative control: the same channel with the forcing removed
   reproduces `u = 0` exactly (the metric cannot manufacture agreement from
   nothing). Realised as the `Re → 0` limit of the ladder plus the already
   recorded free-slip control of `benchmarks/poiseuille` (no wall shear ⇒ no
   steady parabola).

## 7. Predeclared thresholds and stopping rule

* **PASS** if at the finest predeclared resolution `e_L2 < 5 %` **and**
  `|e_c| < 5 %` for route A.
* **STRONG PASS** (owner platform standard) if additionally `< 3 %`.
* **FAIL** otherwise; a FAIL is reported as honest-FAIL with the stage at which
  it broke.
* Stopping rule: the resolution ladder is fixed at `H/dx ∈ {8, 12, 16, 24}`.
  No resolution may be added to the ladder to rescue the verdict, and the
  thresholds are not re-tuned after the results are read.

## 8. Variables, confounders, controls

| kind | quantity | control |
|---|---|---|
| changed (ladder 1) | `H/dx` | 8, 12, 16, 24 |
| changed (ladder 2) | `Re_c = u_max H/ν` | 0.1, 1.0, 12.26 at `H/dx = 16` |
| changed (ladder 3) | support `h/dx` | 1.5, 1.90, 2.3 at `H/dx = 16`, `Re_c = 1` |
| changed (ladder 4) | periodic length `L/H` | 0.6, 1.2 at `H/dx = 16`, `Re_c = 1` |
| held | `H = 0.1 m`, `ρ0 = 1000 kg/m³`, 5 wall layers, `CFL 0.25`, viscous safety 0.125, run length = 10 diffusive times `H²/(νπ²)`, averaging over the last 10 % | fixed in `make_config` |
| confounder 1 | run length: too short looks "steady" | `t_end` is fixed in units of the **diffusive** time, so the transient is resolved at every `Re` |
| confounder 2 | averaging window bias | the run averages the last 10 % of `t_end`; the momentum-budget gate re-fits the other windows (`wall_shear_budget.rel_error_pct_band`) |
| confounder 3 | periodic images truncating the support | `L` is checked with ladder 4 (`L/H` 0.6 vs 1.2 ≥ 2·2h) |
| confounder 4 | resolution and support radius moving together | latticed separately: resolution ladder at fixed `h/dx`, support ladder at fixed `H/dx` |

`Re_c` is built on the *analytical* peak velocity, so it is fixed before the
run and cannot be read off a result and then declared the target.

## 9. Repetitions

The solver is deterministic (fixed lattice, deterministic sparse solve, no
random seeds, no wall-clock-dependent decisions), so a repeated run is a
bit-for-bit repeat, not a statistical sample: **1 run per configuration**,
plus one independent re-run of the headline configuration to prove
reproducibility (reported in the results JSON as the duplicate case). Error
bars therefore come from the *resolution ladder* (a systematic discretisation
study), not from run-to-run scatter — which is the right uncertainty statement
for this class of verification case.

## 10. Environment and analysis plan

* CPU-only, `torch.set_num_threads(1)` (documented 13× speed-up for these
  tensor sizes), `numpy`/`scipy` sparse CG for the pressure Poisson solve.
* Raw artefacts: `runs/poiseuille_morris/results.json` (all profiles, per-row
  reference values, gates, timings), figures `runs/poiseuille_morris/*.png`.
* Aggregation: per-row errors → band RMS/max (`profile_metrics`); convergence
  order from least squares over the whole ladder (`convergence_order`).
* No outliers are removed: every lattice row inside the band is used, and the
  per-row residual is plotted rather than summarised away.
* Comparison method between routes: identical geometry, forcing, run length and
  metric, differing only in the solver.

## 11. Safety / resource limits

Longest single case ≈ `H/dx = 24`; the whole sweep ran as one scheduler job
(`run_compute`, cores=4, timeout 2400 s), not on the 99 control plane.
Reproducible in ≈ 5 min wall clock.

## 12. Interpretation limits

* The reference is an *exact solution*, so a small error proves the discrete
  operators, wall treatment and time integration are consistent — it does not
  validate against measurement, and it says nothing about free-surface,
  turbulence or compressibility errors.
* The claim is scoped to `Re_c ≤ 12.26` (laminar, steady, x-invariant). Nothing
  here supports higher Reynolds numbers, unsteady transitions or 3-D flows.
* The paper's own printed numbers are not compared (closed access); agreement
  with the analytic curve is a *necessary* condition for agreement with the
  paper, not a sufficient one.