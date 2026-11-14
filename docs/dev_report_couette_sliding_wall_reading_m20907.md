# Couette sliding-wall shear reading: the 17 % record is an estimator artefact (m20907)

**Anchor**: `XVAL-SPH-METHOD-SELECTION-2026-11-12` — TensorSPH method-selection evidence
**Type**: verification debt (re-check a banked, merged record against the analytic solution)
**Artifact**: `benchmarks/couette/out_readings/wall_shear_reading_audit.json`
**Driver**: `benchmarks/couette/wall_shear_reading_audit.py`
**Test**: `tests/test_couette_wall_shear_reading_audit.py`

## 1. Case

Plane Couette channel, both walls no-slip, top wall sliding at `U`, periodic in `x`,
2-D, `H = 0.1 m`, `nu = 0.02 m^2/s`, `U = 0.2 m/s`, `c0 = 20 m/s`, `rho0 = 1000 kg/m^3`,
`h = 1.3 dx`, 3 wall layers, `wall_friction = 1`, 12 fluid columns, run from rest to
`t_end = 1.2 H^2/nu`, averaged over the last 20 %. Exact solution
`u = U y/H`, `tau = rho nu U/H = 40 Pa` (Batchelor 1967 §4.2).

The method-selection registry banks, for `wcsph_wall_no_slip`, the **sliding**-wall
reading `|1 - tau_SPH/tau_exact|` = 14.20 % (`H/dx=8`) and 17.08 % (`H/dx=16`), while
the **fixed**-wall reading on the same runs is 9.55 % and 1.81 %. The 17.08 % record is
outside the owner 15 % standard and is the worst banked number of the no-slip scheme, so
the selector carries it as a caveat. A reading that *grows* under refinement cannot be a
discretisation error: it is either a wall-model defect or an estimator property.

## 2. What was done

1. Re-ran the exact code path that produced the records (`run.py:run_case`, no
   re-implementation of the time loop) at `H/dx = 8` and `16`, `t_end = 1.2 H^2/nu`.
2. Re-read **the same profile** with four estimators, all normalised by `U/H`:
   the banked one-spacing secant, a 4-row and an 8-row least-squares slope at the
   sliding wall, and the interior-band (0.15 ≤ y/H ≤ 0.85) slope.
3. Mirror control at `H/dx = 8`: identical lattice flipped in `y` (`y -> H-y`, every
   particle keeps its own velocity), so the sliding band sits at `y = 0`. This
   separates "property of the sliding wall" from "property of the top of the domain".
4. Cross-checked against the committed sweep
   (`benchmarks/couette/out_design/viscous_time_convergence.json`): the re-run
   reproduces its `viscous_times = 1.2` readings exactly.

## 3. Results (programmatic from the artifact)

Readings are `|1 - slope/slope_exact|` in %, sliding wall unless stated; `l2` is the
interior L2 profile error, `Q` the mass-flux ratio.

| H/dx | sliding wall | secant1 (banked) | fit4 | fit8 | interior | fixed wall secant1 | l2 % | Q |
|---|---|---|---|---|---|---|---|---|
| 8 | y=H | 27.52 | 2.83 | 8.50 | 6.63 | 4.34 | 2.99 | 1.0391 |
| 8 | y=0 | 27.52 | 2.83 | 8.50 | 6.63 | 4.34 | 41.36 | 1.0391 |
| 16 | y=H | 20.87 | 8.60 | 13.30 | 4.11 | 0.48 | 2.93 | 1.0442 |

Reproduction of the committed sweep at `t_end = 1.2 H^2/nu`: sliding secant 27.52 %
(`H/dx=8`) and 20.87 % (`H/dx=16`), fixed wall 4.34 % / 0.48 % — identical to
`viscous_time_convergence.json` (27.5238 / 20.8671 / 4.3384 / 0.4781). The registry's
14.20 % / 17.08 % are the same estimator at the longest *admissible* run
(`viscous_times = 3.6`), so the audit reads the same quantity, one time level earlier.

Three facts follow from the table:

* **Estimator ordering.** At both rungs `secant1 > fit8 > fit4`, and the interior slope
  is the smallest: 27.52 -> 8.50 -> 2.83 -> 6.63 % (`H/dx=8`) and
  20.87 -> 13.30 -> 8.60 -> 4.11 % (`H/dx=16`). The one-spacing secant is dominated by
  the lag of the outermost row; the lag decays over roughly four spacings. The banked
  17.08 % therefore measures the last-spacing kernel truncation, not the stress the
  wall transmits.
* **Attribution (mirror control).** With the sliding wall moved from `y = H` to `y = 0`
  at identical lattice the reading is unchanged to the last digit (27.52 % / 2.83 % /
  8.50 % / 6.63 %), so the deficit belongs to the sliding wall, not to the top of the
  domain. The mirror arm is provably a real mirror: its interior L2 against
  `u = U y/H` rises to 41.36 % (the reference no longer matches), while every
  slope-based reading, taken in absolute value, stays the same.
* **Fixed vs sliding.** The fixed wall reads 4.34 % and 0.48 % with the *same* secant
  estimator, so the estimator bias is one-sided: the last row next to the sliding wall
  lags more than the first row next to the fixed wall. That asymmetry is the physical
  residue worth chasing (see §5), and it is not the 17 % the registry reports.

## 4. Error verdict against the owner 15 % standard

| quantity (sliding wall) | H/dx=8 | H/dx=16 | < 15 %? |
|---|---|---|---|
| banked one-spacing secant | 27.52 % | 20.87 % (17.08 % at the admissible run) | **no** |
| 4-row least squares | 2.83 % | 8.60 % | **yes** |
| interior band | 6.63 % | 4.11 % | **yes** |

So the answer is estimator-dependent, and the honest statement is: **the defensible
reading of the transmitted wall stress is inside the 15 % standard at both rungs
(2.8-8.6 %), while the banked number is outside it — because it is a one-spacing
secant, not a stress measurement.** The 4-row reading is nevertheless twice as large at
the finer rung (2.83 -> 8.60 %), i.e. it does not converge monotonically under
refinement at fixed `t_end/(H^2/nu)`; that residual trend is real and is the open item.

## 5. Caveats / what is NOT closed

* **The force-based reading was not obtained.** The textbook definition of wall shear is
  the tangential force on the wall, available from `rhs(..., split=True)` wall loads. It
  cannot be split between the three periodic copies used for kernel support (the
  summation density, hence the pressure, is recomputed per call); the attempt
  (`benchmarks/couette/diag_wall_force.py`) left a 12 Pa Newton's-third-law residual on a
  40 Pa stress and was discarded. It must be measured on a non-periodic variant or with
  per-copy bookkeeping inside `rhs` — that is the next step, not done here.
* One time level (`t_end = 1.2 H^2/nu`) and one mirror rung (`H/dx = 8`). The registry's
  `H/dx=16` number is at `viscous_times = 3.6`; the estimator conclusion is drawn at
  `1.2` and is not re-derived at `3.6`.
* No new solver change is proposed on this evidence: the near-wall lag is a kernel/wall
  model property, and re-badging it would need the force reading above.
* The registry was **not** edited: its numbers are reproduced faithfully and the
  selector's caveat is factually correct ("one-spacing secant"). What this audit
  establishes is that the caveat must be read as an estimator caveat, and it supplies
  the quantified alternative (4-row / interior) for the follow-up record.

## 6. Reproduce

```bash
PYTHONPATH=src python benchmarks/couette/wall_shear_reading_audit.py --res 8        # ~45 s
PYTHONPATH=src python benchmarks/couette/wall_shear_reading_audit.py --res 16 --only normal  # ~62 s
PYTHONPATH=src python -m pytest tests/test_couette_wall_shear_reading_audit.py -q
```

## 7. Next selfstart lead

New ledger entry proposal: `XVAL-COUETTE-WALL-FORCE-READING` (domain TensorSPH,
acceptance: tangential wall force from `rhs(split=True)` closes Newton's third law to
<1 % of `rho nu U/H` and reads both walls within 5 % of the analytic `tau` at
`H/dx = 8, 16`). It closes §5 and would let the selector bank a stress-based record
instead of a secant.

## Appendix: files

* `benchmarks/couette/wall_shear_reading_audit.py` (driver, uses `run.py:run_case`)
* `benchmarks/couette/out_readings/wall_shear_reading_audit.json` (artifact)
* `tests/test_couette_wall_shear_reading_audit.py` (invariants)
* `benchmarks/couette/diag_wall_asym.py`, `benchmarks/couette/diag_wall_mirror.py`,
  `benchmarks/couette/diag_wall_force.py` (diagnostics kept for reproduction; the force
  one is the documented dead end)
