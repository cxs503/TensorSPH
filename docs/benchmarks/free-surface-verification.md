# Free-surface verification face (campaign P482-N04)

Four machine-checkable measurements of the free-surface SPH solver, each with a
gate and a negative control, plus the evidence store that makes every reported
number traceable to a SQL query.

| stage | benchmark | gate | negative control |
|---|---|---|---|
| M1 dam break | `benchmarks/verified/sph_dam_break_2d` | front `Z/a(tau)` and column height within 10 % | `g = 0`: the column must not collapse |
| M2 hydrostatic | `benchmarks/verified/sph_hydrostatic` | interior-band error vs `rho g (H-z)` <= 2 %, `N >= 1000` | `rho = 0`: pressure must be zero |
| M3 mass | `benchmarks/verified/sph_dam_break_2d --controls` | accounted mass loss <= 0.1 % over >= 1000 steps | forgetting the outflow must lose > 1 % |
| M4 convergence | `benchmarks/verified/sph_convergence` | interpolation order >= 1 | kernel support `R < 2h` must break the sum |

Code: `src/tensorsph/verification/` (`evidence.py` is the SQLite store,
`dam_break_2d.py` / `hydrostatic.py` / `conservation.py` / `convergence.py` the
four measurements, `report.py` the figures and the report tables).
Tests: `tests/test_verification_free_surface.py` (fast versions of every gate
*and* of every control, plus the provenance test).

Report: `docs/reports/sph-verification-P482-N04.md`, generated from the three
`evidence.sqlite` stores by `docs/reports/make_sph_verification_report.py` --
no number in it is typed by hand.

## What the face says about the solver

* J2, J3, J4 pass: the hydrostatic profile is recovered to ~1 %, the mass
  bookkeeping is exact, the discretisation converges at order ~2 and the
  truncated-support control destroys the hydrostatic balance as it should.
* **J1 fails honestly.** The front runs behind Martin & Moyce for
  `tau < 2` and ahead of it for `tau > 3`. The bundled experimental table is
  graded WITHHELD by `benchmarks/dam_break/reference_audit.json`, and there is no
  digitised `h(t)` curve anywhere in reach, so the height gate is anchored on two
  exact analytic statements instead of on data. Both facts are carried into the
  report rather than smoothed over.
* The A/B diagnosis (`diagnose_surface_treatment.py`) shows the free-surface
  density treatment is the dominant term: the shipped pressure clamp delays the
  collapse (front ~50 % low), while `density_renormalization` restores the
  driving pressure and removes most of the early deficit but leaves the late
  front too fast because no dissipation model (bottom friction, turbulence) is
  present.

## Reproducing

```bash
python benchmarks/verified/sph_dam_break_2d/run.py --spacings 0.02
python benchmarks/verified/sph_dam_break_2d/run.py --spacings 0.01
python benchmarks/verified/sph_dam_break_2d/run.py --spacings 0.02 --controls
python benchmarks/verified/sph_dam_break_2d/run.py --figures
python benchmarks/verified/sph_hydrostatic/run.py --figure
python benchmarks/verified/sph_convergence/run.py --figure
python docs/reports/make_sph_verification_report.py
pytest tests/test_verification_free_surface.py -q
```

The runs are sized for one CPU core (~10 min total at `dx >= 0.01`); the
`evidence.sqlite` files are written next to each runner and are the only source
the report reads.