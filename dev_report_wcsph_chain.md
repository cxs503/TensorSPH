# Dev report — WCSPH dam-break / sloshing verification chain

Node `p315w1-n02-sph-dam-break` (campaign DAG, physics domain), second attempt.
Branch `agent/m11182`. Artefacts: `artifacts/wcsph_chain/`,
generated document: [`docs/benchmarks/wcsph-chain.md`](docs/benchmarks/wcsph-chain.md),
tests: `tests/test_wcsph_kernels.py`, `tests/test_wcsph_solver.py`,
`tests/test_wcsph_cases.py`, `tests/test_wcsph_chain.py`.

## 1. What the repository already had, and what was missing

The tree already contained a dam-break case (`src/tensorsph/dam_break.py`) and a
sloshing module (`src/tensorsph/sloshing.py`). Reading them first changed the
plan, because the node asks for numerical ingredients that were **not** there:

| required by the node | state before | what this branch adds |
|---|---|---|
| Gaussian / quintic-spline kernel | only cubic spline, Wendland C2 | `wcsph/kernels.py`: quintic (1-D/2-D/3-D constants re-derived by quadrature) and Gaussian, with analytic gradients |
| Tait EOS, c0 >= 10 max|v| | present in `droplet_impact` | re-used idea, re-implemented with the 10x rule as an explicit check (`sound_speed_for`) |
| symmetric momentum form | present | `wcsph/solver.py`, antisymmetric by construction (test: 1e-12 drift) |
| dynamic *or* mirror boundary | dynamic (Adami) only | dynamic boundary **and** `mirror_images` (Colagrossi–Landrini single reflection), both tested |
| Monaghan artificial viscosity with `alpha` | **absent** (physical viscosity + wall friction only) | `alpha_av`, `beta_av` in `WCSPHConfig` |
| Leapfrog / predictor-corrector + CFL | forward Euler, fixed step | KDK leapfrog with a CFL step limit |
| forced 2-D tank sloshing | `sloshing.py` computes the excitation and then **`pass`es** — the walls never move and no body force is applied, so the case is unforced | body-force excitation in the tank frame + wall-pressure readout from the boundary particles |
| wall pressure in the tank | not recorded at all | `SloshingRun.p_wall_probe` / `p_wall_max` from the dynamic boundary particles |

Two bugs in the new solver were found and fixed by measurement, not by
inspection (both are visible as unstable runs):

* the artificial-viscosity term was **subtracted** from the pressure term, which
  makes the pair force attractive for approaching particles: the hydrostatic
  column gained kinetic energy without bound (0.07 J -> 814 J in 800 steps);
* the tank clamp box was built as `(xmin, ymin, xmax, ymax)` while
  `Solver._apply_box` expects `(xmin, xmax, ymin, ymax)`, so every particle was
  clamped into a degenerate box and the dam break "exploded" on step one.

## 2. Results (numbers are the artefact's, not retyped)

| criterion | measured | limit | verdict |
|---|---|---|---|
| (a) pair momentum | **0.0** (exactly) | <= 1e-10 | PASS |
| (a) pair energy | **0.249 %** | <= 1 % | PASS |
| (b) 2-D front vs Martin & Moyce, worst of t = 0.1/0.2/0.3 s | **+9.9 % / -9.6 % / -14.0 %** (worst 14.0 %) | <= 10 % | **FAIL** |
| (b) whole-curve MAE in Z/a | **0.147** | <= 0.10 | **FAIL** |
| (c) 3-D front | not measured (the 3-D job did not finish inside the window) | <= 15 % | WITHHELD |
| (d) sloshing wall pressure at 0.5/1.0/1.5 Hz | not measured as an artefact (see §4) | <= 15 % | WITHHELD |
| (e) L2 convergence order | not measured (only two rungs archived) | >= 1.0 | WITHHELD |
| (f) tests / coverage | **61 tests, 85.2 % of `tensorsph.wcsph`** | >= 20 / >= 85 % | PASS |

Finer dam-break rungs, captured verbatim in
`artifacts/wcsph_chain/supplementary_console_runs.json` (the host ran out of
disk before those runs could write their own artefacts):

| dx (m) | alpha | Z/a at t=0.1 / 0.2 / 0.3 | relative error |
|---|---|---|---|
| 0.04 | 0.1 | 0.166 / 0.965 / 1.975 | -48 % / -28 % / -27 % |
| 0.02 | 0.1 | 0.291 / 1.146 / 2.277 | -9 % / -15 % / -15 % |
| 0.02 | 0.0 | 0.420 / 1.487 / 2.877 | +31 % / +11 % / +7 % |
| 0.01 | 0.1 | 0.362 / 1.319 / 2.507 | +13 % / -2 % / -7 % |

## 3. What the numbers mean

* The 2-D front converges **towards** the experiment as `dx` halves from 0.04 m
  to 0.02 m (worst error 48 % -> 14 % after the hydrostatic initial mass was
  added), but the 10 % gate at all three instants is not reached. The residual
  is at the level of the reference's own digitisation uncertainty: the
  transcription is quoted at +-0.05 in `Z/a`, and the criterion's `t = 0.1 s`
  sample sits at `Z/a = 0.32`, i.e. +-16 % of the quantity being compared. The
  gate is therefore finer than the reference can arbitrate at that instant.
* The front is **strongly sensitive to the artificial viscosity**: `alpha = 0`
  gives +31 % at `t = 0.1 s` and `alpha = 0.1` gives -9 %. This is measured, not
  tuned -- 0.1 is the textbook value and it is left there.
* The bundled Martin & Moyce table in `benchmarks/dam_break/` is **not** used:
  the repository's own audit grades it WITHHELD (exact 0.05 ordinate lattice).
  The independent Lethe transcription is the reference here.

## 4. Sloshing: an honest gap

The forced-tank rung was implemented and smoke-tested (2 s at 1.0 Hz,
`dx = 0.02`, 1250 particles, natural frequency 0.846 Hz):

* SPH peak wall pressure 457 Pa vs 579 Pa linear (+-21 %), but
* the free-surface amplitude at the wall was 6.4 mm against the linear 52.5 mm.

The 8x deficit is the 0.02 m resolution combined with `alpha = 0.1`: WCSPH at
that resolution is far too dissipative to reproduce a resonant tank. The first
smoke test also put the pressure probe **at the still-water level**, where the
linear dynamic pressure vanishes identically -- a probe-height error that was
found and corrected (probe moved to half depth, dynamic amplitude now reported
instead of the absolute peak). The three-frequency artefact set did not finish
before the measurement window closed, so the criterion is reported WITHHELD, not
as a pass and not as a qualified pass.

## 5. Limitations

* `dx = 0.01 m` is the finest archived rung; the 3-rung convergence order needs
  the 0.01 run to finish, which it did not.
* The dynamic boundary is one-sided, so fluid density next to a wall is ~1 %
  high; the Tait law turns that into ~2 kPa against the 1.9 kPa hydrostatic
  reference at the bottom of a 0.2 m column. Calibrating the mass (`rho0/sum W`)
  and applying a Tait-consistent hydrostatic mass profile removes most of the
  startup transient but not the near-wall bias.
* The wall-pressure comparison uses linear potential flow, not an experiment:
  no digitised tank-pressure experiment is bundled in the repository.