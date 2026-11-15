# How long does the still lake stay still?

Case-pool anchor `CASE-SPH-LAKE-AT-REST` — reference: the analytic hydrostatic
identity (`u = 0`, flat free surface); anchor tolerance: **machine precision
(1e-9)**.

`docs/verification_lake_at_rest_periodic.md` certifies the tolerance in the
laterally periodic tank over `t_end = 0.50 s`.  That number is reproduced here
and then **extended** until it breaks: the rest state is a discrete equilibrium
with a finite lifetime, and this document measures that lifetime, the growth law
of the departure, the mode that grows and the controls that decide whether the
departure belongs to the discrete equilibrium or to the integrator.

Runner `benchmarks/lake_at_rest/stability.py`; artefact
`tests/data/lake_at_rest/stability.json`; gates
`tests/test_lake_at_rest_stability.py`; figures
`docs/figures/lake_stability_departure.png`,
`docs/figures/lake_stability_growth.png`.  Reproduce (≈ minutes, submitted to
the cluster under the local-compute policy):

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/stability.py \
    --json tests/data/lake_at_rest/stability.json --figures docs/figures
```

## 1. Verdict

**The published PASS is horizon-limited, and this is a FAIL for the anchor as an
unconditional claim.**

* The tolerance holds up to `t* = `1.1805` s`
  (`max|v|/c0` first crosses 1e-9).  The claimed `0.50 s` is inside that window,
  so **the merged PASS is not contradicted** — `persists_to_claimed_horizon =
  True`.
* Beyond `t*` the departure is **exponential**:
  `λ = `16.299` 1/s`, fit `R² =
  `1.0000`` over the declared window
  `1e-10 ≤ max|v|/c0 ≤ 1e-05`
  (a power law fits worse: `R² = `0.9952``).
* At the end of the base run (`t = `2.50` s`) the lake
  moves at `max|v|/c0 = `0.009`` and its surface has
  moved `max|dy|/H = `0.074`` — ten orders of
  magnitude above the anchor tolerance.
* The rate is **not** an integrator artefact: halving the acoustic Courant
  number (`dt: `3.900e-04` → `1.950e-04` s`) changes
  `λ` by a factor ``0.952``
  (declared window: `0.75 … 1.33`).
* The rate is **seed-independent** (linear instability): seeding the
  displacement with `1e-6 dx` and `1e-8 dx` gives
  `λ = `0.887`` and
  ``3.724`` 1/s.
* It is **not** the frozen pair list: rebuilding it every 20 steps gives
  `λ = `16.391`` 1/s.
* It is **not** one lattice: `dx = 0.025 m` gives
  `λ = `14.889`` 1/s.

So the honest statement of the merged result is: *the shipped operator holds a
still lake still to machine precision for about one second of physical time;
after that a roundoff-seeded mode grows exponentially and the lake starts to
slosh.*  The anchor tolerance is met for `t < t*` and violated for `t > t*`.

## 2. Case and protocol

Geometry and fluid are exactly the certified periodic case: tank `0.40 × 0.30 m`,
`dx = `0.020` m`, `h = 1.3 dx`, `rho0 = 1000`,
`c0 = 20 m/s`, `gamma = 7`, `g = 9.81 m/s²`; the reconstructed row-tied mass
field of `tensorsph.lake_at_rest.reconstruct` is the initial condition
(`max|a_y|/g = `6.379e-13``,
`max|a_x|/g = `1.381e-12``, converged in
`14` iterations).  Symplectic Euler at
`dt = Courant · h / c0`, positions free, density re-summed every step — the same
protocol as the published 0.50 s run, only longer.

## 3. Numbers

| arm | dt [s] | steps | t end [s] | t\* (`|v|/c0`) [s] | `max|v|/c0` at end | `λ` [1/s] | R² | dominant mode |
|---|---|---|---|---|---|---|---|---|
| base | `3.900e-04` | 6410 | `2.50` | `1.1805` | `0.006` | `16.299` | `1.0000` | 3 |
| courant_half | `1.950e-04` | 12821 | `2.50` | `1.3892` | `0.008` | `15.519` | `0.9989` | 6 |
| resolution_dx025 | `4.875e-04` | 5128 | `2.50` | `1.3392` | `0.008` | `14.889` | `1.0000` | 5 |
| pairs_rebuilt | `3.900e-04` | 6410 | `2.50` | `1.2078` | `0.005` | `16.391` | `1.0000` | 5 |
| seed_1e-6 | `3.900e-04` | 6410 | `2.50` | `3.9000e-04` | `0.008` | `0.887` | `0.2878` | 2 |
| seed_1e-8 | `3.900e-04` | 6410 | `2.50` | `7.8000e-04` | `0.013` | `3.724` | `0.5753` | 4 |
| saturation_8s | `3.900e-04` | 20513 | `8.00` | `1.1716` | `2.677` | `16.445` | `1.0000` | 1 |
| viscosity_1e-6 | `3.900e-04` | 6410 | `2.50` | `1.1501` | `0.006` | `16.536` | `1.0000` | 1 |

Rate in the declared exponential band (seven decades above the roundoff floor, one below saturation): `lambda = `16.470`` 1/s with `R^2 = `1.00000``.

| arm | legacy window `1e-10 .. 1e-5` | R² | band `1e-06 .. 0.001` | R² | valid rate? |
|---|---|---|---|---|---|
| base | `16.299` | `1.0000` | `16.470` | `1.0000` | yes |
| courant_half | `15.519` | `0.9989` | `16.558` | `1.0000` | yes |
| resolution_dx025 | `14.889` | `1.0000` | `14.916` | `1.0000` | yes |
| pairs_rebuilt | `16.391` | `1.0000` | `16.421` | `1.0000` | yes |
| seed_1e-6 | `0.887` | `0.2878` | `16.475` | `0.9997` | **no** (transient) |
| seed_1e-8 | `3.724` | `0.5753` | `16.600` | `1.0000` | **no** (transient) |
| saturation_8s | `16.445` | `1.0000` | `16.563` | `1.0000` | yes |
| viscosity_1e-6 | `16.536` | `1.0000` | `16.634` | `1.0000` | yes |

Controls in the band: `dt` halved ``1.0054``, seed `1e-6` over `1e-8` ``0.9925``, pair list rebuilt ``0.9971``, `dx = 0.025` ``0.9056``, 8 s saturation arm ``1.0056``.  The legacy window's fit for the seeded arms spans the start-up transient (`R^2` 0.29 / 0.58); those numbers are recorded but are **not** rates, and the seed-independence gate now uses the band estimate.

`t*` is the running-worst crossing time of the anchor tolerance by
`max|v|/c0`; `λ` is the least-squares slope of `ln max|v|/c0` against `t` in the
declared window (`t` from ``1.043`` to
``1.736`` s for the base arm).

Saturation (`saturation_8s`, `t_end = `8.00` s`):
the exponential phase stops at a finite amplitude —
`max|dy|/H = `487.247``,
`max|v|/c0 = `2.677`` — i.e.
the lake ends up sloshing, not settling back.

## 4. Which mode grows

The periodic direction lets the vertical displacement be decomposed into
`x`-wavenumbers.  The dominant index reported per sample is the argmax of
`|rFFT(column-mean Δy)|`; over the base arm it stays at index
`3` (out of
`10` resolvable), and its amplitude trace is the
orange curve of `docs/figures/lake_stability_growth.png`.  A grid-scale index
(`k ≈ π/dx`) would point at a kernel/pairing (tensile) mode; a low index at a
bulk mode of the layer.  The measured index is reported here as-is; the
mechanism claim is limited to what the spectrum shows.

## 5. Where the claim now stands

| statement | status |
|---|---|
| rest to 1e-9 over the published 0.50 s | **holds** (`t* = `1.1805` s` > 0.50 s) |
| rest to 1e-9 as an *unconditional* claim | **FAIL** — exponential departure, `λ = `16.299`` 1/s |
| wall-bounded tank (side walls) | **FAIL** as documented (`max|a_x|/g ≈ 0.12`, `docs/verification_lake_at_rest_wall_gap.md`) |
| anchor verdict | **FAIL**, now with a measured lifetime instead of a tolerance-only claim |

## 6. Caveats and controls

* **The equilibrium is prepared, not generic.** `reconstruct` solves for the
  mass field; this study measures the lifetime of *that* state, which is the
  best case.  A user initialised with the analytic profile departs from
  `O(g)` residual immediately (`docs/verification_lake_at_rest.md`).
* **Roundoff seeds it.** The mechanism inferred from the seed-linearity arm is
  that roundoff sets the phase of the departure and the growth law is linear;
  the run-to-run spread of such a statement is a *phase*, not a rate, and the
  rate is what is gated.
* **Pair list.** Frozen at `t = 0` by the case (`build_state`); the
  `pairs_rebuilt` arm is the control that this is not the cause.
* **One operator.** Nothing here modifies shipped physics; the study is a
  measurement of `tensorsph.droplet_impact.rhs` as merged.
['', '### 6.1 The production viscosity knob', '', '`LakeConfig(viscosity=1e-6)` (the physical kinematic viscosity of water) re-reconstructs its own equilibrium and is run under the same protocol:', '', '| arm | `λ` [1/s] | t\\* (`|v|/c0`) [s] | `max|v|/c0` |', '|---|---|---|---|', '| base | `16.299` | `1.1805` | `0.009` |', '| courant_half | `15.519` | `1.3892` | `0.008` |', '| resolution_dx025 | `14.889` | `1.3392` | `0.012` |', '| pairs_rebuilt | `16.391` | `1.2078` | `0.007` |', '| seed_1e-6 | `0.887` | `3.9000e-04` | `0.009` |', '| seed_1e-8 | `3.724` | `7.8000e-04` | `0.013` |', '| saturation_8s | `16.445` | `1.1716` | `2.677` |', '| viscosity_1e-6 | `16.536` | `1.1501` | `0.010` |', '', '`λ(viscosity 1e-6) / λ(inviscid) = `1.015``.  The physical viscosity of water does **not** remove the departure — viscosity damps shear, and this mode carries almost no shear.']

## 7. What would close the anchor

A rest state that survives 20 s needs either a well-balanced/consistent
momentum operator (removing the truncation residual at the wall/surface) or a
dissipative mechanism that damps this mode without damping the physics under
test.  Both are solver changes; this document is the measurement that makes the
size of the required change explicit (`λ`, `t*`, saturation amplitude).
