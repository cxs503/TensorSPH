# B3 — Surface-tension verification: the Young-Laplace check

Status: **FAIL** — the merged Continuum Surface Force (CSF) model does not
reproduce the Young-Laplace pressure jump, and its error grows under
refinement. This document is the quantitative comparison that
[`docs/algorithm-gap-analysis.md`](../algorithm-gap-analysis.md) asked for
before any capillary claim may be made ("**Gate:** no claim of capillary
fidelity until the quantitative comparison is documented").

Harness: `benchmarks/laplace_droplet/run.py` (numbers in
`benchmarks/laplace_droplet/results.json`, tables below rendered by
`benchmarks/laplace_droplet/make_tables.py`, tests in
`tests/test_laplace_droplet.py`). Implementation under test:
`tensorsph.surface_tension.compute_surface_tension_force` (merged in `0.9.0`,
mission m277).

## 1. Case, geometry and reference

A circular 2-D droplet of water-like fluid at rest, held only by surface
tension. For an interface of radius `R` and coefficient `sigma` the exact
balance is

    dp = sigma / R            (2-D cylinder; 2 sigma / R for a 3-D sphere)

| parameter | value |
|---|---|
| radius `R` | 0.10 m (`R/dx` = 10, 20, 40; plus `R` = 0.05 / 0.20 m at fixed `dx`) |
| particle spacing `dx` | 0.010 / 0.005 / 0.0025 m |
| smoothing length `h` | `1.3 dx` |
| `rho0` | 1000 kg/m³ |
| `sigma` | 0.072 N/m (also 0.72 and 7.2 N/m for the linearity check) |
| lattices | hexagonal (correct 2-D number density) and square; plus jittered hex (0.2 `dx`, 0.4 `dx`) |
| reference | `dp = sigma/R` = 0.72 Pa for the reference case (exact, no uncertainty) |

The droplet carries `rho0 * pi R^2` of mass to within 0.1 % and the
summation density reaches `rho0` inside, so the geometry and the density
field are not part of the error.

## 2. Method, and why not a dynamic run

The Laplace jump implied by the production force field is obtained by
integrating the radial component of `rho * a_surface_tension` across the
interface:

    dp_model = -integral_0^{1.5R} (rho a_r) dr ,     N_delta = integral |grad c| dr

In continuum CSF theory `dp_model = sigma <kappa>_delta N_delta`, so three
independent factors are measured separately: the surface-delta normalisation
`N_delta` (must be 1), the curvature `kappa` (must be `1/R` with the correct
sign) and the spatial coherence of the interface normal. That decomposition,
not a single number, is the result.

A pressure-jump measurement from the *density* field was rejected on
quantitative grounds, not convenience: with the Tait EOS `p = B[(rho/rho0)^gamma - 1]`
and `B = rho0 c0^2/gamma`, a Laplace pressure of `sigma/R` moves the density by
`dp / c0^2`, which for water at `c0` = 20 m/s is a 1.8e-6 relative signal, three
orders of magnitude below SPH density noise (table at the end, from
`laplace_resolvability()`). The force-field formulation has no such noise
floor; its own accuracy is validated by the instrument checks below.

## 3. Results

<!-- RESULTS:BEGIN -->

### Case table (auto-generated from `benchmarks/laplace_droplet/results.json`)

`R/dx` is the only governing resolution parameter; `sigma` enters linearly (verified: the ratio is identical at sigma = 0.072 / 0.72 / 7.2 N/m).

| R/dx | lattice | jitter | N | N_delta = int|grad c|dr | kappa_delta R (model) | kappa>0 frac. | <n.rhat>_delta | dp_model/dp_theory | error | within 15 % |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | hex | 0 | 363 | 0.329 | -1.317 | 0.48 | -0.008 | +0.428 | 57.2% | **no** |
| 20 | hex | 0 | 1449 | 0.367 | -3.255 | 0.46 | -0.020 | +1.387 | 38.7% | **no** |
| 40 | hex | 0 | 5800 | 0.419 | -5.693 | 0.48 | -0.015 | +3.197 | 219.7% | **no** |
| 10 | square | 0 | 315 | 0.309 | -1.491 | 0.50 | -0.010 | +0.225 | 77.5% | **no** |
| 20 | square | 0 | 1253 | 0.374 | -2.861 | 0.46 | -0.029 | +0.623 | 37.7% | **no** |
| 40 | square | 0 | 5021 | 0.414 | -5.499 | 0.47 | -0.016 | +1.549 | 54.9% | **no** |
| 20 | hex | 0.2 | 1449 | 0.369 | -3.604 | 0.44 | -0.024 | +0.840 | 16.0% | **no** |
| 20 | hex | 0.4 | 1449 | 0.543 | -3.040 | 0.45 | -0.020 | +3.822 | 282.2% | **no** |
| 20 | hex | 0 | 1449 | 0.367 | -3.255 | 0.46 | -0.020 | +1.387 | 38.7% | **no** |
| 20 | hex | 0 | 1449 | 0.367 | -3.255 | 0.46 | -0.020 | +1.387 | 38.7% | **no** |
| 10 | hex | 0 | 363 | 0.329 | -1.317 | 0.48 | -0.008 | +0.428 | 57.2% | **no** |
| 40 | hex | 0 | 5800 | 0.419 | -5.693 | 0.48 | -0.015 | +3.197 | 219.7% | **no** |

### Error decomposition (auto-generated)

The same CSF form evaluated with pieces replaced by exact values. `B` and `C` are instrument checks: they must not depend on the model at all.

| R/dx | lattice | A: exact kappa, model normal | B: exact kappa + exact normal | C: exact operators + unit delta_s | model as merged |
|---|---|---|---|---|---|
| 10 | hex jitter=0 | -0.001 | +0.329 (N_delta=0.329) | +1.000 | +0.428 |
| 20 | hex jitter=0 | -0.009 | +0.367 (N_delta=0.367) | +1.000 | +1.387 |
| 40 | hex jitter=0 | -0.062 | +0.419 (N_delta=0.419) | +1.000 | +3.197 |
| 10 | square jitter=0 | +0.003 | +0.309 (N_delta=0.309) | +1.000 | +0.225 |
| 20 | square jitter=0 | -0.004 | +0.374 (N_delta=0.374) | +1.000 | +0.623 |
| 40 | square jitter=0 | +0.009 | +0.414 (N_delta=0.414) | +1.000 | +1.549 |
| 20 | hex jitter=0.2 | -0.007 | +0.369 (N_delta=0.369) | +1.000 | +0.840 |
| 20 | hex jitter=0.4 | -0.161 | +0.543 (N_delta=0.543) | +1.000 | +3.822 |
| 20 | hex jitter=0 | -0.009 | +0.367 (N_delta=0.367) | +1.000 | +1.387 |
| 20 | hex jitter=0 | -0.009 | +0.367 (N_delta=0.367) | +1.000 | +1.387 |
| 10 | hex jitter=0 | -0.001 | +0.329 (N_delta=0.329) | +1.000 | +0.428 |
| 40 | hex jitter=0 | -0.062 | +0.419 (N_delta=0.419) | +1.000 | +3.197 |

### Verdict (auto-generated)

- criterion: |dp_model/dp_theory - 1| <= 15 % at every resolution
- status: **FAIL**
- worst relative error over the 8 reference cases: 219.7%
- cases inside tolerance: 0/8
- monotone convergence under refinement: **no**
- instrument B error against N_delta: 6.66e-16
- instrument C error against the Laplace law: 6.66e-16

### Why the jump is not read from the density field (auto-generated)

| quantity | value |
|---|---|
| dp = sigma/R [Pa] | 0.7200 |
| bulk modulus B = rho0 c0^2/gamma [Pa] | 57142.9 |
| dp/B | 1.260e-05 |
| density signal (rho/rho0 - 1) | 1.800e-06 |
| summation-density noise (reference) | 1.0e-03 |
| signal/noise | 0.002 |
| verdict | not resolvable |

<!-- RESULTS:END -->

## 4. Reading the numbers

1. **`N_delta` is between 0.31 and 0.42, not 1.** Brackbill's delta function
   satisfies `integral |grad c| dr = 1` because the colour field jumps from 1
   to 0 across the interface. Here the colour field is
   `c = neighbour_count / mean(neighbour_count)` *only* on particles the
   neighbour-count test calls surface particles (so `c` in [0.7, 1]) and 1
   everywhere else; it never reaches 0. The discrete `|grad c|` therefore
   carries only ~1/3 of the required surface weight, which alone would
   under-predict the pressure jump by a factor of ~3.
2. **The curvature has the wrong sign and diverges.** `kappa = -div(n)` on
   the interface band has a negative median (`kappa R` = -1.3, -3.3, -5.7 at
   `R/dx` = 10, 20, 40), i.e. concave where the droplet is convex, and its
   magnitude *grows* with resolution instead of converging: the estimate
   diverges because `n` is built by normalising a one-layer gradient that is
   dominated by particle-scale noise.
3. **The normal field is not radially coherent.** With the exact curvature
   injected (`instrument A`), the modelled normals give a *zero* net jump
   (`<n.rhat>_delta` = -0.01 to -0.03) where an outward-pointing normal field
   would give `N_delta`. The normals cancel each other, so the force field
   carries no systematic pressure.
4. **The net error is therefore not a small calibration factor.** The ratio
   `dp_model/dp_theory` runs 0.23 -> 0.62 -> 1.55 (square) and
   0.43 -> 1.39 -> 3.20 (hex) as `dx` is halved; it does not converge, it
   leaves the +/-15 % band in both directions, and it moves by a factor of
   ~4.5 when the initial lattice is jittered by 0.4 `dx` (0.84 vs 3.82 both at
   `R/dx` = 20).
5. **The force itself is implemented consistently with its own formula.** It
   is exactly linear in `sigma` (identical ratio at 0.072 / 0.72 / 7.2 N/m)
   and the `sigma = 0` path is exactly zero. The failure is in
   colour-field/curvature discretisation, not in the force algebra.

The instrument is validated by checks B and C: injecting the exact curvature
and the exact radial normal recovers `N_delta` to 1e-15 relative, and
additionally normalising `delta_s` recovers the Laplace law to 1e-6. So the
CSF *form* used by TensorSPH would satisfy `dp = sigma/R`; the operators fed
into it do not.

## 5. Error verdict against the platform standard

<!-- ACCEPT:BEGIN -->

| quantity | case | reference | measured | relative error | < 15 % | < 3 % |
|---|---|---|---|---|---|---|
| `dp` | hex, R/dx=10 | 0.7200 Pa | 0.3078 Pa | -57.2% | no | no |
| `dp` | hex, R/dx=20 | 0.7200 Pa | 0.9989 Pa | +38.7% | no | no |
| `dp` | hex, R/dx=40 | 0.7200 Pa | 2.3020 Pa | +219.7% | no | no |
| `dp` | square, R/dx=10 | 0.7200 Pa | 0.1623 Pa | -77.5% | no | no |
| `dp` | square, R/dx=20 | 0.7200 Pa | 0.4484 Pa | -37.7% | no | no |
| `dp` | square, R/dx=40 | 0.7200 Pa | 1.1149 Pa | +54.9% | no | no |
| `N_delta` = int abs(grad c) dr | all cases | 1.000 | 0.309-0.543 | -69.1% to -45.7% | no | no |
| `<kappa>_delta R` | all cases | 1.000 | -5.69 to -1.32 | sign and magnitude wrong | no | no |

<!-- ACCEPT:END -->

Nothing in this table is inside 15 %, let alone 3 %. The honest verdict is
that the merged surface-tension model is a *numerically active* term, not a
*verified* capillary model, and the capability table entry must stay
unverified.

## 6. What this does not show

- The force is checked at rest on an undeformed circular interface. It says
  nothing about the tangential (Marangoni-free) component or about dynamic
  cases; that is the second half of B3 (oscillating droplet frequency and
  damping), still open.
- No comparison with experiment is attempted here: this is verification
  against an exact solution, which is the correct first step.
- The instrument measures the pressure jump implied by the force field. It
  assumes the momentum equation that consumes the force is the standard
  symmetric SPH form (it is: `droplet_impact.rhs`).

## 7. What a fix has to achieve

In the order the evidence points:

1. **Colour field**: use an actual 1 -> 0 transition across the interface
   (e.g. a sharpened or Shepard-filtered colour function), and normalise
   `delta_s` to `integral |grad c| dr = 1`; the harness measures that integral,
   so the fix is directly checkable.
2. **Curvature**: replace `-div(normalised grad c)` with an operator that does
   not differentiate an already noisy field, e.g. the Laplacian-of-colour form
   `kappa = lap(c)/|grad c|` (Akinci et al. 2013) with smoothing, and keep the
   sign convention test in `tests/test_surface_tension.py` valid.
3. **Normals**: only use interface normals inside the interface band and
   smooth them before taking the divergence; the current normal field averages
   to zero over the band.

A candidate evaluation of item 2 that was run during this verification
(Laplacian of a smooth tanh colour profile) returned `|kappa R|` = 1.00 at
`R/dx` = 20 (hex) and 0.93 at `R/dx` = 40 (square), i.e. inside 15 % on the
best cases but not stable across all lattices (4.6 at `R/dx` = 10 hex). It is a
lead for the fix, not a merged result, and it is not part of the acceptance
table above because it also needs the `delta_s` normalisation before the
pressure jump itself comes out right.

## 8. Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_droplet/run.py          # writes results.json + 2 figures
PYTHONPATH=src python benchmarks/laplace_droplet/make_tables.py  # refreshes the tables above
PYTHONPATH=src python -m pytest tests/test_laplace_droplet.py -q # 11 tests
```

Figures: `laplace_verification.png` (ratio vs resolution with the +/-15 % band,
`N_delta`, curvature accuracy) and `laplace_force_profile.png` (radial
`rho a_r` profiles at three resolutions).