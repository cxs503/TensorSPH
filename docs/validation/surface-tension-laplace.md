# Surface tension: Laplace-law verification

**Scope.** Quantitative verification of the CSF surface-tension chain in
`src/tensorsph/surface_tension.py` against the analytic Laplace law for a static
circular interface. This document is rendered from
[`benchmarks/surface_tension_laplace/validation_report.json`](../../benchmarks/surface_tension_laplace/validation_report.json);
the numbers below are injected programmatically from that JSON, not typed by hand.

## 1. Physics and geometry

A static circular interface of radius `R` separating two immiscible media
supports the Laplace pressure jump

```
dp = sigma * kappa = sigma / R            (2-D, circular interface)
```

In the CSF (continuum surface force) model of Brackbill, Kothe & Zemach (1992) the
interface is represented by a colour function `c` and the force is

```
f = -sigma * kappa * n * delta_s,   kappa = div(n),  n = grad(c)/|grad(c)|,  delta_s = |grad(c)|
```

For a *static* interface the body force balances the pressure gradient, so the
pressure jump is the radial line integral of the volume force:

```
dp = int_0^inf  sigma * kappa(r) * |grad(c)(r)| dr = sigma * kappa * Delta c
```

with `Delta c = int |grad(c)| dr` the colour jump the discretisation actually
resolves. This is the quantity measured below: it needs no time integration, so
it is a pure operator benchmark and can be run on any machine.

| parameter | value |
|---|---|
| interface radius R | 0.1 m |
| surface tension sigma | 0.072 N/m |
| reference density rho0 | 1000.0 kg/m³ |
| smoothing h | 1.3 · dx |
| kernel | 2-D cubic spline, support 2h |
| analytic jump sigma/R | **0.7200 Pa** |

Two particle clouds are used:

* **two-phase** — colour 1 inside, 0 outside, equal density on both sides. The
  jump is fully resolved (`Delta c -> 1`), so `dp -> sigma/R` is a real benchmark.
* **one-sided free surface** — the shipped droplet configuration: particles only
  inside the interface, colour from the neighbour-count surface detector.

## 2. Method

1. Build a square particle lattice, clip to the desired cloud, assign
   `V_j = m_j/rho_j` so the SPH gradient is correct on an irregular support.
2. Colour field: kernel-smoothed phase indicator
   `c_i = sum_j V_j c_j W_ij / sum_j V_j W_ij` (primary case), or the raw lattice
   step (sensitivity case), or the neighbour-count field on a one-sided cloud.
3. `grad(c)` → `n` → `kappa = div(n)` using the library operators, unchanged.
4. Bin particles by radius (bin width `dx`), average `sigma*kappa*|grad c|*(n·r_hat)`
   per bin, sum over bins: that is `dp`.
5. Repeat for dx = 0.01 / 0.005 / 0.0025 m and report the error against `sigma/R`.

Reproduce with:

```bash
PYTHONPATH=src python benchmarks/surface_tension_laplace/run.py --plot
PYTHONPATH=src python -m pytest tests/test_surface_tension_laplace.py -q
```

## 3. Results

### 3.1 Two-phase cloud, kernel-smoothed colour (primary case)

| dx [m] | N | dp [Pa] | error vs sigma/R | Delta c | n · r_hat | kappa [1/m] |
|---|---|---|---|---|---|---|
| 0.0100 | 584 | 0.6467 | 10.18 % | 0.935 | 0.995 | 9.58 ± 6.97 |
| 0.0050 | 2380 | 0.6885 | 4.37 % | 0.985 | 0.996 | 9.62 ± 12.05 |
| 0.0025 | 9692 | 0.6868 | 4.61 % | 0.989 | 0.996 | 9.43 ± 25.50 |

Observed error-reduction order between successive refinements: **1.22, -0.08**
(first order, i.e. the gradient/curvature pair is not superconvergent).

### 3.2 Sensitivity: raw lattice step colour, same operators

| dx [m] | dp [Pa] | error | Delta c | kappa spread [1/m] |
|---|---|---|---|---|
| 0.0100 | 0.6090 | 15.41 % | 0.950 | 28.80 |
| 0.0050 | 0.6512 | 9.56 % | 0.999 | 54.81 |
| 0.0025 | 0.5138 | 28.64 % | 0.980 | 114.59 |

The sharp step puts the whole jump inside one smoothing length. Its second
derivative — the curvature — is then noise of order `1/h`: the spread grows with
refinement and the error is no longer monotone. Conclusion: the operators are
sound, but the colour field must be resolved by the kernel.

### 3.3 One-sided free surface (configuration shipped in `compute_surface_tension_force`)

| dx [m] | N | dp [Pa] | error vs sigma/R | Delta c | n · r_hat |
|---|---|---|---|---|---|
| 0.0100 | 284 | 0.1130 | 84.31 % | 0.272 | 0.959 |
| 0.0050 | 1208 | 0.1972 | 72.62 % | 0.280 | 0.972 |

`Delta c ≈ 0.28`: with no particles outside the interface only about a quarter of
the colour jump exists, so the force integral **cannot** reach `sigma/R`. The
one-sided free-surface force field is therefore *not* Laplace-consistent; a
two-phase or ghost-particle layer would be required.

## 4. Defect found and fixed

`compute_color_gradient` updated the `j` side of every pair with `-V_j (c_j - c_i) grad W_ij`.
The correct pair-symmetric increment is `+V_i (c_j - c_i) grad W_ij`, because
`grad_j W_ji = -grad_i W_ij` and the weight on that side is `V_i`. With the old
form the two updates cancelled whenever `V_i = V_j`, leaving only antisymmetric
residue. Measured at dx = 0.005 on the two-phase cloud:

| quantity | pre-fix form | fixed form |
|---|---|---|
| outward-normal alignment `n · r_hat` | 0.0000 | 0.9963 |
| Laplace jump dp [Pa] | 0.0293 (error 95.93 %) | 0.6885 (error 4.37 %) |
| curvature spread on the interface [1/m] | 100.1 | 12.1 |

A linear colour field `c = x` returned `|grad c| ~ 1e-17` under the old form (exact
cancellation on a uniform lattice) and `1.01` after the fix. Downstream the shipped
free-surface force field improved accordingly
(`benchmarks/static_droplet/validation_report.json`): mean inward alignment
`0.174 -> 0.805`,
median `0.996`,
positive ratio `90.7 %`,
and the mean acceleration
`0.00746 m/s²` sits at
`1.036 ×`
the capillary scale `sigma/(rho R²)`.

## 5. Error verdict (against the platform accuracy standard)

| key quantity | basis | measured error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| Laplace jump, smoothed colour, dx = 0.005 m | analytic sigma/R | 4.37 % | 否 | 是 |
| Laplace jump, smoothed colour, dx = 0.0025 m | analytic sigma/R | 4.61 % | 否 | 是 |
| resolved colour jump Delta c, dx = 0.0025 m | exact 1.0 | 1.05 % | 是 | 是 |
| gradient of a linear field, dx = 0.0025 m | exact 1.0 | 3.77 % | 否 | 是 |
| curvature `div(n)` on analytic `n = r_hat`, dx = 0.005 m | exact 1/r | 1.40 % | 是 | 是 |

* The **3 %** target is **not** met for the Laplace jump: the discretisation is
  first order and 4 % is where it lands at the affordable resolutions. The
  **15 %** acceptance bar is met.
* The Laplace jump is verified against an **analytic** reference, not against an
  experiment; the repository's `physical_validation` status stays
  `not_evaluated` for the surface-tension cases.
* The one-sided free-surface configuration is not a Laplace benchmark at all
  (`Delta c ≈ 0.28`); its `static_droplet` report says so explicitly now.

## 6. Caveats

1. Equal density on both sides of the interface. A real water/air interface has a
   density ratio of ~1000, which changes `V_j` across the interface; that case is
   not covered here.
2. Static interface only: the test constrains the force-field normalisation and
   the curvature operator. Capillary-wave dynamics and the stability of the
   capillary time-step limit are covered separately
   (`tests/test_surface_tension.py`).
3. The curvature of the *shipped* neighbour-count colour field remains noisy
   (spread ≈ 12 per m at dx = 0.005 vs an exact
   10 per m). Smoothing that colour with the kernel is the recommended next step
   (measured effect: spread 55 → 12 per m, error 9.6 % → 4.4 %).
4. `dp` is a line integral of a *modelled* body force, not an evolved pressure
   field. Tait-EOS pressure noise at `c0 = 20 m/s` is ~1 % of `rho0 c0²`, five
   orders of magnitude above the `sigma/R = 0.72 Pa` signal, so measuring the jump
   by running the droplet to equilibrium is not feasible in this weakly
   compressible formulation — hence the operator-level benchmark.

## 7. Files

| path | content |
|---|---|
| `benchmarks/surface_tension_laplace/run.py` | driver (sweeps, criteria, figure) |
| `benchmarks/surface_tension_laplace/validation_report.json` | machine-readable results |
| `benchmarks/surface_tension_laplace/laplace_convergence.png` | error vs dx, resolved colour jump |
| `tests/test_surface_tension_laplace.py` | 7 regression tests (operator + Laplace + legacy form rejected) |
| `benchmarks/static_droplet/` | one-sided free-surface sanity check (scope-narrowed) |

## References

* Brackbill, Kothe & Zemach (1992), *A continuum method for modeling surface tension*, J. Comput. Phys. 100, 335.
* Morris (2000), *Simulating surface tension with smoothed particle hydrodynamics*, Int. J. Numer. Methods Fluids 33, 333.
* Adami, Hu & Adams (2010), *A new surface-tension formulation for multi-phase SPH using a reproducing divergence approximation*, J. Comput. Phys. 229, 5011.
