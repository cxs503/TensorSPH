# Laplace-law verification of the CSF surface-tension model

**Case** `benchmarks/laplace/run.py` · stage A = operator vs an analytic circle ·
stage B = free-surface droplet relaxation · 922 particles in stage B,
2116 quadrature points in the stage-A patch · torch 2.7.1 /
python 3.11.6 · total wall 94 s (CPU, 128 cores
visible, run under heavy machine load).

## 1. What is verified, and why

`src/tensorsph/surface_tension.py` (m277 / PR #13) implements the Continuum
Surface Force model (Brackbill, Kothe & Zemach 1992) and shipped ten unit tests.
None of them compares the model with an analytic surface-tension result: they
check shapes, the `sigma = 0` short circuit and unit-length normals. The
reference used here is the Laplace law for a 2-D static droplet:

    Delta p = p_in - p_out = sigma / R

Two independent measurements are reported: the **operator** (curvature and
interface delta function on a *prescribed* analytic interface) and the
**free-surface droplet** the solver actually carries in `rhs`.

## 2. A defect found on the way (fixed in this branch)

`compute_color_gradient` negated the j-side `index_add` and used `V_j` for it.
Each pair then contributed opposite signs to its two endpoints, so the two
roughly equal populations cancelled and the gradient of a *smooth* field came
out as noise around zero. Measured on the linear field `c = 3x`: the interior
mean gradient was `1e-18` before the fix and 3.0316
after it, against the analytic `3.0` (rel. err
1.05 %). The CSF force was therefore
inert, and all ten original tests still passed -- a uniform colour field has a
zero gradient in either version.

## 3. Stage A -- operator verification (PASS)

`c(r) = 0.5 (1 - tanh((r - R)/w))` sampled on a full lattice patch of
half-width 56 mm (20158 pairs), `h = 1.3 dx`,
`R = 40 mm`, `dx = 2.5 mm`. `kappa` must equal `1/R = 25 1/m` and
`integral |grad c| dr` must equal the colour jump (= 1). Their product is the
Laplace jump implied by the discretised operator.

| w/dx | ring particles | kappa [1/m] | kappa err | int\|grad c\|dr | int err | Delta p [Pa] | Delta p err |
|---|---|---|---|---|---|---|---|
| 1.0 | 382 | 25.299 | 1.20 % | 1.011 | 1.05 % | 1.8408 | 2.26 % |
| 1.5 | 474 | 25.404 | 1.62 % | 1.010 | 1.05 % | 1.8483 | 2.68 % |
| 2.0 | 596 | 25.287 | 1.15 % | 1.010 | 1.00 % | 1.8388 | 2.15 % |
| 3.0 | 842 | 25.299 | 1.20 % | 1.005 | 0.48 % | 1.8303 | 1.68 % |

Analytic `Delta p = sigma/R = 1.8000 Pa`. The error is flat
in `w/dx` (1.68 -
2.68 %), i.e. the
operator is *consistent* rather than tuned for one interface width. Headline at
`w = 2 dx`: kappa = 25.287 1/m (1.15 %),
`int |grad c| dr` = 1.010
(1.00 %), Delta p = 1.8388 Pa,
i.e. **2.15 %**.

Method note that matters: evaluating the analytic colour field *only on the
fluid side* (the droplet lattice) removes the outer half of the tanh, drops
`integral |grad c| dr` to 0.35 instead of 1 and biases every Laplace number by a
factor ~3. An operator test must span both sides of the interface.

## 4. Stage B -- free-surface droplet (FAIL, 57.8 %)

Static droplet, `gravity = 0`, no walls, `rho0 = 1000`,
`sigma = 0.072`, `c0 = 0.8`, `nu = 0.002`,
hex lattice `R = 40 mm`, `R_eq = 39.86 mm`.

| quantity | value |
|---|---|
| dt | 6.602e-04 s (binding limit: library(stable_dt)) |
| steps / sim time | 2857 / 1.886 s = 2.00 tau_sigma |
| tau_sigma = sqrt(rho R^3 / sigma) | 0.943 s |
| Delta p measured, relaxed | 0.7629 Pa |
| Delta p static force balance, t = 0 | 0.3346 Pa |
| Delta p analytic sigma/R_eq | 1.8065 Pa |
| relative error, relaxed | **57.77 %** |
| relative error, static balance | **81.48 %** |

Both measurements under-predict, so the routine is **not acceptable** under the
15 % criterion, and the 3 % figure is missed by an order of magnitude.

## 5. Why it under-predicts, and what would fix it

The jump equals `sigma * integral kappa * delta_s dA`, with `delta_s = |grad c|`
normalised so that `integral delta_s dr` across the interface equals the colour
jump. In a single-phase droplet the exterior is vacuum: there are no particles
outside `r = R`, so `|grad c|` is only integrated over the inner half of the
interface and `integral delta_s dr` reaches only ~0.25-0.35 of the jump. The
colour function used by `compute_surface_tension_force` -- neighbour counts with
a `0.7 * <count>` threshold, whose own jump is ~0.4 rather than 1 -- multiplies
the deficit. Both factors belong to the *indicator*; the curvature operator that
stage A exercises is accurate to ~0.3 %.

Remediation options, cheapest first:

1. normalise `delta_s` per interface so that its integral over the band equals
   the colour jump between the interior and the outermost shell;
2. use a density-ratio colour function `c = rho/rho0` (which the module
   docstring already claims) with an explicit free-surface threshold, and
   normalise `delta_s` by that jump;
3. two-phase CSF with a genuine exterior phase, which is what Adami, Hu &
   Adams (2010) use for their Laplace test and what removes the half-support
   factor entirely.

`tests/test_laplace_validation.py::test_free_surface_droplet_laplace_deficit`
records the current deficit as `xfail(strict=True)`: whoever fixes it turns the
suite red on purpose, so the fix cannot be merged silently.

## 6. Pitfalls recorded

- The first stage-B attempt used `dt = 0.2 h / c0` (acoustic limit only) and
  diverged within 40 steps to ~9 km/s. At `nu = 1e-2` the viscous limit
  `0.125 h^2 / nu` is five times tighter, and the same divergence occurs with
  `sigma = 0`: it is a base-solver instability, not a CSF one. The benchmark now
  takes the minimum of the library's `stable_dt` and `capillary_dt_limit`.
- A single instant of a relaxing droplet is not converged. The table therefore
  also reports the static momentum balance `Delta p = -integral rho (a . r_hat) dA`,
  which needs no time marching and is bit-reproducible.

## 7. Error verdict

| quantity | basis | error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| colour-gradient consistency | analytic linear field | 1.05 % | yes | yes |
| curvature kappa | analytic 1/R | 1.15 % | yes | yes |
| interface delta integral | analytic colour jump | 1.00 % | yes | yes |
| Laplace jump, operator level | analytic sigma/R | 2.15 % | yes | yes |
| Laplace jump, free-surface droplet | analytic sigma/R_eq | 57.77 % | **no** | **no** |

The operator (after the sign fix) reproduces the Laplace law to better than 3 %.
The free-surface force routine does not, and this report states it rather than
rounding it away. Figure `artifacts/laplace/laplace.png`: (a) kappa and
`integral |grad c| dr` against `w/dx` with the analytic line, (b) the droplet
pressure jump history against `sigma/R_eq(t)`.
