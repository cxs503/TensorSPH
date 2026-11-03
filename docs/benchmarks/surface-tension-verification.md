# Young–Laplace verification of the surface tension model (B5)

**Status: the model is NOT quantitatively verified.** The current formulation is
structurally sound but its amplitude is wrong by 30 % at the coarsest
resolution and gets worse under refinement. The `PASS` recorded for this
feature when it was merged (m277, `benchmarks/static_droplet/validation_report.json`)
was produced by an acceptance criterion that the data did not satisfy; the
evidence is in §1 and the criterion is withdrawn in §5.

Everything below is reproducible from a clean checkout:

```bash
PYTHONPATH=src python benchmarks/laplace_pressure/run.py \
    --json-out benchmarks/laplace_pressure/validation_report.json
PYTHONPATH=src python benchmarks/laplace_pressure/run.py --scheme legacy   # 0.8.0
```

## 1. Why this case exists

The m277 report declared the static-droplet benchmark `PASS` with the
criterion *"Force magnitude is non-zero and physically reasonable"*. The two
numbers it printed next to each other are the problem:

| quantity | m277 report | analytic |
|---|---|---|
| computed Laplace pressure `sigma/R` | — | 0.7200 Pa |
| "expected acceleration scale" | 0.7200e-3 m/s² | 0.7200e-3 m/s² |
| **measured max acceleration** | **0.5585 m/s²** | — |
| ratio measured / expected | **776** | 1 |

and, in the same JSON, `avg_dot_product = 0.174` against a docstring criterion
of *"dot product > 0.5"*. The report kept the number, relaxed the criterion to
the median (`0.453 > 0.3`) and recorded `PASS`. A model that applies traction
776× too large and makes 40 % of the surface push outward is not validated;
it was recorded as validated. That is the debt this case pays.

## 2. Analytic reference

For a 2-D droplet, Young–Laplace is `p_in - p_out = sigma / R`. The equivalent
**force** statement — which can be tested without time integration — is the
capillary force balance across a diameter:

    |F_half| = 2 sigma          (force per unit depth)

Cut the droplet along any diameter: the free surface of the half is a
semicircular arc, and integrating `sigma * kappa * n` along it gives exactly
`2 sigma` across the cut. Three properties make this the best available gate:

* the target does **not** depend on `R`, on `dx` or on `c0`, so a correct
  implementation returns the same ratio everywhere — any spread is
  discretisation error, not physics;
* the traction on a **closed** surface must sum to exactly zero, and the
  sampling is mirror-symmetric, so a spurious transverse force must be exactly
  zero. These are invariances, not tolerances;
* `sigma_eff / sigma = |F_half| / (2 sigma)` is directly interpretable as *the
  surface tension the model actually applies*.

No experimental data is needed, so the provenance problem that keeps the B2
dam-break experimental gate `not_evaluated` does not apply here.

## 3. Numerics

Static evaluation of the traction field on an undeformed circular droplet on a
square lattice (centred, so the sampling itself is symmetric), `rho0 = 1000`,
`sigma = 0.072 N/m`, `h = 1.3 dx`, `cell_linked_pairs` with a `2h` support.
`F_half` is `sum_i m_i a_i` over the `x > 0` half.

## 4. Results

`sigma_eff / sigma` — 1.000 is correct. The same value appears wherever `R/h`
coincides, so the error is a function of `R/h` alone.

| `R` (m) | `dx` (m) | `N` | `R/h` | `sigma_eff/sigma` |
|---|---|---|---|---|
| 0.04 | 0.005 | 208 | 6.2 | 0.704 |
| 0.06 | 0.005 | 448 | 9.2 | 0.919 |
| 0.08 | 0.005 | 812 | 12.3 | 1.087 |
| 0.04 | 0.0025 | 812 | 12.3 | 1.087 |
| 0.06 | 0.0025 | 1804 | 18.5 | 1.499 |
| 0.08 | 0.0025 | 3228 | 24.6 | 1.878 |
| 0.04 | 0.00125 | 3228 | 24.6 | 1.878 |
| 0.06 | 0.00125 | 7232 | 36.9 | 2.576 |
| 0.08 | 0.00125 | 12892 | 49.2 | 3.340 |

Structural observables (exact target in brackets):

| observable | current | 0.8.0 (m277) | target |
|---|---|---|---|
| mirror asymmetry `|F_y|/|F_x|` | **3.8e-15** | 0.415 – 0.457 | 0 |
| 90° rotation asymmetry | **3.9e-15** | 0.725 – 0.932 | 0 |
| net force on closed surface `|F_tot|/(2sigma)` | **2.7e-14** | — | 0 |
| max traction in the bulk `(r<R/2)/max` | **0.0** | 0.0 | 0 |
| fraction of surface pulled inward | 0.935 | 0.582 – 0.617 | 1 |

Observed order of accuracy of `sigma_eff/sigma`: **−1.57, −4.29, −4.74**
(negative = the error grows with refinement, in proportion to `1/h`).

## 5. Verdict, gate by gate

| gate | analytic target | measured | tolerance | result |
|---|---|---|---|---|
| mirror symmetry | 0 | 3.8e-15 | 0.05 | **pass** |
| 90° rotation symmetry | 0 | 3.9e-15 | 0.05 | **pass** |
| closed-surface momentum | 0 | 2.7e-14 | 0.05 | **pass** |
| bulk traction free | 0 | 0.0 | 1e-3 | **pass** |
| traction points inward | 1 | 0.935 | 1.000 | **FAIL** |
| capillary force balance | 1 ± 15 % | 0.704 – 3.340 | 0.15 | **FAIL** |
| refinement convergence | positive order | −4.74 | 0 | **FAIL** |

**Is the key quantity within 15 % of the analytic reference? No.** At `R/h = 6.2`
the model applies 70 % of the nominal surface tension; at `R/h = 49.2` it
applies 334 %. The sign of the order of accuracy is wrong: refining the mesh
makes agreement worse, in proportion to `1/h`. The model is therefore not
convergent in the quantity that Young–Laplace constrains, and the answer to
"is it within 3 %?" is no by two orders of magnitude.

## 6. What was fixed here, and what was not

The colour field was replaced by the SPH kernel sum (Akinci, Akinci & Teschner
2013): `c_i = sum_j V_j W_ij`, whose analytic gradient `sum_j V_j grad W_ij`
**vanishes identically for a complete stencil** — measured 8.9e-15 on the bulk
stencil of this case. Three defects of the 0.8.0 formulation disappear as a
consequence, and they are the reason the numbers in the last column above are
so bad:

1. **Arbitrary surface detection.** Surface particles were `neighbours <
   0.7 * mean(neighbours)`, a threshold with no physical content. It cut a
   ragged, lattice-dependent band, which is why a symmetric droplet received a
   transverse force of 40 % of the radial one. The kernel-sum gradient needs no
   threshold: it is zero in the bulk and `O(1/h)` on the surface.
2. **Unnormalised colour field.** `c = neighbours / mean(neighbours)` is neither
   a phase indicator nor `O(1)` across the interface, so `delta_s = |grad c|`
   carried an arbitrary amplitude into the force. With the kernel sum,
   `|grad c|` integrates to one across the interface by construction.
3. **Momentum.** Because the traction now sums to 2.7e-14 of `2 sigma` on a
   closed surface instead of an `O(1)` fraction, the model no longer creates
   momentum.

What was **not** fixed, and is the whole of the remaining error: `kappa` is a
second derivative of a field that is only about `2h` wide. The discrete
divergence of the normal over a one-particle-wide band is biased at `O(1/h)`,
so `kappa` measures ≈2× the true `1/R` at `R/h = 12` and ≈17× at `R/h = 49`.
This is the known hard part of single-phase SPH surface tension and it needs a
different estimator (curvature from a fitted/MLS surface, or a renormalised
divergence with a smoothed normal field), not a parameter change. Damping it by
tuning `sigma` would be curve-fitting, so it was not done.

## 7. Consequences

* `benchmarks/static_droplet` is **superseded** by this case; its `PASS` is
  withdrawn (see `superseded_by` in its report). It measured agreement to
  within a factor of 776 and called it validation.
* The feature must be described as *implemented, structurally correct,
  amplitude unverified*. Nothing that depends on the absolute value of `sigma`
  (Weber-number scaling, droplet oscillation frequency, breakup) should be run
  against a reference until §6 is closed.
* CI runs the four structural gates (they are exact, and they were violated
  before this change). The three quantitative gates are reported as `fail` in
  the JSON and in this table — recorded, not waived. `--strict` makes the exit
  code follow them.

## 8. Next steps

1. Replace the curvature estimator. Candidate: MLS/fitted-sphere curvature over
   the interface band, or `kappa = -div(n)` on a normal field smoothed over
   `2h`, tested by this harness at three resolutions. Acceptance: positive
   observed order and `|sigma_eff/sigma - 1| < 0.15` at every level.
2. Only after that, revisit the dynamic form of the gate (run to equilibrium
   and read `p_in` from the Tait EOS): it measures the same relation and would
   confirm that the static balance translates into a pressure jump.
3. Re-check the capillary time-step limit once the amplitude is right; the
   current `sqrt(rho h^3 / sigma)` bound is unrelated to the observed error but
   has never been exercised against a resolved capillary wave.