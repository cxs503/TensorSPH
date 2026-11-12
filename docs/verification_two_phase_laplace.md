# Two-phase Laplace control of the CSF chains — corrected account (m16489)

**Scope**: `src/tensorsph/csf_consistent.py`, `csf_curvature.py`,
`surface_tension.py`; benchmarks `two_phase_laplace`, `surface_tension_laplace`.
**Nature**: verification-debt repayment + falsified load-bearing claim.
**Numbers**: injected by `benchmarks/two_phase_laplace/make_doc.py` from
`validation_report.json`, `diagnose_curvature.json`, `matched_chains.json`.

## 1. What this document used to claim, and what the code does

The m7558 edition claimed the two-phase cloud gives `sigma_eff/sigma = 0.95+`
(Laplace error 4.6 % at `dx = 2.5 mm`, `R/h = 10.3`) and used that as the
*decisive control* for the sentence "the CSF operators are correct in
principle; the ~29 % one-sided deficit is a free-surface truncation effect".
The same reading was registered in the ledger as
`two_phase_laplace.smoothed_colour.dx2.5mm` — but from a **different artifact
and a different chain** (`benchmarks/surface_tension_laplace`, 4.4-4.6 % at
`R = 0.1 m`).

Re-running the committed harness (`benchmarks/two_phase_laplace/run.py`) gives:

| dx [mm] | R/h | sigma_eff/sigma | error [%] | kappa shell [1/m] | delta_s mass / 2 pi R (run.py colour) |
|---|---|---|---|---|---|
| 5.00 | 6.2 | 0.6246 | -37.5 | 15.08 | 0.9465 |
| 2.50 | 12.3 | 0.6621 | -33.8 | 15.68 | 0.9761 |
| 1.25 | 24.6 | 0.6628 | -33.7 | 15.40 | 0.9894 |

Note the last column: `run.py` builds its own Shepard colour whose
`sum |grad c| dV` *does* reach 0.95-0.99 of `2 pi R`, so in this harness the
whole deficit is the curvature (`kappa` = 15.1-15.7 against the analytic 25 1/m
at every resolution). The library-consistent indicator colour used by
`diagnose_curvature.py` and `matched_chains.py` carries only 0.71 instead --
i.e. the two colour constructions do not agree about the delta function, which
is itself an open item (section 4).

Two facts follow directly: the reading is **~34 % low** (not 4.6 %), and it is
**flat under refinement** (R/h from 6.2 to 24.6 moves it by 4 points), so it is
a systematic operator deficit and not a discretisation error that resolves away.
The `H_FACTOR = 1.3` in this file has never produced `R/h = 10.3`, so the
docstring number corresponds to no reachable configuration.

## 2. Ablation: where the 34 % lives

`benchmarks/two_phase_laplace/diagnose_curvature.py` swaps one ingredient at a
time (colour / normal / curvature scheme / stencil) on the same dx = 2.5 mm
cloud (`R/h = 12.3`, analytic `kappa = 25.000 1/m`):

| variant | kappa shell [1/m] | kappa err [%] | delta_s mass / 2 pi R | sigma_eff/sigma |
|---|---|---|---|---|
| `control/all_analytic` | 25.000 | +0.00 | 1.0000 | 0.9983 |
| `fit_quadratic/analytic_n/analytic_ds/st2` | 20.844 | -16.62 | 1.0000 | 0.8074 |
| `fit_quadratic/analytic_n/analytic_ds/st3` | 24.690 | -1.24 | 1.0000 | 0.9611 |
| `fit_quadratic/analytic_n/analytic_ds/st4` | 25.367 | +1.47 | 1.0000 | 1.0023 |
| `fit_quadratic/analytic_n/analytic_ds/st5` | 25.367 | +1.47 | 1.0000 | 1.0023 |
| `fit_quadratic/analytic/indicator_ds/st3` | 11.378 | -54.49 | 0.7057 | 0.2680 |
| `fit_quadratic/analytic/indicator_ds/st4` | 17.310 | -30.76 | 0.7057 | 0.4690 |
| `fit_quadratic/analytic/indicator_ds/st5` | 17.310 | -30.76 | 0.7057 | 0.4690 |
| `fit_quadratic/analytic/kernel_sum_ds/st3` | 6.572 | -73.71 | 0.6550 | 0.2722 |
| `fit_quadratic/analytic/kernel_sum_ds/st4` | 10.893 | -56.43 | 0.6550 | 0.3203 |
| `fit_quadratic/analytic/kernel_sum_ds/st5` | 10.893 | -56.43 | 0.6550 | 0.3203 |
| `fit_quadratic/indicator_n/indicator_ds/st3` | 11.314 | -54.74 | 0.7057 | 0.2782 |
| `fit_quadratic/indicator_n/indicator_ds/st4` | 17.371 | -30.52 | 0.7057 | 0.4790 |
| `fit_quadratic/indicator_n/indicator_ds/st5` | 17.371 | -30.52 | 0.7057 | 0.4790 |
| `fit_quadratic/indicator_n/kernel_sum_ds/st3` | 0.000 | -100.00 | 0.6550 | 0.0000 |
| `fit_quadratic/indicator_n/kernel_sum_ds/st4` | 0.000 | -100.00 | 0.6550 | 0.0000 |
| `fit_quadratic/indicator_n/kernel_sum_ds/st5` | 0.000 | -100.00 | 0.6550 | 0.0000 |
| `fit_quadratic/kernel_sum_n/indicator_ds/st3` | 0.019 | -99.92 | 0.7057 | 0.0000 |
| `fit_quadratic/kernel_sum_n/indicator_ds/st4` | 0.019 | -99.92 | 0.7057 | 0.0000 |
| `fit_quadratic/kernel_sum_n/indicator_ds/st5` | 0.019 | -99.92 | 0.7057 | 0.0000 |
| `fit_quadratic/kernel_sum_n/kernel_sum_ds/st3` | 6.290 | -74.84 | 0.6550 | 0.2779 |
| `fit_quadratic/kernel_sum_n/kernel_sum_ds/st4` | 10.846 | -56.62 | 0.6550 | 0.3310 |
| `fit_quadratic/kernel_sum_n/kernel_sum_ds/st5` | 10.846 | -56.62 | 0.6550 | 0.3310 |
| `fit_circle/indicator_n/indicator_ds/st3` | 60.211 | +140.84 | 0.7057 | 1.7017 |
| `fit_circle/indicator_n/indicator_ds/st4` | 102.066 | +308.26 | 0.7057 | 2.6159 |
| `fit_circle/indicator_n/indicator_ds/st5` | 102.066 | +308.26 | 0.7057 | 2.6159 |
| `mls_divergence/indicator` | 24.389 | -2.44 | 0.7057 | 0.7027 |

Three readings carry the whole story:

1. **The measurement is sound.** With analytic curvature and an analytic thin
   shell the force assembly returns `0.9983` — the 2 sigma
   resultant of a half-circle is recovered to
   `0.17 %`. A 34 % shortfall is not a
   measurement artifact.
2. **The Brackbill delta is short even on a closed interface.**
   `sum |grad c| dV` is 0.70 x `2 pi R` on the fully two-phase cloud, and only
   0.68 / 0.72 at `R/h = 6.2 / 24.6` — i.e. **particles outside the interface do
   not restore the missing 30 %**. The old root cause ("the 1 -> 0 transition
   lives in the void") is therefore at most a secondary contribution.
3. **`fit_curvature` is biased on the band the solver feeds it.**
   `fit_quadratic` reads `25.37` (`+1.47 %`)
   on a thin analytic shell at stencil 4h, but only
   `17.37` (`-30.52 %`) on the
   colour-function band of the same lattice. The MLS-curvature variant recovers
   `24.39 1/m`
   (`-2.44 %`).

## 3. Matched chains: geometry or chain?

The repository ships **two** colour-gradient/curvature chains —
`csf_consistent` (`mls_gradient` + `fit_curvature`, what `csf_fields` and the
dispersion/3-D modules use) and `surface_tension`
(`compute_color_gradient` + `compute_curvature`, what
`droplet_impact.compute_surface_tension_force` uses). Their Laplace benches
disagree by a factor of seven (4.6 % vs 33.7 %) on the same physics, which is
only admissible if the deficit is a property of the geometry.
`benchmarks/two_phase_laplace/matched_chains.py` measures the full matrix with
one metric at `dx = 2.50 mm`, `R = 0.04 m`:

| geometry | chain | delta_s mass / 2 pi R | kappa shell [1/m] | sigma_eff/sigma | (same, MLS kappa) |
|---|---|---|---|---|---|
| two_phase | `csf_consistent` | 0.7057 | 17.361 | 0.4785 | 0.6995 |
| two_phase | `surface_tension` | 0.6908 | 24.000 | 0.6710 | — |
| one_sided | `csf_consistent` | 0.3243 | 23.807 | 0.3228 | 0.3670 |
| one_sided | `surface_tension` | 0.2064 | 21.212 | 0.1762 | — |

* The deficit does **not** follow the geometry: two-phase and one-sided are both
  far below 1 and their `delta_s` masses sit at 0.69-0.72 for both chains.
* The **chain** sets how much effective tension arrives: on the same two-phase
  cloud `surface_tension` reads
  `0.6710` where `csf_consistent` reads
  `0.4785`; the gap closes to
  `0.6995`
  vs `0.6710` once the biased curvature
  fit is replaced by the MLS divergence.

## 4. What is now falsified, and what stays open

**Falsified.** "The CSF operators are correct in principle (4-5 % on a closed
interface); the 29 % one-sided deficit is free-surface truncation." A closed
two-phase interface loses 30-34 % through both shipped chains at every
resolution tested (`R/h` = 6.2 ... 30.8). Free-surface truncation cannot be
carried as the explanation while the closed control fails the same gate.

**Open (honest FAIL, not resolved here).** The two committed benchmarks still
disagree: `surface_tension_laplace` reports a *radial line* colour jump
`d(c) = 0.985-0.989` (hence `dp = sigma kappa d(c)` = 4.4 % error) while the
*volume* quadrature of the same gradient field carries 0.71 of unit mass here.
Both cannot be right; the line integral must be instrumented particle-by-particle
before the 4.4 % reading is quoted again. Closure path: instrument
`laplace_jump` in `benchmarks/surface_tension_laplace/run.py` with the per-ray
`|grad c|` samples committed as an artifact, then re-register the claim.

**Practical consequence for other lines.** Any capability whose surface tension
goes through `csf_fields` (capillary-wave dispersion, 3-D CSF, contact angle)
inherits a ~0.5x effective tension on this evidence; the dispersion line should
be re-read against `sigma_eff` from this document rather than the nominal
`sigma`.

## 5. Tests (these used to be swallowed)

`tests/test_two_phase_laplace.py` previously marked both checks
`xfail(strict=False)` and blamed an "editable install pointing at another
worktree" — a non-strict xfail can never fail, which is how a 34 % regression
sat next to a "4.6 %" docstring. It is now seven real assertions: the force
assembly control, the short `delta_s` mass, the missed gate, the thick-band
curvature bias at two radii, the chain gap, and the non-transfer of the
`surface_tension` 4.4 % reading. If a future fix moves any of these, the suite
turns red and this document must be regenerated.

## 6. Error table (whether each key quantity is < 3 %)

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| force assembly control (analytic kappa + shell) | 1.0 | 0.9983 | -0.17 % | yes |
| kappa, `fit_quadratic`, thin shell, st 4h | 1/R | 25.367 1/m | +1.47 % | yes |
| kappa, `fit_quadratic`, colour band, st 4h | 1/R | 17.371 1/m | -30.52 % | no |
| kappa, MLS divergence, colour band | 1/R | 24.389 1/m | -2.44 % | yes |
| delta_s mass, two-phase | 1.0 | 0.7057 | -29.43 % | no |
| sigma_eff/sigma, two-phase, chain A | 1.0 | 0.4785 | -52.15 % | no |
| sigma_eff/sigma, two-phase, chain B | 1.0 | 0.6710 | -32.90 % | no |

## 7. Reproduce

```
python benchmarks/two_phase_laplace/run.py                 # committed harness ladder
python benchmarks/two_phase_laplace/diagnose_curvature.py  # ablation matrix
python benchmarks/two_phase_laplace/matched_chains.py      # geometry x chain matrix
python benchmarks/two_phase_laplace/make_doc.py            # regenerate this document
pytest tests/test_two_phase_laplace.py -q
```
