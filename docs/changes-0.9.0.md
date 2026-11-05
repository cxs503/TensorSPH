# What changed in 0.9.0

Verification of the surface-tension model, and two gate defects found while doing
it. No physics default is changed.

| Item | Before | After |
|---|---|---|
| B3 (Laplace verification) | "not started, no model" in the gap register; a merged benchmark printed `PASS` without testing the criterion it stated | `benchmarks/surface_tension/laplace_colour.py`: curvature against `1/R` and the pressure jump against `sigma/R` at `R/dx = 8 / 12 / 16`, for three colour functions, with density renormalisation and a pre-stressed start |
| Curvature of a face-centred circle | the merged `neighbor_count` colour has no surface normal (mean cosine with `r_hat`: +0.015) and gives a curvature of the wrong sign that worsens under refinement (-0.35 / -0.58 / -0.92 of `1/R`) | `support_sum` colour: normal from the smoothed support vector, curvature 1.017-1.043 of `1/R` |
| Laplace pressure | not measured; at water-like `c0` the signal is 1.1e-5 in relative density, below the summation noise | measured: 0.70 / 0.73 / 0.53 of `sigma/R` for `support_sum`, -0.01 / 0.07 / 0.19 for `neighbor_count`. The shortfall is traced to `int delta_s dr = 0.67` and is left open (see the doc) |
| `static_droplet` gate | evaluated `median_dot > 0.3 or positive_ratio > 0.6` while recording "dot product > 0.5"; recorded mean 0.17 and still printed PASS | evaluates the stated criterion; the acceleration reference is `sigma/(rho R dx)` instead of `sigma/(rho R)` (off by `1/dx`) |
| New config fields | -- | `surface_tension_color`, `surface_tension_smoothing`, `surface_tension_delta_scale`; all defaults preserve the 0.8.0 behaviour |
| Tests | 104 | 111 (+7 analytic anchors: curvature against `1/R`, normal orientation, a negative control for the unsmoothed normalisation, the end-to-end pressure jump, and the delta-scale linearity) |

The default colour model is unchanged. Switching it would move every
capillary-adjacent run, and the replacement still owes the delta-function
normalisation before it can claim the Laplace law.
