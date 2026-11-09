# Two-phase Laplace-law verification of the CSF model

**Scope**: TensorSPH `src/tensorsph/csf_consistent.py`, `csf_curvature.py`

**Purpose**: Verify the CSF surface-tension chain against the analytic Young-Laplace
jump `dp = sigma / R` using a **two-phase** cloud where particles exist on both
sides of the interface, so the colour function completes the full 1 → 0 transition.

## Key finding

The CSF operators (kernel-sum colour, MLS gradient/divergence, fit_quadratic
curvature) are **correct in principle**: when the interface is fully resolved
(two-phase), the Laplace error is within the platform gate. The ~29 % effective
sigma deficit in the shipped one-sided droplet is a **free-surface truncation
effect**, not a bug in the discrete operators.

| Case | R/h | sigma_eff/sigma | < 15 %? |
|------|-----|-----------------|---------|
| Two-phase (this benchmark) | 10.3 | 0.95+ | yes |
| One-sided (shipped) | 6.15 | 0.71 | no |

## Root cause of the one-sided deficit

In the one-sided droplet, the colour function `c = sum V_j W_ij` falls from ~1
in the bulk to ~0.63 at the outermost particle layer. The remaining 0.63 → 0
transition lives in the void where there are no particles. The CSF delta function
`delta_s = 2 |grad c|` therefore carries only ~75 % of the unit mass required
by the Brackbill identity `int delta_s dn = 1`.

The factor of 2 in `delta_s = 2 |grad c|` assumes `c_surface = 0.5`, but the
actual minimum colour on a hex lattice with h = 1.3 dx is ~0.63. The correct
normalization factor is `1 / (1 - c_surface) ≈ 2.74`, not 2.0.

## Implications

1. The CSF model is **physically correct** for fully resolved interfaces
2. The one-sided free surface requires a **geometry-aware delta_s normalization**
3. Future work: implement the local normalization factor `1 / (1 - c_surface)`
   and verify it brings the one-sided effective sigma within 15 %

## Files

- `benchmarks/two_phase_laplace/run.py` — benchmark script
- `benchmarks/two_phase_laplace/validation_report.json` — results
- `tests/test_two_phase_laplace.py` — regression test
