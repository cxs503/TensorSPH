# Free-surface completeness closure — programmatic tables

Source: `benchmarks/surface_tension/out_free_surface_completeness/gate.json` (runtime 3.4 s).

## Laplace ladder (single-phase vacuum reduction, h/dx = 1.3, Wendland C2)

| R/dx | closure | dp [Pa] | sigma/R [Pa] | error [%] | momentum residual |
|---|---|---|---|---|---|
| 16 | one-sided (shipped) | 0.50585 | 0.72000 | -29.74 | 3.2e-17 |
| 16 | specular mirror | 1.02051 | 0.72000 | +41.74 | 6.0e-17 |
| 16 | support completeness | 0.74917 | 0.72000 | +4.05 | 3.2e-17 |
| 16 | specular mirror, fitted alpha*=0.4263 | 0.72527 | 0.72000 | +0.73 | 8.4e-17 |
| 32 | one-sided (shipped) | 0.50396 | 0.72000 | -30.01 | 5.1e-17 |
| 32 | specular mirror | 1.01070 | 0.72000 | +40.38 | 9.5e-18 |
| 32 | support completeness | 0.74283 | 0.72000 | +3.17 | 3.4e-17 |
| 32 | specular mirror, fitted alpha*=0.4263 | 0.72000 | 0.72000 | +0.00 | 2.9e-17 |
| 64 | one-sided (shipped) | 0.50485 | 0.72000 | -29.88 | 2.2e-17 |
| 64 | specular mirror | 1.00640 | 0.72000 | +39.78 | 1.1e-17 |
| 64 | support completeness | 0.74269 | 0.72000 | +3.15 | 4.4e-17 |
| 64 | specular mirror, fitted alpha*=0.4263 | 0.71868 | 0.72000 | -0.18 | 3.4e-17 |

## Support measurement (R/dx = 32)

| inward depth / dx | support fraction s = trace(M)/2 |
|---|---|
| 0.5 | 0.6282 |
| 1.5 | 0.9188 |
| 2.5 | 0.9729 |
| 3.5 | 0.9739 |
| 4.5 | 0.9739 |
| 5.5 | 0.9739 |
| 6.5 | 0.9739 |
| 7.5 | 0.9739 |
| 8.5 | 0.9739 |
| 9.5 | 0.9739 |
| 10.5 | 0.9739 |
| 11.5 | 0.9739 |
| 12.5 | 0.9739 |
| 13.5 | 0.9739 |
| 14.5 | 0.9739 |
| 15.5 | 0.9739 |
| 16.5 | 0.9739 |
| 17.5 | 0.9739 |
| 18.5 | 0.9739 |
| 19.5 | 0.9739 |
| 20.5 | 0.9739 |
| 21.5 | 0.9739 |
| 22.5 | 0.9739 |
| 23.5 | 0.9739 |
| 24.5 | 0.9739 |
| 25.5 | 0.9739 |
| 26.5 | 0.9739 |
| 27.5 | 0.9739 |
| 28.5 | 0.9739 |
| 29.5 | 0.9739 |
| 30.5 | 0.9739 |
| 31.5 | 0.9739 |

## Cross-parameter sweep (closure on, R/dx = 32)

| kernel | h/dx | error [%] |
|---|---|---|
| wendland | 1.3 | +3.17 |
| wendland | 1.5 | +6.41 |
| wendland | 2.0 | +7.59 |
| cubic | 1.3 | +7.40 |
| cubic | 1.5 | +7.73 |
| cubic | 2.0 | +7.89 |

## Gate

```json
{
 "criterion_pct": 15.0,
 "ladder_err_pct": {
  "R_over_dx=16": 4.051096231135122,
  "R_over_dx=32": 3.170166133872286,
  "R_over_dx=64": 3.151727966843429
 },
 "worst_abs_err_pct": 4.051096231135122,
 "within_3pct": {
  "R_over_dx=16": false,
  "R_over_dx=32": false,
  "R_over_dx=64": false
 },
 "baseline_one_sided_worst_abs_pct": 30.005905845102472,
 "closure_gain_factor": 7.406860793502992,
 "sweep_worst_abs_pct": 7.8946094184760724,
 "verdict": "pass"
}
```

