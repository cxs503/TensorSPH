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
