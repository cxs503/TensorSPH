# Tables (generated from `benchmarks/surface_tension/out_mixture_laplace/gate.json`)

Reproduce with:

```bash
PYTHONPATH=src python benchmarks/surface_tension/mixture_laplace.py \
    --radii-dx 16,32,64 --rho-ratios 1,0.1,0.01,0.001
python runs/m18642_round2/emit_tables.py
```

wall clock of the whole sweep: 7.2 s (CPU, no cluster needed)

## T0. Case and reference

| quantity | value |
|---|---|
| `model` | `SPHinXsys continuous surface stress + mixture contact fraction` |
| `raw_sources` | `docs/xval_sphinxys_surface_tension/raw/ (sha256 in MANIFEST.json)` |
| `theory` | `Young-Laplace, 2-D circular interface: Delta p = sigma/R` |
| `sigma_n_m` | `0.072` |
| `radius_m` | `0.1` |
| `rho_water` | `1000.0` |
| `rho_air` | `1.0` |
| `h_over_dx` | `1.3` |

## T1. Discrete-operator floor (analytic unit-integral stress bump)

`S = sigma f(r) (I - n n^T)`, `int f dn = 1`; exact `div S = -sigma f(r) n / r`, exact reading `sigma/R`.

| R/dx | dp/(sigma/R) | error [%] | momentum residual |
|---|---|---|---|
| 16 | 0.9730 | -2.70 | 5.7e-17 |
| 32 | 0.9737 | -2.63 | 1.9e-16 |
| 64 | 0.9739 | -2.61 | 1.7e-16 |

## T2. Laplace ladder -- every configuration

`flux_water` = normal integral over the water disc only; `flux_all` = over both phases (the Laplace jump).  `stress` = `sigma int|grad c| dn / R`.

| R/dx | mode | assembly | bracket | hourglass | stress err [%] | flux_water err [%] | flux_all err [%] | momentum residual |
|---|---|---|---|---|---|---|---|---|
| 16 | single_phase | consistent | mixture | on | -3.86 | -0.89 | -0.89 | 7.8e-01 |
| 16 | single_phase | consistent | symmetric | on | -3.86 | -0.89 | -0.89 | 7.8e-01 |
| 16 | single_phase | consistent | symmetric | off | -3.86 | -29.74 | -29.74 | 3.2e-17 |
| 16 | single_phase | upstream | mixture | on | -3.86 | -96.13 | -96.13 | 7.8e-01 |
| 16 | single_phase | upstream | symmetric | on | -3.86 | -96.13 | -96.13 | 7.8e-01 |
| 16 | two_phase | consistent | mixture | on | -3.95 | -16.50 | -29.66 | 7.3e-01 |
| 16 | two_phase | consistent | symmetric | on | -3.95 | -16.52 | -4.70 | 7.3e-01 |
| 16 | two_phase | consistent | symmetric | off | -3.95 | -29.79 | -4.69 | 4.3e-17 |
| 16 | two_phase | upstream | mixture | on | -3.95 | -96.74 | -96.74 | 7.7e-01 |
| 16 | two_phase | upstream | symmetric | on | -3.95 | -96.74 | -96.74 | 7.7e-01 |
| 32 | single_phase | consistent | mixture | on | -2.13 | -375.10 | -375.10 | 7.5e-01 |
| 32 | single_phase | consistent | symmetric | on | -2.13 | -375.10 | -375.10 | 7.5e-01 |
| 32 | single_phase | consistent | symmetric | off | -2.13 | -30.01 | -30.01 | 5.1e-17 |
| 32 | single_phase | upstream | mixture | on | -2.13 | -102.69 | -102.69 | 7.5e-01 |
| 32 | single_phase | upstream | symmetric | on | -2.13 | -102.69 | -102.69 | 7.5e-01 |
| 32 | two_phase | consistent | mixture | on | -2.22 | -2.66 | -29.94 | 7.3e-01 |
| 32 | two_phase | consistent | symmetric | on | -2.22 | -2.68 | -5.82 | 7.3e-01 |
| 32 | two_phase | consistent | symmetric | off | -2.22 | -30.05 | -5.80 | 2.9e-17 |
| 32 | two_phase | upstream | mixture | on | -2.22 | -99.05 | -99.05 | 7.4e-01 |
| 32 | two_phase | upstream | symmetric | on | -2.22 | -99.05 | -99.05 | 7.4e-01 |
| 64 | single_phase | consistent | mixture | on | -1.55 | +215.04 | +215.04 | 7.6e-01 |
| 64 | single_phase | consistent | symmetric | on | -1.55 | +215.04 | +215.04 | 7.6e-01 |
| 64 | single_phase | consistent | symmetric | off | -1.55 | -29.88 | -29.88 | 2.2e-17 |
| 64 | single_phase | upstream | mixture | on | -1.55 | -99.23 | -99.23 | 7.6e-01 |
| 64 | single_phase | upstream | symmetric | on | -1.55 | -99.23 | -99.23 | 7.6e-01 |
| 64 | two_phase | consistent | mixture | on | -1.65 | +25.55 | -29.80 | 7.3e-01 |
| 64 | two_phase | consistent | symmetric | on | -1.65 | +25.53 | -6.22 | 7.3e-01 |
| 64 | two_phase | consistent | symmetric | off | -1.65 | -29.93 | -6.21 | 6.5e-18 |
| 64 | two_phase | upstream | mixture | on | -1.65 | -99.69 | -99.69 | 7.3e-01 |
| 64 | two_phase | upstream | symmetric | on | -1.65 | -99.69 | -99.69 | 7.3e-01 |

## T3. Normalisation identity (`upstream` vs `consistent`)

| mode | consistent/upstream | particle mass m_w | relative deviation |
|---|---|---|---|
| two_phase | 21.56807522 | 0.03906250 | 5.5e+02 |
| single_phase | 25.60000000 | 0.03906250 | 6.5e+02 |

## T4. Mixture density-ratio sweep (contact fraction `f_k`, R/dx = 16)

| rho_air/rho_water | f_w | int|grad c| dn | stress err [%] | flux_all err [%] |
|---|---|---|---|---|
| 1 | 0.5000 | 0.4807 | -51.93 | -11.86 |
| 0.1 | 0.9091 | 0.8740 | -12.60 | -6.16 |
| 0.01 | 0.9901 | 0.9519 | -4.81 | -4.85 |
| 0.001 | 0.9990 | 0.9605 | -3.95 | -4.70 |

## T5. Machine gate

- criterion: |error| < 15.0 % vs `sigma/R`
- model: two_phase, hourglass off (momentum-exact), delta=sphinxys; assembly: consistent (V_j/rho_i); bracket: symmetric (S_i + S_j)
- reading: sum over both phases of rho (-a.n) A / (2 pi R)
- ladder: R/dx=16 -> -4.69 %, R/dx=32 -> -5.80 %, R/dx=64 -> -6.21 %
- worst |error| = **6.21 %** at R/dx = 64
- stress-reading worst |error| = 3.95 %
- **verdict: pass**

