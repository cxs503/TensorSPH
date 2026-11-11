# WCSPH dam-break + tank-sloshing verification chain

**Generated file** -- every number below is injected from `artifacts/wcsph_chain/report.json` by `benchmarks/wcsph_chain/make_doc.py`.
Do not edit by hand; rerun the generator instead.

* provenance: `agent/m11182` @ `43877a6554c6969736735049611182fa3b773807`, torch 2.7.1, python 3.11.6, float64, generated 2026-11-10T23:19:10Z

## 1. What this chain measures

A new weakly-compressible SPH solver (`src/tensorsph/wcsph/`) is verified bottom-up: a two-particle collision (the discrete invariants), a 2-D dam break against the Martin & Moyce (1952) experiment, the same column in 3-D, a forced rectangular tank against the linear potential-flow solution, and a resolution ladder for the observed convergence order.

## 2. Settings

| rung | setting | value |
|---|---|---|
| pair | v0 / c0 / alpha | 0.5 m/s / 20.0 m/s / 0.0 |
| pair | steps / dx | 2000 / 0.01 m |
| dam 2-D dx=0.04 | c0 / alpha / t_end | 25.0 m/s / 0.1 / 0.5 s |
| dam 2-D dx=0.02 | c0 / alpha / t_end | 25.0 m/s / 0.1 / 0.5 s |

## 3. Results

| criterion | measured | limit | verdict |
|---|---|---|---|
| a-momentum | 0.000e+00 | 1e-10 (<=) | **pass** |
| a-energy | 2.488e-03 | 0.01 (<=) | **pass** |
| b-2d-front | 0.1398 | 0.1 (<=) | **fail** |
| b-2d-front-mae | 0.1466 | 0.1 (<=) | **fail** |
| c-3d-front | n/a | 0.15 (<=) | **withheld** |
| f-test-count | 61 | 20 (>=) | **pass** |
| f-coverage | 0.8522 | 0.85 (>=) | **pass** |

Overall: **fail** (4 pass / 2 fail / 1 withheld).

### 3.1 Dam-break front vs experiment

| t [s] | τ | Z/a (SPH) | Z/a (exp) | rel. error |
|---|---|---|---|---|
| 0.1 | 0.990 | 0.352 | 0.320 | +9.9% |
| 0.2 | 1.981 | 1.212 | 1.341 | -9.6% |
| 0.3 | 2.971 | 2.313 | 2.689 | -14.0% |

Whole-curve mean absolute error in `Z/a`: **0.147**.

## 4. Figures

![Leading edge in the experiment's non-dimensional form.](figures/front_trajectory.png)

*Leading edge in the experiment's non-dimensional form.*

![Free-surface profile at the last archived instant.](figures/dam_break_profile.png)

*Free-surface profile at the last archived instant.*

![Particle-pair energy error with alpha = 0.](figures/pair_conservation.png)

*Particle-pair energy error with alpha = 0.*

## 5. Limitations

* The sloshing reference is linear potential flow, not an experiment: no digitised tank-pressure experiment is bundled in the repository, so the experimental gate is reported as withheld, never as a pass.
* The dam-break reference is a transcription of the 1952 figure via Lethe; its own uncertainty is a few percent of `a`, and the transcription has not been checked against the paywalled original.
* Near the first sloshing resonance the linear reference is only formally valid (zero damping, unbounded growth); the comparison is quoted only with that caveat.
