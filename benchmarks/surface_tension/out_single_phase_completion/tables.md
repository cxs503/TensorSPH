# Single-phase interface-completion closure: tables (generated)

Generated from `out_single_phase_completion/gate.json` by `benchmarks/surface_tension/emit_single_phase_tables.py` -- do not edit by hand.

## T1

| R/Δx | particles | Δp [Pa] | σ/R [Pa] | error | within 15 % ? |
|---|---|---|---|---|---|
| 16 | 797 | 0.50585 | 0.72000 | -29.74 % | no |
| 32 | 3209 | 0.50396 | 0.72000 | -30.01 % | no |
| 64 | 12853 | 0.50485 | 0.72000 | -29.88 % | no |

## T2

| α | Δp [Pa] | error vs σ/R |
|---|---|---|
| 0.00 | 0.50396 | -30.01 % |
| 0.10 | 0.55463 | -22.97 % |
| 0.20 | 0.60531 | -15.93 % |
| 0.25 | 0.63064 | -12.41 % |
| 0.30 | 0.65598 | -8.89 % |
| 0.35 | 0.68132 | -5.37 % |
| 0.40 | 0.70666 | -1.85 % |
| 0.50 | 0.75733 | +5.18 % |
| 0.60 | 0.80800 | +12.22 % |
| 0.75 | 0.88402 | +22.78 % |
| 1.00 | 1.01070 | +40.38 % |

## T3

Fitted at the **middle rung** only (R/Δx = 32), by linear interpolation of the zero crossing inside [0.40, 0.50]: **α\* = 0.4263**.

| R/Δx | Δp [Pa] | error | within 15 % ? | momentum residual |
|---|---|---|---|---|
| 16 | 0.72527 | +0.731 % | yes | 8.4e-17 |
| 32 | 0.72000 | +0.000 % | yes | 2.9e-17 |
| 64 | 0.71868 | -0.184 % | yes | 3.4e-17 |

## T4

| h/Δx | error at α\* | within 15 % ? |
|---|---|---|
| 1.1 | -1.01 % | yes |
| 1.2 | -2.63 % | yes |
| 1.3 | +0.00 % | yes |
| 1.4 | +3.90 % | yes |
| 1.5 | +5.84 % | yes |

## T5

| control | value |
|---|---|
| hourglass off, α\* | error +0.00 %, momentum 2.9e-17 |
| hourglass on, α\* | error -89.07 %, momentum 8.8e-18 |
| momentum residual, α = 0 | 5.1e-17 |
| momentum residual, α = α\* | 2.9e-17 |
| two-phase gate (unchanged, closure off) | R/Δx=16: -4.69 %, R/Δx=32: -5.80 %, R/Δx=64: -6.21 % |

## T6

| hypothesis | test | result |
|---|---|---|
| full mirror (α = 1) is the right closure | Laplace error at R/Δx = 32 | +40.38 % — over-restores, rejected |
| α\* = mass fraction of the delta living beyond the free surface | air/water delta mass in the two-phase run | 0.0010 (6.28e-04 vs 6.14e-01) — two orders below α\* = 0.426, rejected |

## T7

| configuration | Δp error vs σ/R | momentum residual |
|---|---|---|
| single phase, penalty **on**, α = 0, R/Δx = 16 | -44.98 % | 1.9e-17 |
| single phase, penalty **on**, α = 0, R/Δx = 32 | -119.07 % | 9.2e-18 |
| single phase, penalty **on**, α = 0, R/Δx = 64 | -573.92 % | 2.4e-18 |
| two phase, penalty **off**, both-phase reading, R/Δx = 32 | -5.80 % | 2.9e-17 |
| two phase, penalty **on**, both-phase reading, R/Δx = 32 | -12.75 % | 8.0e-18 |
| 3-particle generic pair state, penalty **on** | — | 2.6e-17 |
| removed (pre-round-3) pair order, reconstructed from the current API | — | **0.7534** |
| audit re-implementation vs the model, 5-particle star | — | rel. diff 4.6e-17, net 2.9e-18 |
