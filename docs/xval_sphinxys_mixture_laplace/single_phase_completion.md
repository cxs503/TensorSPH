# Single-phase free-surface surface tension: interface-completion closure

**Anchor** `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` · **Repository** TensorSPH
· **Task** m20035 · **Gate** |error| < 15 % vs the exact 2-D
Young-Laplace law · **Verdict** `pass`

Supersedes nothing: it closes the item that
[`README.md`](README.md) §5.2 left open ("the vacuum reduction of the mixture
model is not Laplacian-accurate with this force assembly; it needs the
both-phase shell").  The two-phase gate of that round is re-measured here and is
unchanged.

## 1. Case and parameters

| quantity | value |
|---|---|
| geometry | 2-D water disc, R = 0.1 m, square lattice, spacing Δx |
| phases | **single phase (vacuum reduction)** — this is the free-surface case |
| σ | 0.072 N/m (water–air, 20 °C) |
| ρ | 1000 kg/m³ |
| kernel | Wendland C2, support 2h, h/Δx = 1.3 (upstream default) |
| assembly | `consistent` (`V_j/ρ_i`), bracket `symmetric`, hourglass **off** |
| reference | Δp = σ/R = 0.7200 Pa (exact) |
| ladder | R/Δx = 16, 32, 64 |

The reference contains no discretisation error, so every percentage below is
model error.

## 2. Method

**The defect.** The one-sided (vacuum-reduced) stress flux
`Σ_{j present} V_j (S_i + S_j)·∇_i W_ij` does not represent a full kernel
support: a free-surface particle has no neighbour beyond the interface, so the
tangential part of the stress divergence is under-resolved.  Measured deficit
is −30 % of σ/R, resolution independent (T1).

**The closure.** `tensorsph.sphinxys_css.interface_completion` adds the specular
image of each present kernel gradient across the *local* interface tangent
plane, `∇W' = W'(r)(e − 2(e·n̂)n̂)`, applied with weight α:

`gw_eff = gw + α · gw'`, `flux_ij = (S_i + S_j)·gw_eff`, `n̂` = pair-averaged `∇c`
direction.  The mirrored neighbour is given the present neighbour's stress
(exact for a plane-symmetric `S = σ(I − n̂n̂ᵀ)δ(dist)`).  Because the flux stays a
*symmetric two-point* function of the pair, momentum conservation is untouched
(T5).  α = 0 reproduces the round-2 configuration bit for bit; the default is
`completion="none"`, so nothing else in the library changes.

**Protocol (no leakage).** α is fitted **only at R/Δx = 32** (middle
rung), then applied unchanged to R/Δx = 16 and 64 and to a smoothing-ratio
sweep h/Δx ∈ [1.1, 1.5].

## 3. Results (programmatic, `out_single_phase_completion/gate.json`)

### T1 — baseline: the round-2 open item, reproduced

| R/Δx | particles | Δp [Pa] | σ/R [Pa] | error | within 15 % ? |
|---|---|---|---|---|---|
| 16 | 797 | 0.50585 | 0.72000 | -29.74 % | no |
| 32 | 3209 | 0.50396 | 0.72000 | -30.01 % | no |
| 64 | 12853 | 0.50485 | 0.72000 | -29.88 % | no |

### T2 — closure scan (fit rung R/Δx = 32)

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

### T3 — cross-resolution validation with the transferred α\*

Fitted at the **middle rung** only (R/Δx = 32), by linear interpolation of the zero crossing inside [0.40, 0.50]: **α\* = 0.4263**.

| R/Δx | Δp [Pa] | error | within 15 % ? | momentum residual |
|---|---|---|---|---|
| 16 | 0.72527 | +0.731 % | yes | 8.4e-17 |
| 32 | 0.72000 | +0.000 % | yes | 2.9e-17 |
| 64 | 0.71868 | -0.184 % | yes | 3.4e-17 |

### T4 — cross-parameter transfer (smoothing ratio, R/Δx = 32)

| h/Δx | error at α\* | within 15 % ? |
|---|---|---|
| 1.1 | -1.01 % | yes |
| 1.2 | -2.63 % | yes |
| 1.3 | +0.00 % | yes |
| 1.4 | +3.90 % | yes |
| 1.5 | +5.84 % | yes |

### T5 — controls: hourglass, momentum, two-phase regression

| control | value |
|---|---|
| hourglass off, α\* | error +0.00 %, momentum 2.9e-17 |
| hourglass on, α\* | error -345.10 %, momentum 7.5e-01 |
| momentum residual, α = 0 | 5.1e-17 |
| momentum residual, α = α\* | 2.9e-17 |
| two-phase gate (unchanged, closure off) | R/Δx=16: -4.69 %, R/Δx=32: -5.80 %, R/Δx=64: -6.21 % |

### T6 — two tempting explanations that fail

| hypothesis | test | result |
|---|---|---|
| full mirror (α = 1) is the right closure | Laplace error at R/Δx = 32 | +40.38 % — over-restores, rejected |
| α\* = mass fraction of the delta living beyond the free surface | air/water delta mass in the two-phase run | 0.0010 (6.28e-04 vs 6.14e-01) — two orders below α\* = 0.426, rejected |

Figure: `benchmarks/surface_tension/out_single_phase_completion/gate.png`
(panel a: closure scan with the gate band; panel b: baseline vs closure on the
ladder; panel c: smoothing-ratio transfer).

## 4. Error judgement

Worst |error| over the validation ladder with α = α\* is **0.731 %** at the resolutions tested (inside the ledger's 15 % criterion); the h/Δx transfer sweep stays at 5.84 %. Explicit answer to "is it < 3 %?": **yes** for the calibrated closure at R/Δx = 16/32/64, but the coefficient was fitted against this same exact reference at one rung, so the <3 % figure is a *calibration residual*, not an independent accuracy claim; the independent content is that the round-2 free-surface deficit of −30 % is a removable closure error, not a property of the CSS model.

| quantity | measured | criterion | result |
|---|---|---|---|
| Δp vs σ/R, α = 0 (round-2 config) | −29.7 … −30.0 % | < 15 % | FAIL (reproduced) |
| Δp vs σ/R, α = α\*, R/Δx = 16/32/64 | +0.73 %, +0.00 %, -0.18 % | < 15 % | **PASS** |
| Δp vs σ/R, α = α\*, h/Δx ∈ [1.1, 1.5] | -1.01 %, -2.63 %, +0.00 %, +3.90 %, +5.84 % | < 15 % | **PASS** |
| momentum conservation with closure on | 2.9e-17 | < 1e−12 | **PASS** |
| two-phase gate (round-2 configuration) | -4.69 %, -5.80 %, -6.21 % | < 15 % | **PASS, unchanged** |

## 5. Pitfalls, limits, honest caveats

1. **α is a calibration, not a derivation.** No attempt is made to claim α\* has
   a first-principles value; T6 documents two plausible derivations that are
   falsified (full mirror, and "α\* = missing delta mass fraction").  What is
   established is that the −30 % deficit is *removable* and that one constant
   transfers across three resolutions and five smoothing ratios — i.e. it is a
   closed model error, not a grid artefact.
2. **The <3 % figure at the fit rung is a calibration residual.** The
   scientifically independent numbers are the transferred ones (R/Δx = 16 and
   64 with α fixed at the R/Δx = 32 fit): +0.73 %
   and -0.18 %.
3. **Hourglass penalty must stay off on a free surface** — the upstream penalty
   is unbalanced without a second phase and diverges with resolution (round-2
   §5.2).  The closure does not change that: see T5.
4. **The pair-averaged normal is ill-defined in the deep bulk** (`∇c → 0`, the
   direction is numerical noise).  This is harmless because the flux is
   multiplied by `S ≈ 0` there, and
   `tests/test_single_phase_completion.py::test_closure_injects_no_bulk_force`
   locks it: more than one support width from the interface the closure adds
   exactly zero acceleration.
5. Scope: 2-D, static, inviscid, no gravity, no wall contact, single component.
   Curvature enters only as κΔx, so the R/Δx ladder covers it; a genuinely
   non-circular interface (fingering, break-up) is not tested here and the
   closure is expected to be less accurate where the interface curvature radius
   approaches the support width.
6. Dynamic consequences (capillary-wave dispersion, droplet oscillation
   frequency) are **not** measured in this task; only the static exact anchor.

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/surface_tension/single_phase_completion.py   # ~20 s, CPU
python runs/m20035/emit_tables.py                                            # JSON -> this doc
python -m pytest tests/test_single_phase_completion.py -q                     # 7 passed
```

## 7. Artifacts

| artifact | content |
|---|---|
| `src/tensorsph/sphinxys_css.py` | `interface_completion`, `COMPLETIONS`, `completion` / `completion_weight` on `inner_stress_force` and `surface_tension_acceleration` (all additive, default off) |
| `benchmarks/surface_tension/single_phase_completion.py` | ladder, closure scan, fit, transfer, hourglass, momentum, two-phase regression |
| `benchmarks/surface_tension/out_single_phase_completion/gate.json` | every number above |
| `benchmarks/surface_tension/out_single_phase_completion/gate.png` | 3-panel figure |
| `tests/test_single_phase_completion.py` | 7 tests: reflection identity, no bulk force, default off, baseline lock, gate, transfer, momentum |
| `runs/m20035/emit_tables.py`, `runs/m20035/tables.md` | JSON → tables (zero hand-copy) |
