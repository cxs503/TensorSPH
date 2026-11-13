# TensorSPH #599 — Mirror (Ghost) Wall Poiseuille Verification — Deliverable

## Mission
Re-open PR#599 (originally closed 2026-11-05) with the matured m15408 deliverables:
plane Poiseuille channel verification using a mirror/ghost no-slip wall boundary
condition, verified against the closed-form analytic solution within 5% on all
predeclared gates.

## Files Delivered
| File | Lines | Purpose |
|---|---|---|
| `src/tensorsph/wall_mirror.py` | 100 | Mirror (ghost) particle BC for planar walls |
| `benchmarks/poiseuille_mirror/run.py` | 424 | Main benchmark driver (ladder + control) |
| `benchmarks/poiseuille_mirror/run_sensitivity.py` | 85 | One-at-a-time sensitivity runs |
| `benchmarks/poiseuille_mirror/make_doc.py` | 434 | Figure + document generator |
| `benchmarks/poiseuille_mirror/doc_template.md` | 132 | Verification note template |
| `benchmarks/poiseuille_mirror/out/results.json` | 3250 | Full ladder + control artifacts |
| `benchmarks/poiseuille_mirror/out/sensitivity.json` | 3185 | Sensitivity run artifacts |
| `benchmarks/poiseuille_mirror/out/sensitivity_g_x_x2_long.json` | 27 | Extended g_x×2 attribution run |
| `docs/figures/poiseuille_mirror_profile.png` | — | Measured vs analytic profiles |
| `docs/figures/poiseuille_mirror_convergence.png` | — | Error vs resolution |
| `docs/verification_poiseuille_mirror_wall.md` | 189 | Full verification note |
| `tests/test_poiseuille_mirror_wall.py` | 156 | Unit + integration tests |

## Physics & Reference
Steady plane Poiseuille flow, walls at y = 0, H, no-slip, uniform body force g_x:
- Analytic: u(y) = g_x(H-y)y/(2ν), u_max = g_x·H²/(8ν), τ_w = ρ·g_x·H/2
- Reference: White, Viscous Fluid Flow 3rd ed. §3-2.2; Batchelor §4.2
- Symbolic re-derivation via sympy recorded in results.json `reference_provenance`

## Method
Mirror/ghost particles imposed at each wall plane:
- Ghost carries negated velocity (no_slip) → mean velocity on plane = 0
- Handed to `rhs` as moving wall particles via `wall=` and `wall_vel=`
- Same viscous operator as the solver; only the BC changes

## Predeclared Gates (all ≤ 5%)
| Gate | Metric | H/dx=8 | H/dx=12 | H/dx=16 |
|---|---|---|---|---|
| A | Centre-line error | +0.38% ✓ | +0.02% ✓ | -0.25% ✓ |
| B | Profile RMS (interior) | 2.72% ✓ | 1.23% ✓ | 0.42% ✓ |
| C | Wall shear error (bottom/top) | -0.35%/-0.35% ✓ | -0.38%/-0.38% ✓ | -0.44%/-0.44% ✓ |
| D | Wall slip (parabola intercept) | 2.30% ✓ | 1.04% ✓ | 0.63% ✓ |
| E | Convergence (RMS non-increasing) | ✓ | ✓ | ✓ |
| F | Free-slip control (no steady state) | u_max ratio 7.2x ✓ | — | — |

## Test Results
```
pytest tests/test_poiseuille_mirror_wall.py -v → 10 passed in 51s
pytest tests/test_poiseuille_mirror_recheck.py -v → 8 passed in 6s
pytest tests/test_poiseuille*.py -v → 66 passed in 100s
```

## Key Findings
1. **Mirror wall closes the momentum budget**: wall shear error < 0.5% at all resolutions,
   vs ~50% deficit with the old blend wall model.
2. **Second-order convergence**: profile RMS falls by 6.4× when H/dx doubles (observed
   order 2.68), confirming the BC is applied *on* the plane, not a spacing inside.
3. **Free-slip control validates the mechanism**: with tangential shear removed, the flow
   accelerates rigid-body (u = gt), proving the no-slip result is not a tautology.
4. **Symbolic provenance**: the closed-form solution is re-derived via sympy inside the
   benchmark; a transcription error would show as non-zero ODE residual.

## Reproduction
```bash
# Full ladder
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/run.py \
    --resolutions 8 12 16 --t-end 1.0 --t-average 0.2

# Sensitivity
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/run_sensitivity.py

# Generate figures + doc
PYTHONPATH=src:. python benchmarks/poiseuille_mirror/make_doc.py

# Tests
PYTHONPATH=src:. python -m pytest tests/test_poiseuille_mirror_wall.py -v
PYTHONPATH=src:. python -m pytest tests/test_poiseuille_mirror_recheck.py -v
```

## Artifacts
- `benchmarks/poiseuille_mirror/out/results.json` — full ladder, control, per-row profiles
- `benchmarks/poiseuille_mirror/out/sensitivity.json` — one-at-a-time variations
- `docs/figures/poiseuille_mirror_profile.png` — measured vs analytic
- `docs/figures/poiseuille_mirror_convergence.png` — error vs resolution
- `docs/verification_poiseuille_mirror_wall.md` — complete verification note
