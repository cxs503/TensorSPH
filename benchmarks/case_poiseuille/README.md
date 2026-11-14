# CASE-POISEUILLE-SPH — plane Poiseuille channel, three forcings

Case-pool entry **算例池·泊肃叶流**. One equation, one geometry, three forcings:

```
u_t = nu u_yy + g(t),   u(0,t) = u(H,t) = 0,   u(y,0) = 0
```

| regime | forcing | analytic reference |
|---|---|---|
| R1 constant | `g = g0` | steady parabola `g0 (H-y) y / (2 nu)`; start-up: Boussinesq series |
| R2 harmonic | `g = g0 cos(omega t)` | Womersley periodic state, `Wo = (H/2) sqrt(omega/nu)` |

R2 is the increment: the existing set measures a *mean* momentum balance (steady
channel, start-up) or a *wall-driven* oscillation (Stokes' second problem);
this entry grades the **frequency response** of a pressure-driven channel —
amplitude *and* phase of every row.

Tolerance (feature ledger): **< 5 %** against the analytic reference. Graded
gates G1–G9 and their thresholds are predeclared in `DESIGN.md`.

## Files

| file | purpose |
|---|---|
| `DESIGN.md` | the case, parameters and the predeclared gates |
| `run.py` | SPH driver + gates; writes `out/result.json` |
| `ref.py` | independent Crank–Nicolson finite-difference reference |
| `make_doc.py` | figures + `docs/verification_case_poiseuille.md` |
| `../../src/tensorsph/womersley.py` | library reference (closed form, from-rest series, sympy checks) |
| `../../tests/test_case_poiseuille.py` | reference/estimator tests + SPH smoke run |

## Run

```bash
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py --quick      # smoke, ~1 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py              # ladder 8/12/16, ~9 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_doc.py         # figures + note
```

Exit status is non-zero when a gate fails; `--quick` grades one resolution and
leaves G5/G6 as *not graded* rather than passing them.

## Result (summary)

See `docs/verification_case_poiseuille.md` for the full table. The headline
numbers come from `out/result.json`; nothing in the note is hand-copied.

## Limits

Laminar, `Wo` up to ~3, Stokes layer 3–7 particle spacings, one channel height,
weakly compressible WCSPH (`Ma < 0.08`). Not a turbulent or `Wo >> 10` tool.