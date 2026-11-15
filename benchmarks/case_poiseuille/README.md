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
| `uncertainty.py` | discretization-uncertainty + validation layer (GCI, ASME V&V 20); post-processes `out/result.json`, no solver run |
| `out/uncertainty.json` | observed orders, GCI bands, `E` vs `U_V`, repeatability floor, gates U0-U3 |
| `../../docs/case_poiseuille_uncertainty.md` | generated note: is the residual discretization error or a floor, and does the band certify 5 %? |

## Run

```bash
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py --quick      # smoke, ~1 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py              # ladder 8/12/16, ~9 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_doc.py         # figures + note
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/uncertainty.py      # GCI/V&V-20 layer, ~2 s
```

Exit status is non-zero when a gate fails; `--quick` grades one resolution and
leaves G5/G6 as *not graded* rather than passing them.

## Result (summary)

See `docs/verification_case_poiseuille.md` for the full table. The headline
numbers come from `out/result.json`; nothing in the note is hand-copied.

## Numerical uncertainty (what the ladder can and cannot certify)

`uncertainty.py` grades the same committed ladder against Roache's GCI and the
ASME V&V 20-2009 comparison, asks whether the residual is discretization error
or a measurement floor, and whether the uncertainty band is narrow enough to
certify the 5 % acceptance at all (gate U3, stricter than "the point error
happens to be under 5 %"). Headline, finest rung `H/dx = 16` (numbers read from `out/uncertainty.json`):

| functional | E % | U_num % (GCI) | order status |
|---|---|---|---|
| centre amplitude | 0.074 | 0.43 | pair orders disagree (1.9 vs 4.2) -> no order claimed |
| interior amplitude L2 | 0.188 | 0.94 | pair orders 2.1 vs 3.0 -> no order claimed |
| centre phase | 0.129 | 0.62 | **oscillatory**: the error changes sign between 12 and 16 |
| interior phase L2 | 0.118 | 0.61 | pair orders 2.4 vs 3.0 -> no order claimed |

All four pass U1 (every rung, not just the finest, is inside 5 %), U2
(`E <= U_V`) and U3 (band < 5 %). No order of accuracy is quoted: three rungs
with unequal ratios (r = 1.5, 1.333) cannot pin the order down, and the phase
error changes sign.

### R1 ladder (steady parabola + Boussinesq start-up)

R1 used to carry a single committed resolution (`H/dx = 12`), so its two anchors
could only be compared with the tolerance and was reported as `insufficient`.
The driver now integrates R1 on the same three rungs (`--constant-resolutions`,
default 8/12/16; `--constant-only` merges the ladder into an existing artifact
without re-running the 9 min harmonic section), and `uncertainty.py` grades the
two R1 functionals with the same order/GCI/V&V machinery (gates `U1c`, `U2b`,
`U3b`, plus `G7b`/`G8b` = every rung inside tolerance). The functional is the
band-RMS of the relative deviation from the analytic profile — an *error* whose
truth is exactly zero — so the published per-cent is the same number gate G7/G8
grades; a test pins that identity to round-off. `U_input` for the start-up row is
declared `0` (a single instant has no repeatability) instead of being invented,
which makes that row's band optimistic — stated in the note.

The ladder also *found* something: `U3b` **FAILS** for the steady functional
(exact numbers in `docs/case_poiseuille_uncertainty.md`, read from
`out/uncertainty.json`). Its point error is well inside 5 % and every rung passes
(`U1c` PASS), but the band does not support the acceptance: `U_input` is the half
range of the in-run samples of the functional over the 0.80–1.00 s window, i.e.
the fill of the channel is still visible there, so the "steady" window is not
stationary at the 5 % level. No 5 % certification is claimed for the steady
anchor; closing it needs a longer integration (a genuinely stationary window) or
a sub-window repeatability estimator, which needs the per-sample profiles that
this artifact does not store. The start-up functional passes all three
(`U2b`/`U3b`).

Two pitfalls are documented and guarded by tests: the analytic reference must be
evaluated per rung (row centres move with `dx`, so one reference value invents an
order), and the profile norm must be taken on a common grid with a 5-point
Lagrange stencil (linear interpolation floors the norm error at ~0.11 %, where
the functional stops measuring the solver).

## Limits

Laminar, `Wo` up to ~3, Stokes layer 3–7 particle spacings, one channel height,
weakly compressible WCSPH (`Ma < 0.08`). Not a turbulent or `Wo >> 10` tool.