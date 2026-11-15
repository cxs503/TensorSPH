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
| `steady_window.py` | is the committed steady window stationary vs. residual fill? (gates S1-S4, needs one integration) |
| `steady_drift.py` | is the late decline of the steady functional signal or noise? block bootstrap + AR(1) t, gates D1-D4; `--extend` adds a window at t ~ 3.5 s |
| `out/steady_drift.json` | baseline/window fits, null distribution, sensitivity to block length, gates D1-D4 |
| `../../docs/case_poiseuille_steady_drift.md` | generated note: trend-vs-noise verdict and the acceptance bound it implies |
| `../../tests/test_case_poiseuille_steady_drift.py` | estimator/bootstap tests on synthetic series + artifact and note consistency |
| `wall_shear_harmonic.py` | oscillatory wall shear (pulsatile drag) from the momentum budget, gates W0-W6; post-processes `out/result.json`, no solver run |
| `out/wall_shear_harmonic.json` | `tau_hat` per rung at both frequencies, estimator controls, gates W0-W6 |
| `make_wall_shear_harmonic_doc.py` | figure + `../../docs/case_poiseuille_wall_shear_harmonic.md` generator |
| `../../tests/test_case_poiseuille_wall_shear_harmonic.py` | closed-form/estimator/control tests + artifact and note binding |

## Run

```bash
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py --quick      # smoke, ~1 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/run.py              # ladder 8/12/16, ~9 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_doc.py         # figures + note
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/uncertainty.py      # GCI/V&V-20 layer, ~2 s
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/steady_drift.py     # trend-vs-noise audit, ~1 s (post-process)
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/steady_drift.py --extend  # + window at t ~ 3.5 s, ~4 min
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/wall_shear_harmonic.py  # pulsatile drag, ~1 s (post-process)
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_wall_shear_harmonic_doc.py
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

The ladder also *found* something, and then closed it. `U3b` first **failed**
for the steady functional: `U_input` was the half range of the in-run samples
divided by their *mean*, i.e. by the very functional that decays to zero under
refinement, so the estimator was anti-convergent and mixed units with `E`
(percent of scale). That estimator is kept in the artifact for audit
(`steady_half_range_percent_of_mean`) but no longer grades anything; the
published band uses the same percent-of-scale normalisation as `E`, which keeps
the *conservative* single-sample value rather than the smaller between-window
one. Two independent artifacts now back the steady anchor:

* `steady_window.py` → `docs/case_poiseuille_steady_window.md`: re-integrating to
  t = 1.8 s reproduces the committed window (gate `S1`), the window level is
  stable (`S2`), the surviving Boussinesq fill cannot explain its scatter (`S3`)
  and the between-window band is inside 5 % (`S4`);
* `steady_drift.py` → `docs/case_poiseuille_steady_drift.md`: the level test
  above cannot see a trend *inside* a window, and the committed series has one.
  A block bootstrap of the pre-window samples asks whether that late decline is
  distinguishable from the noise (`D1`/`D2`), adds the drift to the committed
  error (`D3`) and, with `--extend`, grades a third window at t ≈ 3.5 s (`D4`
  "the decay keeps going", which is what turns the committed value into an
  upper bound).

With that, all nine uncertainty gates (`U0`–`U4`, `U1b/c`, `U2b`, `U3b`) PASS and
the start-up functional passes its own three (`U2b`/`U3b`) as well. Honest
residual: `U_input` is the between-sample repeatability *inside* one run, not a
run-to-run ensemble, so it captures the sampling window, not reproducibility
across machines.

Two pitfalls are documented and guarded by tests: the analytic reference must be
evaluated per rung (row centres move with `dx`, so one reference value invents an
order), and the profile norm must be taken on a common grid with a 5-point
Lagrange stencil (linear interpolation floors the norm error at ~0.11 %, where
the functional stops measuring the solver).

### Oscillatory wall shear (pulsatile drag)

The graded ladder certifies the *kinematics* of the Womersley state (amplitude and
phase of every row). `wall_shear_harmonic.py` adds the quantity a duct designer
asks for — the complex wall shear `tau_hat` — from the momentum budget

```
tau_hat = rho H ( g_hat - i omega <u>_hat ) / 2
```

which is exact for the periodic state, contains **no kernel and no wall normal**,
and needs only the cross-section mean velocity already committed in
`out/result.json`. Headline, finest rung (`H/dx = 16`, `Wo = 1.8`):

| quantity | measured | analytic | error |
|---|---|---|---|
| `|tau_hat|` | 309.3794 Pa | 309.7224 Pa | -0.111 % |
| `arg tau_hat` | -40.358 deg | -39.986 deg | -0.372 deg (0.41 % of pi/2) |
| `|tau_fast|/|tau_slow|` (`H/dx = 12`) | 0.5732 | 0.5801 | -1.183 % |

Two pitfalls are guarded by tests, because both are silent otherwise:

* the steady wall-shear estimator of `combined.py` (interior quadratic fit) is
  **not transferable** to the harmonic regime: on the analytic Womersley profile
  it is biased by 5.3 % in amplitude and -23 deg in phase at `Wo = 1.8`, and the
  bias is resolution-independent (0.07 % spread over `H/dx = 8..16`) — model
  bias, not discretisation. It is kept as the *control* of gate `W5`;
* a quasi-steady channel (steady parabola following `g(t)`) passes a *mean*
  momentum balance but is rejected here at +133 % in amplitude, which is what
  makes `W1`-`W4` a test of the unsteady Stokes layer rather than of mass
  conservation.

See `../../docs/case_poiseuille_wall_shear_harmonic.md`; gates `W0`-`W6` are
declared in the module docstring (post-hoc extension of `DESIGN.md`, thresholds
are the case's own 5 % convention).

## Limits

Laminar, `Wo` up to ~3, Stokes layer 3–7 particle spacings, one channel height,
weakly compressible WCSPH (`Ma < 0.08`). Not a turbulent or `Wo >> 10` tool.