# CASE-POISEUILLE-SPH — design and predeclared gates

Case-pool entry **算例池·泊肃叶流** (anchor `CASE-POISEUILLE-SPH`,
`feature_ledger` acceptance: *analytic reference, tolerance < 5 %*).

Written **before** the final resolution ladder was run.  One feasibility probe
(`run.py --quick`: one resolution, `H/dx = 8`, one settle period, one
measurement period) was executed first to verify that the mechanism produces a
signal at all; it is reported in the verification note as a probe and is *not*
part of the graded ladder.  No threshold below was moved after the ladder.

## 1. What the case asks

Plane channel `0 <= y <= H`, no-slip at both walls, periodic in `x`, driven by a
uniform body force per unit mass `g(t)`:

```
u_t = nu u_yy + g(t),      u(0,t) = u(H,t) = 0,      u(y,0) = 0
```

Three regimes of the same equation, one case:

| regime | forcing | analytic reference |
|---|---|---|
| R1 constant | `g = g0` | `u = g0 (H-y) y / (2 nu)` (steady parabola); start-up: Boussinesq series |
| R2 harmonic | `g = g0 cos(omega t)` | Womersley periodic state `Re{Uhat e^{i omega t}}`, `Uhat = (g0/i omega)[1 - cosh(alpha(y-H/2))/cosh(alpha H/2)]` |

R2 is the part no other benchmark in the repository covers: the existing
verification set measures either a *mean* momentum balance (steady channel,
start-up) or a *wall-driven* oscillation (Stokes' second problem).  The
frequency response of a pressure-driven channel — amplitude **and** phase of
every row — is the pulsatile (Womersley) problem.

## 2. Parameters (fixed before the ladder)

| quantity | value | note |
|---|---|---|
| `H` | 0.1 m | channel height |
| `nu` | 0.01 m²/s | `Re = u_max H / nu = 12.3` at the steady maximum |
| `rho0` | 1000 kg/m³ | Tait EOS, `c_s = 20 m/s` |
| `g0` | 9.81 m/s² | same driving as `benchmarks/poiseuille_mirror` |
| `omega` (R2 headline) | 13.0 rad/s | `Wo = 1.80`, `delta/H = 0.39`, `T = 0.483 s` |
| `omega_fast` (G6 only) | 31.4 rad/s | `Wo = 2.80`, `delta/H = 0.25`, `T = 0.200 s` |
| columns | 8 | `x` is periodic and the flow `x`-invariant; 8 columns keep the run cheap |
| resolutions | `H/dx = 8, 12, 16` | `delta/dx = 3.1, 4.7, 6.3` |
| settle / measurement window | 2 periods / 2 periods | transient e-folding time `H²/(nu pi²) = 0.101 s = 0.21 T` |
| samples per period | 64 | harmonic fit with an integer number of periods |

Two frequencies are used on purpose: at `Wo = 1.8` and `Wo = 2.8` the analytic
centre-line amplitude ratio `A(omega_fast)/A(omega)` is **0.39**, whereas a
solver that ignores unsteadiness (quasi-steady, `omega`-independent response)
would return **1.00**.  That is the falsification lever of gate G6.

## 3. Predeclared gates

Amplitude gates are relative; the phase natural scale is a quarter turn
(`pi/2`), so "5 % phase" means `0.05 * pi/2 = 4.5 deg`.  Phase differences are
taken modulo `2 pi` before the tolerance is applied.

| gate | metric | threshold |
|---|---|---|
| **G1** | centre-line amplitude vs `|Uhat(H/2)|`, finest resolution | ≤ 5 % |
| **G2** | RMS of `(A_sph - A_analytic)/A_analytic` over rows `0.15 ≤ y/H ≤ 0.85` | ≤ 5 % |
| **G3** | centre-line phase vs `arg Uhat(H/2)` | ≤ 5 % of `pi/2` (4.5°) |
| **G4** | RMS of the phase error over the same rows | ≤ 5 % of `pi/2` |
| **G5** | periodic state reached: two consecutive measurement periods give the same centre-line amplitude (≤5 %) and phase (≤4.5°) | pass |
| **G6** | frequency discrimination: measured `A(omega_fast)/A(omega)` vs analytic ratio | ≤ 5 % |
| **G7** | R1 anchor: steady-profile RMS error over the same interior band | ≤ 5 % |
| **G8** | R1 anchor: start-up profile RMS vs Boussinesq series at `t = 0.40 s` | ≤ 5 % |
| **G7b** | R1 ladder: steady-profile RMS at *every* rung (8/12/16) | ≤ 5 % |
| **G8b** | R1 ladder: start-up profile RMS at *every* rung | ≤ 5 % |
| **G9** | reference self-check: independent Crank–Nicolson FD solution vs the closed form (amplitude RMS and phase RMS) | ≤ 1 % |

G1–G5 are graded at the **finest** resolution of the ladder (`H/dx = 16`); the
whole ladder is reported so the convergence can be read (gate E of
`benchmarks/poiseuille_mirror` does the same for the steady case).  G7–G9 are
graded at `H/dx = 12`.

## 4. Measurement protocol

* Row means of `u_x` on the lattice rows (the `x`-scatter inside a row is
  recorded as the 2-D check).
* At each sample time `u_x(y_k, t)` is projected on `cos(omega t)` and
  `sin(omega t)` (`tensorsph.stokes_second.harmonic_fit`): the fit returns
  `A = hypot(a, b)` and `phase = -atan2(b, a)` for `u = A cos(omega t + phase)`,
  the same convention as `tensorsph.womersley.phase`.
* The fit residual is recorded per row; a residual above a few percent means
  the row is not single-harmonic (acoustic or particle noise) and would
  invalidate the amplitude before any comparison is made.

## 5. What would count as failure (stated in advance)

* Any of G1–G8 failing at the finest resolution → the case is reported as
  **honest FAIL** with the measured numbers, not retuned.  The gradient
  hypothesis is that WCSPH's numerical dissipation acts as an extra viscosity,
  which would show up as an amplitude deficit growing with `Wo` (i.e. G6
  failing while G1 passes) — that would be a finding about the scheme, not a
  case failure to be papered over.
* G9 failing → the reference is wrong and the case is void until fixed; this is
  the gate that keeps a reference bug from being laundered as physics error.

## 6. Files

| file | purpose |
|---|---|
| `src/tensorsph/womersley.py` | library reference: Womersley closed form, from-rest series, sympy/numeric self-checks |
| `benchmarks/case_poiseuille/ref.py` | independent Crank–Nicolson FD reference (no shared code with SPH) |
| `benchmarks/case_poiseuille/run.py` | driver, gates, `out/result.json` |
| `benchmarks/case_poiseuille/make_doc.py` | figures + verification note generator |
| `docs/verification_case_poiseuille.md` | generated verification note |
| `tests/test_case_poiseuille.py` | reference/estimator unit tests + one SPH smoke run |
## 7. Addendum (post-hoc, m22483): oscillatory wall shear as a graded quantity

Sections 1–6 were written before the ladder and are unchanged.  The gates
G1–G9 grade the *kinematics* of the Womersley state (amplitude and phase of
every row).  They do not grade the **oscillatory wall shear** — the pulsatile
drag — which is the engineering output of a pulsatile duct.

This addendum was written *after* the ladder, and is labelled as such rather
than back-dated.  Declaration discipline: the thresholds are copied from this
document (§3: 5 % amplitude, 5 % of a quarter turn = 4.5 deg phase) and were
not moved to fit the numbers; an exploratory probe on the committed artifact
came first and is recorded in
`docs/case_poiseuille_wall_shear_harmonic.md` §1.  The layer is post-processing
of the *existing* `out/result.json` — no SPH run, no solver parameter, and
therefore no way for it to influence the graded ladder.

| gate | metric | threshold |
|---|---|---|
| **W0 / W0b** | cross-section mean velocity `\|\<u\>_hat\|` / `arg \<u\>_hat`, finest rung | 5 % / 4.5 deg |
| **W1 / W2** | oscillatory wall shear `\|\tau_hat\|` / `arg \tau_hat`, finest rung | 5 % / 4.5 deg |
| **W3** | W1/W2 at *every* rung (8/12/16) | same |
| **W4** | two-frequency drag ratio `\|\tau_fast\|/\|\tau_slow\|` and phase difference, `H/dx = 12` | 5 % / 4.5 deg |
| **W5** | control: the interior quadratic estimator of the steady case, applied to the *analytic* harmonic profile, must be biased by > 5 % | pass |
| **W6** | reference self-check: budget closed form == `mu alpha (g_hat/i omega) tanh(alpha H/2)`, and the `omega -> 0` limit == `rho g H / 2` | 1e-10 |

Estimator: `tau_hat = rho H (g_hat - i omega \<u\>_hat)/2`, the exact momentum
budget of the periodic state, with `\<u\>_hat` the cross-section mean of the
committed per-row harmonic fits.  Two facts make this more than a restatement
of G1–G9:

* the steady wall-shear estimator validated in
  `docs/verification_poiseuille_wall_shear_estimator.md` (interior quadratic
  fit) is biased by 5.3 % / -23 deg at `Wo = 1.8` on the analytic harmonic
  profile, resolution-independently — it is exact only because the steady
  profile is a parabola.  Hence W5 is a control, not a gate on the physics;
* a quasi-steady channel (steady parabola following `g(t)`) passes a mean
  momentum balance and is rejected here at +133 % amplitude, so W1–W4 test the
  unsteady Stokes layer.

Implementation: `benchmarks/case_poiseuille/wall_shear_harmonic.py`,
`benchmarks/case_poiseuille/make_wall_shear_harmonic_doc.py`,
`benchmarks/case_poiseuille/out/wall_shear_harmonic.json`,
`docs/case_poiseuille_wall_shear_harmonic.md`,
`docs/figures/case_poiseuille_wall_shear_harmonic.png`,
`tests/test_case_poiseuille_wall_shear_harmonic.py`.
