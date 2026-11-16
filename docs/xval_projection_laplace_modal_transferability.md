# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE, round 12: does the well-balanced split keep the shape dynamics?

*Anchor:* `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` (TensorSPH, acceptance `< 15 %` on the Laplace
jump). *What this round adds:* an analytic-frequency test of the round-3 device on the branch its
own page excluded (`xval_projection_laplace_wellbalanced.md` §5: "Neither capillary waves nor
droplet oscillation are covered"), and the measurement that decides whether that device may be
labelled a capillary *solver* or only a static instrument.

Every number below is injected from `benchmarks/laplace_sph/results_modal/results_modal.json` by
`benchmarks/laplace_sph/make_doc_modal.py`; gates were declared in the JSON before the sweep ran.

## ① Case (physics, geometry, parameters)

A 2-D liquid drop in an equal-density, initially quiescent carrier block. The drop is *deformed* to
an ellipse of equal area (`a/b = aspect`, `pi a b = pi R_eq^2`), which is **not** an equilibrium: it
is the mode-2 (Rayleigh) shape perturbation `r(theta) = R(1 + eps cos 2 theta)` with
`eps = (a-b)/(a+b)`. A correct capillary solver must make it oscillate at the linear frequency

```
omega^2 = sigma n (n^2 - 1) (k^{2n} - 1) / (2 rho R^3 k^{2n}),   n = 2,  k = R_c / R
```

which is the classic `sigma n(n^2-1)/(rho R^3)` for a drop in vacuum, halved because the equal-density
outer flow carries the same kinetic energy as the inner one, and reduced by `k^{2n}/(k^{2n}-1)`
because the carrier's velocity-pinned layer is a rigid wall at `R_c = R + 12 mm` (`k = 1.6`).
The `(n^2-1)` comes from the incompressibility constraint, not `n^2`: the geometry alone gives
`n^2`, and the O(eps^2) area constraint removes exactly one `n^2`. Both ends of that band are
carried through the tables.

| quantity | value |
|---|---|
| `R_eq` | 20.0 mm |
| `sigma` | 0.072 N/m |
| `rho0` (both phases) | 1000 kg/m³ |
| carrier padding | 12.0 mm (`k` = 1.6 at `a/b` = 1) |
| kernel | cubic spline, `h/dx` = 1.3 |
| time step | `dt` = 0.10 `t_sigma` |
| aspects `a/b` | 1, 1.02, 1.05, 1.1, 1.2, 1.5 |
| resolutions `R_eq/dx` | 8, 10 |
| analytic `omega` band | 4.639 – 6.000 rad/s |

## ② Method (what was measured, and with what)

One step from rest, no time marching. With `u* = dt a_arm` and `u = P u*` (the adjoint projection of
`tensorsph.projection_adjoint`), the modal average of the radial velocity over the near field
(`r < 1.5 R_eq`) must equal `-omega^2 eps R dt` for the linear eigenmode, which gives a measured
frequency per arm:

```
omega_meas = sqrt( -<u . rhat cos 2 theta> / (eps R_eq dt) )
```

Three arms share one lattice, one colour/curvature/force operator and one projector:

| arm | force fed to the momentum update | who ships it |
|---|---|---|
| `full` | the CSF acceleration `-sigma kappa |grad c| n / rho` | rounds 1–2, `relax_projection_consistent` |
| `irrotational` | its discrete gradient part `G q` | **round 3, `relax_projection_balanced`** |
| `solenoidal` | the remainder `a - G q` (discarded by round 3) | — |

This is a *linear-response* measurement, so it needs no march and no periodic fit: the initial
acceleration of an inviscid drop released from rest is exactly the linear response. Tool chain:
`benchmarks/laplace_sph/modal_forcing_split.py` (torch float64, matrix-free CG on `A = D D^T`,
`rel_tol` 1e-8), `tests/test_modal_forcing_split.py` (11 tests), this page's generator.

## ③ Results

### The shipped (`full`) arm against the analytic band

| `a/b` | `eps` | `R_eq/dx` | `kappa` force `omega` [rad/s] | vs wall band | vs unconfined | solenoidal fraction |
|---|---|---|---|---|---|---|
| 1.00 | 0.00 | 8 | (static control) | — | — | 0.157 |
| 1.02 | 0.0099 | 8 | 1.099 | -77.0 % | -78.8 % | 0.188 |
| 1.05 | 0.0244 | 8 | 3.507 | -26.4 % | -32.5 % | 0.228 |
| 1.10 | 0.0476 | 8 | 4.203 | -11.5 % | -19.1 % | 0.172 |
| 1.20 | 0.0909 | 8 | 3.998 | -15.3 % | -23.1 % | 0.220 |
| 1.50 | 0.2000 | 8 | 3.829 | -17.5 % | -26.3 % | 0.258 |
| 1.00 | 0.00 | 10 | (static control) | — | — | 0.203 |
| 1.02 | 0.0099 | 10 | 3.423 | -37.9 % | -43.0 % | 0.204 |
| 1.05 | 0.0244 | 10 | 3.719 | -32.4 % | -38.0 % | 0.214 |
| 1.10 | 0.0476 | 10 | 3.842 | -30.0 % | -36.0 % | 0.205 |
| 1.20 | 0.0909 | 10 | 4.069 | -25.4 % | -32.2 % | 0.242 |
| 1.50 | 0.2000 | 10 | 3.772 | -29.6 % | -37.1 % | 0.272 |

Worst error against the *nearer* end of the band, over every deformed case: **77.0 %** (gate 15 %, verdict **FAIL**).  Restricted to `a/b >= 1.05` — the aspects whose mode-2 content is above the lattice noise at every
level run here — the same worst case is **32.4 %**.

The `a/b = 1.02` row is reported but should not be read as a frequency: `eps = 0.01` is one
lattice-cell-scale deformation at `R_eq/dx = 8`, and the measured `omega` moves by a factor 3
between `R/dx = 8` and 10, which is the signature of discretisation noise rather than of a
restoring force.  It is kept in the table (and in the failing gate) rather than dropped.

### The round-3 balanced arm

| `a/b` | `R_eq/dx` | `omega` balanced [rad/s] | `omega` full [rad/s] | `R2` = balanced/full | `max|u|/v_ref` (balanced) |
|---|---|---|---|---|---|
| 1.00 | 8 | n/a | n/a | 2.33e-08 | 1.29e-09 |
| 1.02 | 8 | n/a | 1.099 | 1.27e-09 | 1.83e-09 |
| 1.05 | 8 | n/a | 3.507 | 1.88e-08 | 1.62e-09 |
| 1.10 | 8 | n/a | 4.203 | 4.46e-09 | 1.28e-09 |
| 1.20 | 8 | 0.000 | 3.998 | 6.05e-09 | 1.62e-09 |
| 1.50 | 8 | n/a | 3.829 | 5.08e-11 | 1.49e-09 |
| 1.00 | 10 | n/a | n/a | 9.46e-09 | 3.21e-09 |
| 1.02 | 10 | n/a | 3.423 | 1.28e-08 | 2.14e-09 |
| 1.05 | 10 | 0.000 | 3.719 | 4.35e-09 | 3.55e-09 |
| 1.10 | 10 | 0.001 | 3.842 | 1.73e-08 | 2.06e-09 |
| 1.20 | 10 | 0.000 | 4.069 | 6.27e-10 | 1.92e-09 |
| 1.50 | 10 | n/a | 3.772 | 4.20e-09 | 1.70e-09 |

`R2` is at the round-off floor of the projector in **every** case (`min R2 = 5.08e-11`): the balanced arm keeps 5.08e-09 % of the capillary dynamics.  This is not a resolution effect and not a viscosity effect — it is an algebraic identity of the
construction: `u* = dt G q` implies `D u* = -A q`, hence the pressure solve returns `p = (rho/dt) q`
and `u = u* - (dt/rho) G p = 0` exactly.  The projector is the inverse of the very operator the arm
injects, so **every** configuration — deformed or not — is a frozen fixed point.  Round 3's static
result (`max|u|/v_ref ~ 3e-7`) is reproduced here as a control
(`3.21e-09`, gate 1.00e-05): that
number was never evidence of a drop *held* at rest, it is evidence of a drop that cannot move.

### Why the discarded part is the physical one

In the continuum the capillary traction of a static circle is a pure gradient, which is what round 3
argued.  Off equilibrium it is not: the hydrostatic balance `grad p = f` is *removed by the
projection itself*, so what drives the interface is precisely the discrete **solenoidal** remainder.
The table's last column is the size of that remainder (`||a_sol||/||a||`, 16–26 % of the force), and
the measured `omega` of the `full` arm follows its modal content — the force fraction is a fifth of
the force but *all* of the dynamics.

| `a/b` | `R_eq/dx` | modal accel full (projected) [m/s²] | modal accel irrotational (projected) | modal accel solenoidal |
|---|---|---|---|---|
| 1.02 | 8 | -0.000 | 3.04e-13 | -0.000 |
| 1.05 | 8 | -0.006 | 1.13e-10 | -0.006 |
| 1.10 | 8 | -0.017 | 7.51e-11 | -0.017 |
| 1.20 | 8 | -0.029 | -1.76e-10 | -0.029 |
| 1.50 | 8 | -0.059 | 2.98e-12 | -0.059 |
| 1.02 | 10 | -0.002 | 2.97e-11 | -0.002 |
| 1.05 | 10 | -0.007 | -2.94e-11 | -0.007 |
| 1.10 | 10 | -0.014 | -2.43e-10 | -0.014 |
| 1.20 | 10 | -0.030 | -1.89e-11 | -0.030 |
| 1.50 | 10 | -0.057 | 2.39e-10 | -0.057 |

## ④ Figures

![modal transferability](../benchmarks/laplace_sph/results_modal/modal_transferability.png)

(a) measured mode-2 frequency of the shipped arm against the two ends of the analytic band (rigid
wall at `k = 1.6`, and unconfined); (b) the fraction of that frequency the round-3 arm keeps,
log scale — it sits at the round-off floor (`1e-7` marked) rather than anywhere near the 0.95 gate;
(c) side by side at the coarsest level: the *force* the split drops (16–26 %) versus the *dynamics*
it drops (100 %).

## ⑤ Notes, limits, scope

* **What round 3 still is good for.** The static Laplace jump it reports (`dp = rho [q]`, +0.03 % at
  `R/dx = 10`) is a valid statement about the *pressure functional* of the CSF field: `q` is the
  discrete potential whose jump equals `sigma/R`.  It is not a statement that the scheme holds a drop
  at rest under its own dynamics.
* **What it must not be used for.** Any capillary dynamics — oscillation, relaxation of a deformed
  drop, break-up, capillary waves — because the arm zeroes the momentum update identically.  In
  `xval_projection_laplace_wellbalanced.md` the caveat is phrased as "scope is the equilibrium
  branch"; the measurement here says the scope is narrower than that: *the arm has no dynamics at
  all*, on or off equilibrium.
* **The shipped arm is not validated either.** Its mode-2 frequency is `-32.4 … -11.5 %` against the wall end of the band over
  `a/b >= 1.05`, without a clean convergence trend between `R_eq/dx = 8` and 10.  So the dynamic
  branch of this anchor stays **open**: the present page supplies its first analytic reference and
  measures the current deficit, it does not close it.  A convergence ladder (`R/dx` = 16, 24) is
  queued as compute job `cj4588`.
* **Systematics not quantified here.** (i) the carrier is a square block with a pinned edge layer,
  analysed as a concentric wall at `k = 1.6`: corners are farther away, so the true reference sits
  between the two band ends; (ii) the force field is *smoothed* over ~3 `dx` by the colour function,
  so part of the deficit is the curvature operator's mode-2 response rather than the projection;
  (iii) `eps` up to 0.2 leaves the linear regime at the margin; the trend across aspects is reported
  as a trend, not as a linear-law fit.
* **Equal densities only.** With a real density ratio, both the analytic reference and the operator
  set change; nothing here transfers to a two-phase solver claim.

## ⑥ Error judgement (required)

| quantity | reference | measured | gate | verdict |
|---|---|---|---|---|
| mode-2 frequency, shipped arm (all deformed cases) | analytic band, nearer end | 77.0 % | ≤ 15 % | **FAIL** |
| mode-2 frequency, shipped arm (`a/b` ≥ 1.05) | analytic band, nearer end | 32.4 % | ≤ 15 % | **FAIL** |
| mode-2 dynamics kept by the balanced arm | 1.0 | 5.08e-11 | ≥ 0.95 | **FAIL** |
| static control, balanced arm on the circle | `max|u|/v_ref` ≤ 1e-5 | 3.21e-09 | ≤ 1e-5 | PASS |
| Helmholtz split reconstruction | 0 | 5.41e-18 | ≤ 1e-12 | PASS |
| Helmholtz split orthogonality | 0 | 3.68e-10 | ≤ 1e-6 | PASS |

**Is every key quantity within 3 %?**  No.  The Laplace *jump* of the static branch is (round 3:
+0.03 % at `R/dx = 10`); the mode-2 *frequency* of the shipped arm is not (deficit `32.4 %` at best over the aspects measured), and the balanced arm's dynamics are
wrong by 100 %.  The honest verdict for this anchor's dynamic branch is **FAIL / open**, with the
mechanism now identified.

## ⑦ Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_sph/modal_forcing_split.py --run --level 8
PYTHONPATH=src python benchmarks/laplace_sph/modal_forcing_split.py --run --level 10
PYTHONPATH=src python benchmarks/laplace_sph/modal_forcing_split.py --merge --figure
PYTHONPATH=src python benchmarks/laplace_sph/make_doc_modal.py
PYTHONPATH=src:. python -m pytest tests/test_modal_forcing_split.py -q
```

| artefact | role |
|---|---|
| `src/tensorsph/projection_adjoint.py` | `helmholtz_split`, `project_velocity` (unchanged) |
| `benchmarks/laplace_sph/modal_forcing_split.py` | single-step measurement, gates, figure |
| `benchmarks/laplace_sph/results_modal/*.json` | per-case records + merged record |
| `tests/test_modal_forcing_split.py` | analytic reference, geometry, instrument, both claims |
