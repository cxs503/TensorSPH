# CASE-TAYLOR-GREEN-SPH — Taylor-Green vortex decay (TensorSPH)

Generated 2026-11-12T03:19:24Z from `/root/tp-agent/work/m15860/runs/taylor_green/m15860/results.json` by
`benchmarks/taylor_green/make_doc.py`; artefacts sha256-pinned in
`/root/tp-agent/work/m15860/runs/taylor_green/m15860/evidence.json`.  No number below is typed by hand.

## ① Case: physics, geometry, parameters

Two counter-rotating vortices in a fully periodic square box; the field is an
exact unsteady solution of the incompressible Navier-Stokes equations, so the
kinetic energy decays with the exact exponent `4 nu k^2`.  Reference: Taylor &
Green (1937) *Proc. R. Soc. Lond. A* **158**, 499-521; see
`docs/references/taylor-green-reference.md`.

| parameter | symbol | value |
|---|---|---|
| box side (periodic) | $L$ | 1 m |
| wavenumber | $k$ | 6.28319 1/m |
| peak initial speed | $U_0$ | 0.2 m/s |
| artificial sound speed | $c_0$ | 10 m/s |
| **Mach number** | $M = U_0/c_0$ | 0.02 |
| kinematic viscosity | $\nu$ | 0.0159155 m²/s |
| **Reynolds number** | $Re = U_0/(\nu k)$ | 2 |
| density / Tait exponent | $\rho_0$, $\gamma$ | 1000, 7 |
| smoothing length | $h/\Delta x$ | 1.3 |
| acoustic Courant number | $C$ | 0.3 |
| simulated time | $t_{end}$ | 0.6 s |
| exact energy-decay rate | $4\nu k^2$ | 2.5132741 1/s |
| exact $E(t_{end})/E(0)$ | $e^{-4\nu k^2 t}$ | 0.22136 |
| main resolution | $\Delta x$ | 0.025 m → 1600 particles, 628 steps, dt 0.0009559 s |

## ② Method and implementation

* **Solver** `tensorsph.vv.tgv`: PyTorch SPH, Morris (1997) viscous Laplacian (the
  operator the production `droplet_impact.rhs` uses), symmetric `p/rho^2` pressure
  term, Tait EOS, minimum-image periodicity, neighbour lists rebuilt every 20
  steps with a 0.4 h skin.
* **Case layer** `tensorsph.taylor_green`: parameter table, reference checks,
  ref-vs-got rows, sensitivity studies, figures, `results.json`, `run.log`,
  `environment.json`.
* **Entries**: `python -m tensorsph --case taylor_green --output <dir>` (package
  CLI), `python -m tensorsph.taylor_green --out <dir>` (module CLI),
  `python benchmarks/taylor_green/run.py --out <dir>` (driver).
* **Downstream caller**: verification-ladder rung C2 (`tensorsph.vv.ladder`) calls
  `taylor_green.run_case` — ladder, CLI and benchmark gate one code path.
* **Measured quantity**: least-squares slope of `ln E(t)` over
  `[0.06, 0.51] s` (10 %–85 % of
  `t_end`), compared with `4 nu k^2`.
* **Reproduction** (one command, 239 s wall on 4 threads):
  `PYTHONPATH=src python -m tensorsph --case taylor_green --output runs/taylor_green/m15860`
* **Toolchain**: Python 3.11.6, torch
  2.7.1, loongarch64,
  4 threads.

## ③ Results

### Reference vs got (main case, dx = 0.025 m)

| quantity | unit | ref (exact) | got (SPH) | error [%] | tol [%] | verdict |
|---|---|---|---|---|---|---|
| energy_decay_rate | 1/s | 2.5132741 | 2.4836508 | 1.179 | 8 | ✅ |
| energy_ratio_at_t_end | - | 0.2213601 | 0.21486398 | 2.935 | 8 | ✅ |
| amplitude_ratio_at_t_end | - | 0.47048922 | 0.4657502 | 1.007 | 8 | ✅ |


Worst row 2.935 %; all rows pass the 8 % criterion:
**PASS**.

### Convergence in resolution

| dx [m] | particles | steps | rate [1/s] | ref [1/s] | error [%] | discrete-Laplacian error [%] | verdict |
|---|---|---|---|---|---|---|---|
| 0.05 | 400 | 314 | 2.438 | 2.513 | 3.00 | 2.65 | ✅ |
| 0.025 | 1600 | 628 | 2.484 | 2.513 | 1.18 | 0.68 | ✅ |
| 0.0125 | 6400 | 1256 | 2.494 | 2.513 | 0.75 | 0.18 | ✅ |


### Mach-number sensitivity (dx = 0.05 m)

| M | U0 [m/s] | rate [1/s] | error [%] | verdict |
|---|---|---|---|---|
| 0.002 | 0.02 | 2.452 | 2.45 | ✅ |
| 0.005 | 0.05 | 2.46 | 2.11 | ✅ |
| 0.01 | 0.1 | 2.458 | 2.18 | ✅ |
| 0.02 | 0.2 | 2.438 | 3.00 | ✅ |
| 0.05 | 0.5 | 6.876 | 173.58 | ❌ |


### Reynolds-number sensitivity (dx = 0.05 m)

| Re | nu [m²/s] | rate [1/s] | ref [1/s] | error [%] | verdict |
|---|---|---|---|---|---|
| 2 | 0.0159155 | 2.438 | 2.513 | 3.00 | ✅ |
| 5 | 0.0063662 | 1.02 | 1.005 | 1.44 | ✅ |
| 10 | 0.0031831 | 0.5663 | 0.5027 | 12.65 | ❌ |
| 20 | 0.00159155 | 0.3458 | 0.2513 | 37.59 | ❌ |


### Negative control (nu = 0)

| quantity | value |
|---|---|
| measured decay rate | 0.13053 1/s |
| `E(t_end)/E(0)` | 0.576 |
| equivalent spurious viscosity | 0.00082656 m²/s |
| fit quality $R^2$ | 0.4507 |

## ④ Figures

| figure | what it shows |
|---|---|
| `/root/tp-agent/work/m15860/runs/taylor_green/m15860/fig1_tgv_velocity_field.png` | exact velocity field and speed at `t = 0` on the particle lattice |
| `/root/tp-agent/work/m15860/runs/taylor_green/m15860/fig2_tgv_energy_decay.png` | SPH `E(t)/E(0)` vs the exact exponential, fit window shaded |
| `/root/tp-agent/work/m15860/runs/taylor_green/m15860/fig3_tgv_convergence.png` | decay-rate and discrete-Laplacian error vs `dx`, with the 8 % line |
| `/root/tp-agent/work/m15860/runs/taylor_green/m15860/fig4_tgv_sensitivity.png` | Mach window and Reynolds window |

## ⑤ Notes, limitations, validity window

1. **The validity window is bounded by spurious dissipation.**  The `nu = 0`
   control loses 42 % of `E(0)` by
   `t_end` (equivalent to 0.00082656 m²/s).
   Once the physical rate approaches that floor the fit is contaminated: inside
   8 % for `Re <= 5`, and **failing** at `Re = 10`
   (12.65 %) and `Re = 20`
   (37.59 %).  This case is therefore declared at
   **Re = 2** and is **not** a high-Reynolds anchor.
2. **The loss is a late-time particle-disorder collapse.**  In the `nu = 0` control
   the energy is flat to ~0.1 % until `t ≈ 0.45 s` and then drops; the fit window
   ends at `0.85 t_end` to stay clear of it.  Refining `dx` does not remove it
   (relative disorder is scale invariant), which is why errors in the low-viscosity
   sweeps are not monotone in `dx`.
3. **Mach window**: flat (~2-3 %) for `M <= 0.02`; at `M = 0.05` the weakly
   compressible assumption collapses (174 %).
4. **Integral quantities only**: decay rate, `E(t_end)/E(0)` and peak-speed ratio
   are gated; pointwise `u(x,t)` errors are not, because the SPH field is not
   remeshed onto the Eulerian reference.
5. **Cost**: 239 s for 3 resolutions + 5 Mach + 4 Reynolds +
   control + figures — below the 10 min local limit, so no cluster job was used.

## ⑥ Error verdict (explicit)

| key quantity | reference | got | relative error | < 3 %? | < 8 % (case tolerance)? |
|---|---|---|---|---|---|
| energy_decay_rate | 2.51327 | 2.48365 | 1.179 % | YES | YES |
| energy_ratio_at_t_end | 0.22136 | 0.214864 | 2.935 % | YES | YES |
| amplitude_ratio_at_t_end | 0.470489 | 0.46575 | 1.007 % | YES | YES |


The main-case decay rate is inside **3 %** (1.18 %)
and inside the case criterion of **8 %**; the energy-ratio and amplitude rows are
also inside 3 %.  The Reynolds sweep beyond `Re = 5` is reported as a **measured
failure**, not hidden behind the passing design point.
