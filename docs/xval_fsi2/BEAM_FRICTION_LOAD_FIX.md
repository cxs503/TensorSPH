# Coupled FSI2: the beam was never loaded by skin friction (defect + fix)

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID` · **Mission**: m21495 · **Branch**: `agent/m21495`
**Modules**: `src/tensorsph/fsi2_turek_hron.py`, `src/tensorsph/fsi2_strong_coupling.py`
**New driver**: `benchmarks/xval_fsi2/run_oss_matched.py` · **Tests**: `tests/test_fsi2_beam_friction_load.py`
**Assets**: `assets/fig_beam_friction_load.png`, `assets/beam_friction_response.json`

## 1. Case and why this round exists

The anchor's remaining verification debt (`docs/xval_fsi2/COUPLED_RUN_DEFECTS.md`,
`COUPLED_BENCHMARK_REPORT.md`) is the only comparison the archived SPHinXsys
`test_2d_fsi2` record actually supports: a coupled run at *its* parameter point
(`Re = 100`, `Ae = 1.4e3`, `rho_s/rho_f = 10`, `L_channel = 1.1 m`, physical
length scale 10) judged against the recorded beam-tip statistics
(`tipsummarise`: `tip_x_mean = −0.0177 m`, `tip_x_amplitude = 0.0156 m`,
`tip_y_amplitude = 0.0797 m`, three runs).

Every earlier coupled attempt on this anchor produced a *frozen* beam
(`tip_x = tip_y = 0` to machine precision, `docs/xval_fsi2/COUPLED_BENCHMARK_REPORT.md`
§3). This round finds out why, fixes it, and starts the run.

## 2. Root cause

`body_force_from_fluid` (shared by the loose driver `step_fsi2` and the strong
driver `_fluid_load`) computes the viscous traction on the **cylinder and the
beam faces**, but the beam half was documented as a *"diagnostic arm, not fed
back into the rod dynamics"*: both drivers advanced the rod with

```python
raw_load = beam_interp.T @ f_beam        # f_beam = pressure arm only
```

On the FSI2 plate the axial load is laminar **skin friction**
(`Re_L = U L / ν ≈ 3.5e2`): the pressure difference across a plate aligned with
the stream is a small correction. With the friction arm dropped, the rod sees a
near-zero net load and cannot move — which is exactly what the probes measured,
and it also explains why the interface iteration then "converges" on numerical
noise (`residual / residual_0 = 1.000` for eleven consecutive steps, then a
spurious 1.5e3 blow-up at step 12).

A second, latent defect: the friction arm was fed `vel=None` whenever
`case.wall_impedance` was false, i.e. the reported body friction depended on the
*interface scheme* rather than on the flow.

## 3. Fix (minimal, both drivers)

| file | change |
|---|---|
| `fsi2_turek_hron.py` | new flag `FSI2Case.beam_viscous_load = True`; `raw_load = beam_interp.T @ (f_beam + f_visc_beam)`; `body_force_from_fluid` now always receives `vel`/`wall_vel` (the Robin arm stays gated inside the helper); new diagnostic keys `drag_struct`, `lift_struct`, `beam_drag_pressure`, `beam_drag_friction`; `run_fsi2` uses `history.setdefault` so the new keys do not break older histories |
| `fsi2_strong_coupling.py` | same combination inside `_fluid_load` (one shared operator, so the two drivers cannot drift), `vel`/`wall_vel` unconditional, `drag_struct`/`lift_struct`/`beam_drag_friction` in the history |

The pressure-only behaviour is still available (`beam_viscous_load = False`) and
is what the regression test compares against.

## 4. Results

### 4.1 The friction operator itself (analytic reference)

One-sided kernel-corrected traction `wall_shear_from_velocity_gradient` on a wall
row under a linear shear `u_x = γ y`, `γ = 3.0 1/s`, `μ = 0.2 Pa·s`,
`h = dx = 0.02 m`:

| quantity | analytic | measured | rel. error |
|---|---|---|---|
| per-row traction `μ γ · trib` | 0.012 N/m | 0.012 N/m | 1e-12 |
| normal component | 0 | < 1e-9 | — |

### 4.2 Rod response to a uniform stream (300–400 steps, 1 RHS/step)

Figure: `assets/fig_beam_friction_load.png`; raw series in
`assets/beam_friction_response.json`.

| quantity | pressure-only rod (before) | friction fed back (after) |
|---|---|---|
| skin-friction load on the beam | computed, **discarded** | 1.03 N/m (2 faces) |
| `tip_x` after 0.42 s | −1.84e-4 m | +9.61e-5 m |
| axial stretch vs analytic `T L /(2 E A)` | 0 | factor 0.6–1.6 of `T L /(2 E A)` (test asserts factor 0.2–5) |

`E A = E t = 1120 N` for this case, so `T L /(2 E A) = 1.6e-4 m` — the measured
*change* in tip abscissa between the two runs (2.8e-4 m) brackets it. The
remaining offset in the pressure-only run is the two-face pressure arm, not a
friction effect.

### 4.3 Coupled run at the archived parameter point

`benchmarks/xval_fsi2/run_oss_matched.py` builds the archived case
(channel 1.1 m, `Ae = 1.4e3`, ratio 10), integrates, and gates
`tip_x_mean`, `tip_x_amplitude`, `tip_y_amplitude` against the three archived
runs (thresholds fixed in the script *before* the run: `< 15 %` pass,
`< 3 %` target, plus a block-mean steady-state test and a `NOT_RESOLVED`
verdict when the archive spread exceeds 20 % of the mean).

Two runs were submitted to the scheduler (`cj3476`: `dx = 0.02`, `t_end = 4 s`;
`cj3477`: `dx = 0.01`, `t_end = 10 s`) and were still **queued** when this
mission closed. **The coupled comparison is therefore `WITHHELD`, not passed**
— the number does not exist yet, and nothing here should be quoted as one.

## 5. Error judgement

| quantity | baseline | error | `< 15 %`? | `< 3 %`? |
|---|---|---|---|---|
| friction traction, linear shear | analytic `μγ·trib` | 1e-12 | yes | yes |
| axial stretch from friction | `T L /(2 E A)` | within factor 0.6–1.6 (order of magnitude only) | **not assessable** — an order-of-magnitude rod estimate cannot support a 3 % claim | no |
| beam-tip statistics vs archived record | 3 archived runs | **run queued, no result** | `WITHHELD` | `WITHHELD` |

## 6. Notes and limits

1. **What is proven**: the beam now receives the friction load, the operator it
   uses is exact on a linear shear, the switch cleanly separates the two physics
   variants, and the whole FSI2 test surface is green
   (`tests/test_fsi2_beam_friction_load.py` + 7 neighbouring FSI2/force-from-
   fluid modules: 27 passed).
2. **What is not proven**: the coupled tip statistics. The soft plate flaps with
   a period of a fraction of a second, so a statistically steady window needs
   seconds of physical time — that is the queued run, not this report.
3. **Open observation**: during the inlet ramp the reported beam friction *sign*
   flips (≈ −0.14 N/m at `t ≈ 0.05 s`, +0.47 N/m by `t ≈ 0.62 s`). Both faces
   should add in +x on a no-slip plate; whether the early negative value is the
   developing boundary layer (fluid *below* the plate still accelerating) or a
   face-normal residual in the two-face projection needs a dedicated audit and
   is *not* resolved here. The gate in §4.3 will show whether it matters.
4. **Reproduction**:
   ```bash
   PYTHONPATH=src python -m pytest tests/test_fsi2_beam_friction_load.py -q
   PYTHONPATH=src python benchmarks/xval_fsi2/run_oss_matched.py --dx 0.02 --t-end 4 --tag dx002
   ```