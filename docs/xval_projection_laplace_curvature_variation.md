# Round 5 — Young-Laplace at a **non-constant curvature** interface

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · round 5 · rounds 1-4 = `docs/xval_projection_laplace_oss.md`, `docs/xval_projection_laplace_robustness.md`, `docs/xval_projection_laplace_density_ratio.md`.

## 0. Why a fifth round

Rounds 1-4 measured the Young-Laplace jump on a **circular** droplet at `R = 20 mm`, and every gate they declare is a gate on the *mean* over an interface of constant curvature. On a circle a curvature estimator with a station-dependent error cancels in the integral, and a constant relative bias is indistinguishable from a calibration, so those rounds could not separate *the law holds* from *the circle is a friendly geometry*.

The CSF model discretises the **differential** law `dp(s) = sigma kappa(s)` at every interface station, so this round replaces the circle by an **ellipse of the same area** (`pi a b = pi R_eq^2`), whose curvature varies along the interface by `(a/b)^3`. Same `sigma`, same `R_eq`, same lattice, same operators, same rho-blind force-density Poisson of round 4.

![fig. 1](../scripts/xval_projection_laplace/results/curvature_variation.png)

Fig. 1 — (a) pointwise curvature error around the interface for four aspect ratios; (b) the two instruments station by station; (c) systematic error vs aspect ratio (bias) against the station scatter, with the 3 % and 15 % bands; (d) the window-scale ladder (log scale). All four panels are drawn from the JSON.

## 1. Case (physics, geometry, parameters)

| quantity | value |
|---|---|
| interface | ellipse, semi-axes `a`, `b`, `(a/b)` = 1, 1.5, 2, 3 |
| area-equivalent radius | `R_eq = 20.0 mm` (round 1's `R`) |
| surface tension | `sigma = 0.072 N/m` (water/air, 20 C) |
| density (both phases) | `rho0 = 1000 kg/m^3` |
| resolution ladder | `R_eq/dx` = 10, 20, 40 |
| resolutions precondition | a station is *resolved* when its radius of curvature spans >= 5 cells |
| carrier | triangular lattice, filled vapour (equal density), Dirichlet block `12 dx` |
| stations | 24 around the ellipse, windows tile in the ellipse parameter |
| reference | `dp(t) = sigma * kappa_exact(t)`, `kappa = a b / (a^2 sin^2 t + b^2 cos^2 t)^(3/2)` |

Curvature range per configuration (`kappa_max/kappa_min`, and the smallest radius of curvature in cells):

| a:b | `R_eq/dx` | `kappa_max/kappa_min` | `R_curv,min/dx` | resolved |
|---|---|---|---|---|
| 1 | 10 | 1.00 | 10.0 | yes |
| 1 | 20 | 1.00 | 20.0 | yes |
| 1 | 40 | 1.00 | 40.0 | yes |
| 1.5 | 10 | 3.37 | 5.4 | yes |
| 1.5 | 20 | 3.37 | 10.9 | yes |
| 1.5 | 40 | 3.37 | 21.8 | yes |
| 2 | 10 | 8.00 | 3.5 | no |
| 2 | 20 | 8.00 | 7.1 | yes |
| 2 | 40 | 8.00 | 14.1 | yes |
| 3 | 10 | 27.00 | 1.9 | no |
| 3 | 20 | 27.00 | 3.8 | no |
| 3 | 40 | 27.00 | 7.7 | yes |

## 2. Method

Four readouts per configuration, one Poisson solve each; every arm is the same code path with one object replaced:

| arm | curvature `kappa` | `delta_s` | what it isolates |
|---|---|---|---|
| `model` | shipped MLS `div n` | shipped `|grad c|` | **the scheme under test** |
| `kappa_exact` | analytic ellipse `kappa` per particle | shipped | curvature-estimator error |
| `delta_local` | shipped | renormalised per station to `oint delta_s dn = 1` | `delta_s` normalisation (diagnostic: uses exact arclength) |
| `both_exact` | analytic | renormalised | the operator/instrument floor |

* **Instrument 1 (traction flux).** `dp_flux(t) = -sum_k f_k . n_k V_k / L(t)` over the `|d| <= 3 dx` band in the station's window (`d` = signed distance to the ellipse, `n_k` = the particle's own normal, `L` = the window's arclength). It never touches the pressure solve.
* **Instrument 2 (pressure readout).** `dp_pressure(t) = <p>_in - <p>_out` from the round-4 rho-blind Poisson `-grad^2 p = -div(f)` with the Dirichlet gauge on the block edge: the number a projection solver reports.
* **Whitney-Graustein gauge.** `(1/2 pi) oint kappa dl = 1` exactly, for every smooth closed convex curve, at every aspect ratio and resolution.
* **Net traction.** `|oint sigma kappa n dl| / oint |sigma kappa| dl = 0` for a closed interface; on an ellipse the summands are not symmetric, so a curvature field that is right on average but wrong in distribution leaves a residual.
* The reference of a *windowed* readout is the arclength-weighted mean of the exact curvature over that window, not `kappa(t_centre)`: at a 30 deg window on an ellipse the two differ by tens of percent.

**Gates, declared before the final sweep and not relaxed afterwards** (copied verbatim into the JSON):

| gate | value |
|---|---|
| `local_bias_pct_circle_R_over_dx_ge_20` | 3.0 |
| `local_bias_pct_aspect_le_1p5_resolved` | 3.0 |
| `local_bias_pct_all_aspects_R_over_dx_40` | 15.0 |
| `station_rms_pct_aspect_le_2_resolved` | 15.0 |
| `whitney_graustein_err_pct_resolved` | 1.0 |
| `pressure_bias_pct_all_aspects_R_over_dx_40` | 15.0 |
| `net_traction_residual_model_arm` | 0.01 |
| `pressure_vs_flux_bias_diff_pct_aspect_le_1p5` | 5.0 |
| `zero_sigma_abs_dp_Pa` | 1e-09 |
| `circle_vs_module_dp_diff_pct` | 2.0 |
| `window_ladder_rms_pct_at_30deg_aspect_1p5` | 15.0 |
| `window_ladder_noise_halving` | 0.5 |
| `both_exact_bias_pct_aspect_le_2_resolved` | 1.5 |

## 3. Results

### 3.1 Everything, one line per equilibrium (all from the JSON)

| a:b | `R_eq/dx` | `R_curv,min/dx` | res. | N | flux bias % | flux rms % | flux max % | p bias % | p rms % | `kappa` bias % | `kappa` rms % | `oint kappa dl/2pi` % | net trac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 10.0 | y | 2270 | -0.18 | 26.22 | 57.2 | +0.11 | 4.81 | +0.71 | 23.46 | -0.18 | 2.3e-03 |
| 1 | 20 | 20.0 | y | 4773 | -0.46 | 32.17 | 77.3 | -1.38 | 5.49 | -0.38 | 29.29 | -0.46 | 6.9e-04 |
| 1 | 40 | 40.0 | y | 12645 | -0.55 | 29.63 | 55.0 | -1.35 | 8.70 | -0.14 | 29.59 | -0.55 | 7.8e-04 |
| 1.5 | 10 | 5.4 | y | 2716 | +1.46 | 26.41 | 65.3 | -9.95 | 38.13 | +2.93 | 24.86 | -0.12 | 1.2e-03 |
| 1.5 | 20 | 10.9 | y | 6205 | -0.30 | 41.55 | 94.6 | +3.56 | 17.34 | +0.45 | 42.04 | -0.30 | 9.5e-04 |
| 1.5 | 40 | 21.8 | y | 17202 | -0.55 | 35.57 | 125.8 | +1.34 | 12.61 | +0.26 | 35.75 | -0.63 | 1.0e-03 |
| 2 | 10 | 3.5 | n | 3203 | +4.58 | 36.70 | 96.0 | +10.78 | 53.10 | +5.78 | 34.39 | +0.65 | 1.2e-03 |
| 2 | 20 | 7.1 | y | 7614 | +0.97 | 41.70 | 98.5 | +10.74 | 28.01 | +1.29 | 40.45 | -0.27 | 1.2e-03 |
| 2 | 40 | 14.1 | y | 21863 | +0.93 | 35.02 | 97.3 | +6.80 | 19.24 | +1.73 | 35.99 | -0.30 | 7.9e-04 |
| 3 | 10 | 1.9 | n | 4012 | +15.07 | 67.69 | 134.4 | +44.02 | 86.36 | +17.82 | 67.08 | +3.07 | 4.7e-03 |
| 3 | 20 | 3.8 | n | 10098 | +4.82 | 71.27 | 282.0 | +30.35 | 49.93 | +5.68 | 71.73 | +0.26 | 2.2e-03 |
| 3 | 40 | 7.7 | y | 30644 | -0.05 | 38.95 | 101.5 | +15.94 | 30.06 | +0.21 | 38.67 | -0.30 | 2.0e-03 |

### 3.2 Ablations: where the error comes from

| a:b | `R_eq/dx` | model bias % | `kappa_exact` bias % | `delta_local` bias % | `both_exact` bias % | model rms % | `kappa_exact` rms % | `oint delta_s dn` |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 | -0.18 | -1.51 | +0.71 | +0.00 | 26.22 | 8.44 | 0.985 |
| 1 | 20 | -0.46 | -0.81 | -0.38 | +0.00 | 32.17 | 4.48 | 0.992 |
| 1 | 40 | -0.55 | -0.48 | -0.14 | +0.00 | 29.63 | 2.07 | 0.995 |
| 1.5 | 10 | +1.46 | -1.49 | +2.93 | +0.13 | 26.41 | 8.61 | 0.984 |
| 1.5 | 20 | -0.30 | -0.99 | +0.45 | -0.08 | 41.55 | 4.55 | 0.991 |
| 1.5 | 40 | -0.55 | -0.64 | +0.26 | +0.04 | 35.57 | 2.39 | 0.993 |
| 2 | 10 | +4.58 | -1.02 | +5.78 | +1.03 | 36.70 | 7.80 | 0.982 |
| 2 | 20 | +0.97 | -0.57 | +1.29 | +0.64 | 41.70 | 3.92 | 0.988 |
| 2 | 40 | +0.93 | -0.71 | +1.73 | -0.03 | 35.02 | 2.25 | 0.993 |
| 3 | 10 | +15.07 | +2.83 | +17.82 | +6.56 | 67.69 | 17.93 | 0.971 |
| 3 | 20 | +4.82 | -0.22 | +5.68 | +0.94 | 71.27 | 6.46 | 0.990 |
| 3 | 40 | -0.05 | -0.02 | +0.21 | +0.46 | 38.95 | 3.13 | 0.996 |

### 3.3 The law at coarser scales (window ladder, `R_eq/dx` = 40)

| a:b | averaging half-window [deg] | stations | flux bias % | flux rms % | p bias % |
|---|---|---|---|---|---|
| 1 | 7.5 | 24 | -0.55 | 29.63 | -1.35 |
| 1 | 15 | 12 | -0.55 | 5.39 | -1.52 |
| 1 | 22.5 | 8 | -0.55 | 9.43 | -1.46 |
| 1 | 30 | 6 | -1.09 | 4.10 | -1.55 |
| 1.5 | 7.5 | 24 | -0.55 | 35.57 | +1.34 |
| 1.5 | 15 | 12 | +0.67 | 11.58 | +1.28 |
| 1.5 | 22.5 | 8 | +0.21 | 3.35 | +1.05 |
| 1.5 | 30 | 6 | -0.51 | 5.99 | +0.73 |

### 3.4 Gate verdicts

| gate | measured | declared | verdict |
|---|---|---|---|
| `local_bias_pct_circle_R_over_dx_ge_20` | +0.55 | 3.0 | PASS |
| `local_bias_pct_aspect_le_1p5_resolved` | +0.55 | 3.0 | PASS |
| `local_bias_pct_all_aspects_R_over_dx_40` | +0.93 | 15.0 | PASS |
| `station_rms_pct_aspect_le_2_resolved` | +41.70 | 15.0 | **FAIL** |
| `whitney_graustein_err_pct_resolved` | +0.63 | 1.0 | PASS |
| `pressure_bias_pct_all_aspects_R_over_dx_40` | +15.94 | 15.0 | **FAIL** |
| `net_traction_residual_model_arm` | +0.00 | 0.01 | PASS |
| `pressure_vs_flux_bias_diff_pct_aspect_le_1p5` | +3.86 | 5.0 | PASS |
| `zero_sigma_abs_dp_Pa` | +0.00 | 1e-09 | PASS |
| `circle_vs_module_dp_diff_pct` | +0.24 | 2.0 | PASS |
| `both_exact_bias_pct_aspect_le_2_resolved` | +0.64 | 1.5 | PASS |
| `window_ladder_rms_pct_at_30deg_aspect_1p5` | +5.99 | 15.0 | PASS |
| `window_ladder_noise_halving_aspect_1.0` | +0.14 | 0.5 | PASS |
| `window_ladder_noise_halving_aspect_1.5` | +0.17 | 0.5 | PASS |
| `window_ladder_rms_half_window_deg` | [7.5, 15.0, 22.5, 30.0] | informational | PASS |

## 4. Findings

1. **The differential law holds on a general curved interface, in the mean.** Over all 12 equilibria at `R_eq/dx` = 40 the station-mean error of `dp(t)/sigma kappa(t)` is -0.55 % … +0.93 % — i.e. inside the 3 % band at every aspect ratio up to 3:1 (curvature ratio 27:1). The circle was not a lucky geometry for the *mean*.
2. **The pointwise field does not hold to 15 %.** The station scatter of the same quantity is 26.2 % … 41.7 % and does **not** decay with refinement (circle: 32.2 % at `R_eq/dx` = 20 vs 29.6 % at 40). The `kappa_exact` ablation drops it to 2.07 % … 8.61 %, so the scatter is entirely the curvature estimator (`MLS div n` on the `delta_s` shell), not the traction operator.
3. **Why four rounds of circle gates could pass.** `oint kappa dl / 2pi` deviates from 1 by at most 0.63 % in every resolved configuration — the *integrated* curvature is right even where the pointwise field is 40 % off, and every earlier gate was an integral. The mean jump is an integral quantity by construction.
4. **The pressure readout is the weak instrument at high curvature.** The number a projection solver reports (`<p>_in - <p>_out`) is within 1.35 % on the circle but reaches +15.94 % at a:b = 3:1, while the traction integral on the same equilibrium is -0.05 %. The owner's 15 % gate is therefore failed by the *readout*, not by the force model — a distinction that a circle cannot expose.
5. **The `delta_s` hypothesis was falsified.** The first version of this experiment measured `oint delta_s dn` as 0.94 on a:b = 1.5:1 and the bias tracked it, which suggested a curvature-dependent normalisation defect. That deficit was an artefact of selecting the `delta_s` band by projection on a station *plane* (`R(1-cos 30 deg) = 5.4 dx` at `R = 40 dx`); with the band selected by the true signed distance, `oint delta_s dn = 0.984 … 0.996` and the `delta_local` ablation moves the bias by less than a percentage point. Recorded as a negative result: the shipped `delta_s = |grad c|` is normalised to < 1 % on this carrier at every aspect ratio.

## 5. Error verdict (per key quantity, against the owner's gates)

| key quantity | reference | best / worst measured | < 3 % ? |
|---|---|---|---|
| mean `dp` on a circle (rounds 1-4 regression) | `sigma/R_eq` = 3.6000 Pa | -0.55 % (local flux), module global readout -0.21 % | **yes** |
| mean `dp(t)` on an ellipse, a:b <= 1.5 | `sigma kappa_exact(t)` | max |bias| 0.55 % | **yes** |
| mean `dp(t)`, all aspect ratios at `R_eq/dx` = 40 | `sigma kappa_exact(t)` | max |bias| 0.93 % | **yes** |
| station-resolved `dp(t)` (pointwise field) | `sigma kappa_exact(t)` | rms up to 41.7 % | **no** (curvature noise; see finding 2) |
| reported pressure jump, a:b = 3:1, `R_eq/dx` = 40 | `sigma kappa_exact(t)` | +15.94 % | **no** (readout, see finding 4) |
| `oint kappa dl` (Whitney-Graustein) | `2 pi` | max |err| 0.63 % | **yes** |

## 6. Notes, limitations, applicability

* The vapour phase is filled with liquid-density particles, exactly as in rounds 1-4: this is a *static verification carrier* (the Young-Laplace law contains `sigma` and `R` only), not a multiphase flow model. The density ratio is covered by round 4.
* `delta_local` uses the exact arclength and is a diagnostic, not a usable scheme; the usable form would have to estimate the normalisation from the colour field alone.
* Two instrument defects were found and fixed while building this round, before the final sweep: (i) the angular window was measured in the polar angle instead of the ellipse parameter (on a:b = 1.5 that picks the wrong arc, its exact curvature is `atan(a/b) - 45 deg` away); (ii) the `delta_s` band was selected by projection on a station plane instead of the true signed distance (`R(1 - cos 30 deg)` error, which reads as a -33 % deficit at a 30 deg window). Neither survives in the shipped script; the JSON is written by the final run.
* The net-traction residual of the `kappa_exact` arm is *larger* (4.2e-02) than the model arm's (2.3e-03): the shipped, noisy curvature field is more nearly self-consistent with the shipped smoothed normals than the analytic one is. It is a property of the discretisation, not of the physics, and it is why the integral gates are so forgiving.
* Applicability: any static or quasi-static surface-tension verification of this projection-SPH module should quote the traction integral, and should not claim better than 15 % for a *pointwise* pressure jump at `h = 1.3 dx`. At `R_eq/dx = 40` the mean law is good to 1 %; a curvature field with resolved `R_curv,min >= 5 dx` is required for that.

Artifacts: `scripts/xval_projection_laplace/curvature_variation.py` (sweep, gates), `scripts/xval_projection_laplace/make_curvature_doc.py` (this page and the figure), `scripts/xval_projection_laplace/results/curvature_variation.json` + `.png`, `tests/test_xval_projection_laplace_curvature_variation.py`.
