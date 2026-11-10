# TensorSPH zero-to-one verification ladder (generated)

**Generated file** -- every number is injected from `artifacts/verification_ladder/ladder.json` by `benchmarks/verification_ladder/make_doc.py`. Do not edit by hand.

* node: `p83w1-n01-tensorsph-verification`
* schema: `1.0`
* commit: `e4da02752e2bc7c0bc4a936d10c0dfc60e143522` (dirty: True), TensorSPH 0.8.0, torch 2.7.1, python 3.11.6
* generated: 2026-11-10T01:13:57.217347+00:00
* wall clock: 189.9 s

## 1. Case introduction

Three anchors, one per physical mechanism, each with a *closed* reference. The ladder's purpose is to establish that the solver is correct at all before any application is believed:

| rung | physics | reference | gated quantity | limit | measured | verdict |
|---|---|---|---|---|---|---|
| `C1_hydrostatic` | rho g h pressure gradient balancing gravity in a still column | analytic | max interior band pressure error vs rho g h | 1.0 % | **0.924 %** | PASS |
| `C2_tgv` | viscous dissipation of a periodic vortex, lap(u) = -2 k^2 u | analytic | energy decay rate vs analytic 4 nu k^2 | 3.0 % | **1.179 %** | PASS |
| `C3_dambreak` | nonlinear free-surface collapse, leading-edge trajectory | analytic+experimental | front position error vs Ritter, validity window, established front (tau >= 0.5) | 5.0 % | **93.831 %** | FAIL |

### 1.1 Rung C1 -- hydrostatic balance

Square tank 0.4 m x 0.4 m, `dx` = 0.02 m, 400 particles, `c0` = 20.0 m/s, hydrostatic initial masses: True, simulated 4.0 s (10257 steps, 128 pressure samples averaged over the last 25 % of the run).

| band | y [m] | particles | p SPH [Pa] | p = rho g h [Pa] | error [%] | fluctuation [%] | gated |
|---|---|---|---|---|---|---|---|
| 0 | 0.0232 | 46 | 3682.7 | 3696.0 | -0.361 | 4.2 | yes |
| 1 | 0.0726 | 54 | 3211.2 | 3211.5 | -0.009 | 4.7 | yes |
| 2 | 0.1255 | 51 | 2696.7 | 2693.0 | 0.138 | 5.0 | yes |
| 3 | 0.1754 | 50 | 2203.3 | 2203.7 | -0.019 | 5.7 | yes |
| 4 | 0.2270 | 53 | 1682.0 | 1697.2 | -0.894 | 5.7 | yes |
| 5 | 0.2763 | 46 | 1202.4 | 1213.6 | -0.924 | 9.0 | yes |
| 6 | 0.3216 | 44 | 764.7 | 768.8 | -0.527 | 8.3 | yes |
| 7 | 0.3697 | 56 | 276.4 | 297.1 | -6.956 | 10.0 | no |

* max interior band error: **0.924 %** (gate 1.0 %) -> PASS
* mean interior band error: 0.410 %
* instantaneous extreme over the averaging window: 9.6 % (the acoustic sloshing that the time mean removes)
* bands within 2h = 0.0520 m of the free surface are excluded from the gate: their summation density is truncated by the surface, so they are not a hydrostatic measurement. Excluded: [7]

![hydrostatic](../../artifacts/verification_ladder/fig1_hydrostatic_profile.png)

### 1.2 Rung C2 -- Taylor-Green vortex

Periodic box L = 1.0 m, `dx` = 0.025 m (1600 particles), k = 2 pi / L, U0 = 0.2 m/s, `c0` = 10.0 m/s (Mach 0.020), nu = 0.015915 m^2/s (Re = 2.0), 628 steps of dt = 0.00095541 s to t = 0.6 s.

The reference is *exact*, not numerical: for `u = U0 exp(-2 nu k^2 t) sin(kx) cos(ky)`, `v` its quadrature partner,
the convective term is a pure gradient (absorbed by the pressure) and `lap u = -2 k^2 u`, so `E(t) = E(0) exp(-4 nu k^2 t)` at every Reynolds number.

| quantity | value |
|---|---|
| analytic energy decay rate 4 nu k^2 | 2.51327 1/s |
| measured (log-linear fit, window 0.060-0.510 s) | 2.48365 1/s |
| fit quality R^2 | 0.999987 |
| E(t_end)/E(0) measured | 0.21486 |
| E(t_end)/E(0) analytic | 0.22136 |
| peak |u| ratio at t_end | 0.46575 |
| peak |u| ratio analytic | 0.47049 |
| viscous operator L2 error vs `-2 k^2 u` (pressure term held out) | 0.679 % |
| **error in the decay rate** | **1.179 %** (gate 3.0 %) -> PASS |

![tgv](../../artifacts/verification_ladder/fig2_tgv_energy_decay.png)

#### 1.2.1 Validity window in Mach number (regime study)

Same rung at `dx` = 0.05 m, `c0` = 10.0 m/s, nu = 0.015915 m^2/s, sweeping the peak velocity. The anchor is meaningful only while the `O(rho U0^2)` pressure field that balances the exact convective term stays above the lattice's summation-density noise.

| U0 [m/s] | Mach | decay rate [1/s] | analytic | error [%] | R^2 |
|---|---|---|---|---|---|
| 0.020 | 0.0020 | 2.4518 | 2.5133 | 2.45 | 1.000000 |
| 0.050 | 0.0050 | 2.4601 | 2.5133 | 2.11 | 0.999996 |
| 0.100 | 0.0100 | 2.4584 | 2.5133 | 2.18 | 0.999999 |
| 0.200 | 0.0200 | 2.4379 | 2.5133 | 3.00 | 0.999985 |
| 0.500 | 0.0500 | 6.8758 | 2.5133 | 173.58 | 0.945270 |
| 1.000 | 0.1000 | 8.5514 | 2.5133 | 240.25 | 0.993032 |

The plateau (2.1-3.0 %) is the viscous-operator error at this spacing, and the rung collapses between Mach 0.02 and 0.05. The gated case uses Mach 0.02; a Taylor-Green verification run at the library's usual Mach ~0.1 would report a 240 % error that says nothing about the viscosity model.

![mach](../../artifacts/verification_ladder/fig5_tgv_mach_window.png)

### 1.3 Rung C3 -- dam-break free surface

Column a = 0.2 m, H = 0.4 m (H/a = 2.0), tank 2.0 m, `dx` = 0.02 m, 200 fluid particles, `c0` = 25.0 m/s, free slip, no surface tension, t_end = 0.400 s.

* analytic front (Ritter, inside its validity window t < a/sqrt(gH) = 0.1010 s): max error **93.831 %** over 16 samples (gate 5.0 %) -> FAIL
* experimental cross-check (Martin & Moyce 1952, transcribed table): MAE 0.491 in Z/a, RMSE 0.557; worst point at tau = 3.842 (experiment 3.050, simulation 4.011). This is **not** the gate; see section 5.
* Ritter front speed delivered by the collapse: `dZ/dt / (2 c0)` = **0.314** (a slope, so it does not diverge at t = 0)
* refinement (dx 0.04 -> 0.02 m): experimental MAE 0.546 -> 0.491 Z/a (observed order 0.154), front-speed ratio 0.250 -> 0.314 (order 0.128). The offset is essentially resolution independent, i.e. it is not a discretisation error that refinement removes.
* mass drift 0.000000000000 kg, max Mach 0.126, max density compression 2.66 %, escaped particles 0.

![dambreak](../../artifacts/verification_ladder/fig3_dambreak_front.png)

## 2. Negative controls

| control | mechanism switched | metric | measured | verdict |
|---|---|---|---|---|
| `C4a_gravity_off` | gravity enters the pressure balance | max interior band pressure error vs rho g h | `{"error_pct": 100.0, "mean_band_error_pct": 99.99994001440733, "pressure_max_pa": 0.003476937999948859}` | PASS |
| `C4b_viscosity_off` | viscosity is the only dissipation in the rung | measured energy decay rate / analytic 4 nu k^2 | `{"analytic_rate": 2.5132741228718345, "numerical_dissipation_floor_pct": 0.1658889844729609, "rate": 0.004169244919453802, "ratio": 0.0016588898447296092}` | PASS |
| `C4c_surface_tension` | capillary force on the free surface (CSF model) | leading-edge transverse spread and detached-particle count | `{"front_response_pct": 0.011211110330887482, "sigma_off": {"detached_fraction": 0.0, "detached_particles": 0, "front_filtered": 0.39555275440216064, "front_raw": 0.39555275440216064, "leader_mean_y": 0.06488215178251266,` | INCONCLUSIVE |

![controls](../../artifacts/verification_ladder/fig4_negative_controls.png)

## 3. Error determination (explicit)

| quantity | reference | measured error | threshold | within threshold? |
|---|---|---|---|---|---|
| max interior band pressure error vs rho g h | analytic | 0.924 % | 1.0 % | yes |
| energy decay rate vs analytic 4 nu k^2 | analytic | 1.179 % | 3.0 % | yes |
| front position error vs Ritter, validity window, established front (tau >= 0.5) | analytic+experimental | 93.831 % | 5.0 % | **no** |

Summary: 2/3 rungs pass, 2/3 controls pass, all_passed = **False**, honest failures = ['C3_dambreak', 'C4c_surface_tension'].

### 3.1 Campaign acceptance criteria, one by one

| criterion | target | measured | verdict |
|---|---|---|---|
| C1 max hydrostatic pressure error < 1 % | 1.0 % | 0.924 % | PASS |
| C2 Taylor-Green energy decay rate error < 3 % | 3.0 % | 1.179 % | PASS |
| C3 dam-break free-surface position error < 5 % (Martin & Moyce) | 5.0 % | MAE 0.491 Z/a (49.1 % of a); Ritter 93.8 % | FAIL |
| C4 negative control (mechanism effectiveness) | control must move the metric | gravity off: PASS, viscosity off: PASS, surface tension off: INCONCLUSIVE | INCONCLUSIVE |
| C5 end-to-end JSON schema with all anchors | all anchors present, schema valid | schema 1.0, 3 anchors, 0 schema errors | PASS |

*C3: two references disagree with the column by 49 % (experiment, MAE 0.49 in Z/a) and 94 % (Ritter shallow-water front); the gap is a resolution-independent physical offset (H/a = 2 is outside shallow-water theory) and the bundled experimental table is audited WITHHELD as a digitisation, so the 5 % criterion cannot be met by refinement alone*

*C4: the campaign expects the dam-break jet to disperse with sigma = 0; at We ~ 550 the capillary force is dynamically negligible, so the measured response is 0.01 % and the test does not discriminate*

## 4. Reproduce

```
python benchmarks/verification_ladder/run.py
python benchmarks/verification_ladder/make_doc.py
pytest tests/test_vv_ladder.py -q
```

## 5. Limitations and open points

* Rung 2 is two-dimensional and periodic; the 3-D Taylor-Green vortex has no analytic solution and is not covered (it would need reference DNS data).
* Rung 3's experimental cross-check uses a transcribed table that the repository audit grades WITHHELD as a digitisation, so a disagreement with it cannot be attributed to the solver alone.
* The dam-break free-slip wall condition is not the no-slip condition of the Martin & Moyce tank.
* No turbulence model, no surface tension in the gated dam-break run, and no density diffusion; each is available in the library but would add unverified physics to a verification case.

