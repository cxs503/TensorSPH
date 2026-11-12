# CASE-CAPILLARY-WAVE: capillary-wave dispersion (TensorSPH)

**Status: PASS** (tolerance 10 %)

## Reference

* formula: `omega^2 = sigma*k**3/(rho1*coth(k*h1) + rho2*coth(k*h2))`
* source: Lamb, H. (1932). Hydrodynamics, 6th ed., Cambridge University Press, Art. 267 (capillary waves, deep water, inviscid: omega^2 = T k^3 / rho).
* evaluator: mpmath at dps=50 (no digit transcribed)
* `omega_ref` = 47.218370 rad/s, `T_ref` = 1.330665e-01 s, `c_ref` = 0.150301 m/s
* reductions (Lamb Art. 267, finite-depth at g=0): {'two_layer_deep_rho2_zero_vs_lamb_art267': 0.0, 'two_layer_rho2_zero_vs_finite_depth_gc': 0.0, 'all_below_1e-12': True}

## Parameter table

| quantity | value |
|---|---|
| `wavelength_m` | 0.02 |
| `sigma_N_m` | 0.072 |
| `rho_liquid_kg_m3` | 1000.0 |
| `rho_gas_kg_m3` | 1.2 |
| `depth_fluid_factor` | 0.8 |
| `depth_gas_factor` | 0.5 |
| `amplitude_fraction` | 0.02 |
| `h_factor` | 1.3 |
| `curvature_stencil` | 4.0 |
| `depth_fluid_m` | 0.016 |
| `depth_gas_m` | 0.01 |

## Reference vs measured

| curvature | ppw | dx [mm] | omega_ref [rad/s] | omega_got [rad/s] | err [%] | stiffness ratio +- spread | delta_s integral | verdict |
|---|---|---|---|---|---|---|---|---|
| fit_quadratic | 16 | 1.250 | 47.218370 | 35.757073 | -24.27 | 0.5763 +- 0.0802 | 1.0115 | FAIL |
| fit_quadratic | 24 | 0.833 | 47.218370 | 36.729950 | -22.21 | 0.6059 +- 0.0447 | 1.0081 | FAIL |
| fit_quadratic | 32 | 0.625 | 47.218370 | 38.774200 | -17.88 | 0.6807 +- 0.1315 | 1.0092 | FAIL |
| fit_quadratic | 48 | 0.417 | 47.218370 | 38.224034 | -19.05 | 0.6557 +- 0.0315 | 1.0081 | FAIL |
| mls | 16 | 1.250 | 47.218370 | 46.045062 | -2.48 | 0.9551 +- 0.1254 | 1.0115 | PASS |
| mls | 24 | 0.833 | 47.218370 | 45.224510 | -4.22 | 0.9186 +- 0.0674 | 1.0081 | PASS |
| mls | 32 | 0.625 | 47.218370 | 47.112077 | -0.23 | 1.0049 +- 0.1932 | 1.0092 | PASS |
| mls | 48 | 0.417 | 47.218370 | 46.013341 | -2.55 | 0.9501 +- 0.0448 | 1.0081 | PASS |

## Gates

| gate | requirement | value | pass |
|---|---|---|---|
| A_design_point_within_tol | |omega_got/omega_ref - 1| < 10 % at ppw=32, curvature=mls | 0.22510905100956435 | True |
| B_resolution_ladder_within_tol | all ppw in (16, 24, 32, 48) within 10 % (a passing point that diverges under refinement is not evidence) | 4.222636246660369 | True |
| C_delta_s_integrates_to_one | sum_i delta_s,i V_i / L = 1 +- 5 % (Brackbill unit mass) | 1.0091557697045754 | True |
| D_sigma_linearity | stiffness ratio constant under sigma x (0.5, 1.0, 2.0) (spread < 5 %) | 0.0 | True |

## Sensitivity ladders

### amplitude

| amplitude_over_lambda | omega_got/omega_ref | stiffness ratio | err [%] |
|---|---|---|---|
| 0.01 | 0.6331 | 0.8016 | -36.69 |
| 0.02 | 0.9977 | 1.0049 | -0.23 |
| 0.05 | 0.9592 | 0.9211 | -4.08 |

### sigma

| sigma_over_ref | omega_got/omega_ref | stiffness ratio | err [%] |
|---|---|---|---|
| 0.5 | 0.9977 | 1.0049 | -0.23 |
| 1 | 0.9977 | 1.0049 | -0.23 |
| 2 | 0.9977 | 1.0049 | -0.23 |

wall clock: 10.7 s
