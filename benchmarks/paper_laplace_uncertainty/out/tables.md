# Laplace-law verification: machine-rendered tables

Runs aggregated: 23. Theory: Delta p = sigma/R (2-D, one principal curvature). Ensemble: hexagonal lattice phase offsets (0,0)/(0.5dx,0)/(0,0.5dy)/(0.5dx,0.5dy)/(0.25dx,0.75dy) with the drop centre fixed; one member adds 2%dx seeded jitter. Sampling: t = 1.0 t_sigma. Verdict rule: PASS: |e|+CI<3%; FAIL: |e|>=15%; else WITHHELD.

## Table 1a -- pressure jump, metric A (core vs interface band)

Metric A: Delta p = mean(p, r<0.5R) - mean(p, 0.8R<r<0.95R). Used as the primary metric because it samples the pressure *at the interface*, which is where Young-Laplace is asserted.

| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] | numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] | verdict | <3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 159 | 5 | 249.63 | 34.01 | 360.00 | -110.37 | 360.00 | -30.66 | 9.45 | FAIL | no |
| 2.50 | 8.0 | 227 | 5 | 102.73 | 13.44 | 360.00 | -257.27 | 360.00 | -71.46 | 3.73 | FAIL | no |
| 2.00 | 10.0 | 358 | 5 | 51.29 | 27.56 | 360.00 | -308.71 | 360.00 | -85.75 | 7.66 | FAIL | no |
| 1.50 | 13.3 | 640 | 5 | 41.07 | 25.19 | 360.00 | -318.93 | 360.00 | -88.59 | 7.00 | FAIL | no |
| 1.00 | 20.0 | 1447 | 3 | 8.95 | 28.82 | 360.00 | -351.05 | 360.00 | -97.51 | 8.00 | FAIL | no |

## Table 1b -- pressure jump, metric B (core vs free surface)

Metric B: Delta p = mean(p, r<0.5R) - mean(p, 0.95R<r<1.05R) (analytic p_out = 0). This is the metric a surface-tension test would like to use, but it is only valid if the interior pressure is uniform -- see Table 3 for the gradient that invalidates it.

| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] | numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] | verdict | <3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 159 | 5 | n/a | n/a | 360.00 | n/a | 360.00 | n/a | n/a | WITHHELD | no |
| 2.50 | 8.0 | 227 | 5 | 258.12 | 11.76 | 360.00 | -101.88 | 360.00 | -28.30 | 3.27 | FAIL | no |
| 2.00 | 10.0 | 358 | 5 | 317.34 | 13.67 | 360.00 | -42.66 | 360.00 | -11.85 | 3.80 | WITHHELD | no |
| 1.50 | 13.3 | 640 | 5 | 264.42 | 15.50 | 360.00 | -95.58 | 360.00 | -26.55 | 4.31 | FAIL | no |
| 1.00 | 20.0 | 1447 | 3 | 264.00 | 24.78 | 360.00 | -96.00 | 360.00 | -26.67 | 6.88 | FAIL | no |

## Table 1c -- pressure jump, metric C (force-implied)

Metric C: Delta p = |F_cut| / (2R), F_cut = sum_{x>0} m a_st,x (analytic |F_cut| = 2 sigma). Independent of any pressure measurement, so it isolates the force amplitude from the pressure transport.

| dx [mm] | R/dx | N | n | measured [Pa] | 95% CI [Pa] | theory [Pa] | numerator [Pa] | denominator [Pa] | rel. err [%] | 95% CI [%] | verdict | <3%? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 159 | 5 | 273.71 | 14.35 | 360.00 | -86.29 | 360.00 | -23.97 | 3.99 | FAIL | no |
| 2.50 | 8.0 | 227 | 5 | 270.31 | 9.45 | 360.00 | -89.69 | 360.00 | -24.92 | 2.63 | FAIL | no |
| 2.00 | 10.0 | 358 | 5 | 267.65 | 8.48 | 360.00 | -92.35 | 360.00 | -25.65 | 2.36 | FAIL | no |
| 1.50 | 13.3 | 640 | 5 | 256.40 | 6.23 | 360.00 | -103.60 | 360.00 | -28.78 | 1.73 | FAIL | no |
| 1.00 | 20.0 | 1447 | 3 | 270.71 | 26.06 | 360.00 | -89.29 | 360.00 | -24.80 | 7.24 | FAIL | no |

## Table 2 -- CSF curvature, three estimators vs analytic 1/R

| dx [mm] | R/dx | kappa exact [1/m] | MLS div n mean | rel. err [%] (95% CI) | verdict | quadratic fit mean | rel. err [%] (95% CI) | verdict | circle fit mean | rel. err [%] (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.00 | 6.7 | 50.00 | 75.51 | 51.01 (19.51) | FAIL | 54.48 | 8.97 (17.82) | WITHHELD | 100.98 | 101.96 (16.15) | FAIL |
| 2.50 | 8.0 | 50.00 | 101.86 | 103.73 (70.01) | FAIL | 52.28 | 4.57 (31.73) | WITHHELD | 149.79 | 199.58 (79.73) | FAIL |
| 2.00 | 10.0 | 50.00 | 127.31 | 154.62 (74.35) | FAIL | 52.54 | 5.09 (13.19) | WITHHELD | 163.04 | 226.09 (16.25) | FAIL |
| 1.50 | 13.3 | 50.00 | 162.07 | 224.14 (121.38) | FAIL | 53.47 | 6.94 (21.75) | WITHHELD | 222.68 | 345.35 (95.02) | FAIL |
| 1.00 | 20.0 | 50.00 | 255.52 | 411.04 (392.43) | FAIL | 55.97 | 11.94 (40.46) | WITHHELD | 391.38 | 682.75 (322.56) | FAIL |

## Table 3 -- independent checks (force magnitude, free surface, spurious current)

| dx [mm] | R/dx | F_cut vs -2 sigma [%] | p_shell / (sigma/R) [%] | max|v| / sqrt(sigma/(rho R)) | rho_core/rho0 |
|---|---|---|---|---|---|
| 3.00 | 6.7 | -23.97 | n/a | 0.056 | 1.0005 |
| 2.50 | 8.0 | -24.92 | 0.00 | 0.063 | 1.0003 |
| 2.00 | 10.0 | -25.65 | 0.00 | 0.052 | 1.0008 |
| 1.50 | 13.3 | -28.78 | 0.00 | 0.060 | 1.0007 |
| 1.00 | 20.0 | -24.80 | 9.00 | 0.058 | 1.0007 |

## Table 4 -- time dependence of the measured jump (metric A, t/t_sigma)

| dx [mm] | t=0.25 | t=0.5 | t=0.75 | t=1.0 |
|---|---|---|---|---|
| 3.00 | 31.96 ± 22.55 | -31.01 ± 16.38 | -24.59 ± 12.50 | -30.66 ± 9.45 |
| 2.50 | -30.76 ± 8.67 | -52.17 ± 7.53 | -68.01 ± 4.04 | -71.46 ± 3.73 |
| 2.00 | -80.74 ± 8.09 | -87.22 ± 6.81 | -84.57 ± 9.68 | -85.75 ± 7.66 |
| 1.50 | -100.33 ± 4.24 | -102.26 ± 3.71 | -95.65 ± 5.37 | -88.59 ± 7.00 |
| 1.00 | -118.06 ± 6.25 | -125.89 ± 5.01 | -110.50 ± 4.38 | -97.51 ± 8.00 |
