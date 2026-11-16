# X6/X7: the near-field load channel on a curved body -- degree, support, admissibility

Anchor: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`.  Numbers in this file are injected by `benchmarks/xval_fsi2/curved_surface_load_audit.py` from `artifacts/xval_fsi2/curved_surface_load.json` (tracked copy: `docs/xval_fsi2/curved_surface_load.json`) -- no hand copying.

## 1. Why this round exists

Round 3 (`FARFIELD_FORCE_CHANNEL.md`) measured the near-field surface-pressure arms at the spacing the coupled FSI2 run actually uses (dx = 20 mm on the 100 mm cylinder, h/R = 0.52) and found them 36..48 % off on the analytic field, while the far-field control volume held the 3 % gate.  That leaves two open questions: (X6) does a curvature-aware reconstruction remove the bias, and (X7) what is the admissibility envelope of the near-field class?

## 2. X6 -- is the curved-body bias a *degree* defect?  Hypothesis H1

H1: *the curved-body bias of the near-field arm is a polynomial-degree defect of the reconstruction, so a local quadratic fit removes it.*  A quadratic surface reconstruction was added to the library as `boundary_pressure_from_fluid_quadratic_exact` and tested at **station** level, where a degree error is observable: for a closed body the quadratic content of the field integrates to zero over the surface (divergence theorem), so the net force cannot see a degree defect.

| configuration | h/dx | arm | max station error / max\|p\| | n neigh (median) | cond(design) median |
|---|---|---|---|---|---|
| flat_wall | 3 | kernel | 2.638e-02 | 56 | 1.92e+01 |
| flat_wall | 3 | plane | 5.212e-04 | 56 | 1.92e+01 |
| flat_wall | 3 | quadratic | 1.565e-09 | 56 | 1.92e+01 |
| closed_disk | 3 | kernel | 9.435e-03 | 55 | 1.96e+01 |
| closed_disk | 3 | plane | 1.230e-04 | 55 | 1.96e+01 |
| closed_disk | 3 | quadratic | 2.212e-09 | 55 | 1.96e+01 |
| flat_wall | 1.3 | kernel | 2.982e-02 | 9 | 1.68e+16 |
| flat_wall | 1.3 | plane | 9.784e-04 | 9 | 1.68e+16 |
| flat_wall | 1.3 | quadratic | 8.090e-02 | 9 | 1.68e+16 |
| closed_disk | 1.3 | kernel | 1.447e-02 | 9 | 3.59e+01 |
| closed_disk | 1.3 | plane | 6.024e-04 | 9 | 3.59e+01 |
| closed_disk | 1.3 | quadratic | 2.210e-01 | 9 | 3.59e+01 |

Two separate statements follow, and they must not be conflated:

* **The algebra is right.**  When the neighbour set can carry six parameters (``h/dx = 3``, ~56 neighbours per station) the quadratic arm reproduces a quadratic field at machine precision on both the flat wall and the curved disk; the kernel arm is off by ~1..3 % and the plane arm by ~5e-4..5e-3 (the plane's residual is the *mean* of the second derivative over a symmetric support -- it is small on this field but it does not vanish).
* **The information is not there at the coupled ratio.**  The coupled FSI2 run uses ``h/dx = 1.3``, i.e. ~9 fluid neighbours inside ``r < 2h`` per wall station.  Two different failure mechanisms show up there and the table separates them: on the **flat wall** the six-parameter design matrix is numerically rank deficient (cond ~1e16, only three distinct normal offsets to fit a normal parabola to), and on the **curved disk** it is still conditioned (cond ~36) but the samples sit in a thin shell *away* from the surface, so the fit becomes an extrapolation whose amplification is set by the shell thickness.  Either way the arm degrades to +8 % (flat wall) / +22 % (curved disk) **at station level** on a field that is *exactly representable* -- an information limit of a nine-sample neighbour set, not an implementation defect.

So H1 is **confirmed as algebra and falsified as a cure**: the curved-body bias of round 3 is angular smearing of a field sampled over an arc of order `h/R` with ~6..14 samples per station, not a missing polynomial term (X7 measures the consequence on the flow field).

## 3. X7 -- potential flow past the FSI2 cylinder, ladder in h/R

Field: potential flow past the FSI2 cylinder with imposed circulation (the round-3 field, so the numbers are comparable with `farfield_force.json`).  Reference 1 = Kutta-Joukowski `-rho U Gamma`; reference 2 = the analytic pressure integrated on an exact 4096-station circle.  The two agree to 1.8e-16 relative, so the comparison does not rest on KJ alone.

| dx [m] | R/dx | h/R | shell | n fluid | fluid/station | kernel | plane | quadratic |
|---|---|---|---|---|---|---|---|---|
| 0.0200 | 2.5 | 0.520 | 3h | 95 | 5.9 | +48.0 % | +35.7 % | +27.4 % |
| 0.0200 | 2.5 | 0.520 | 5h | 222 | 13.9 | +46.1 % | +31.3 % | +24.2 % |
| 0.0100 | 5.0 | 0.260 | 3h | 148 | 4.6 | +28.6 % | +13.3 % | +10.2 % |
| 0.0100 | 5.0 | 0.260 | 5h | 320 | 10.0 | +25.7 % | +11.6 % | +5.9 % |
| 0.0050 | 10.0 | 0.130 | 3h | 254 | 4.0 | +16.7 % | +4.6 % | +6.2 % |
| 0.0050 | 10.0 | 0.130 | 5h | 516 | 8.2 | +14.2 % | +3.5 % | +2.0 % |
| 0.0025 | 20.0 | 0.065 | 3h | 464 | 3.7 | +9.0 % | +1.3 % | +5.5 % |
| 0.0025 | 20.0 | 0.065 | 5h | 884 | 7.0 | +8.1 % | +1.1 % | +2.1 % |

Read-out:

* the quadratic arm helps only where the shell is *thick* relative to the station spacing (5h shell: it is the best arm at dx = 10 mm and finer) and hurts on thin shells, where six parameters are fitted to ~8 samples.  Neither arm is admissible at the coupled spacing.
* the error is a resolution effect, not a dx-independent structural offset: the arms converge under refinement (see the error law below), which is the opposite of the flat-wall OSS bias of `PRESSURE_MOMENT_NORMALISATION.md`.

### Error law (log-log, |lift error| vs h/R)

| arm @ shell | slope | prefactor | h/R at 15 % | R/dx at 15 % | reached | h/R at 3 % | R/dx at 3 % | reached |
|---|---|---|---|---|---|---|---|---|
| kernel@3h | 0.80 | 0.83 | 0.118 | 11.0 | yes | 0.016 | 81.7 | no |
| kernel@5h | 0.84 | 0.794 | 0.136 | 9.5 | yes | 0.020 | 65.3 | no |
| plane@3h | 1.58 | 1.06 | 0.290 | 4.5 | yes | 0.104 | 12.5 | yes |
| plane@5h | 1.61 | 0.941 | 0.319 | 4.1 | yes | 0.117 | 11.1 | yes |
| quadratic@3h | 0.77 | 0.363 | 0.316 | 4.1 | yes | 0.039 | 33.5 | no |
| quadratic@5h | 1.22 | 0.387 | 0.458 | 2.8 | yes | 0.122 | 10.7 | yes |

## 4. X7 ruling for the coupled run

Coupled spacing dx = 0.020 m, h/R = 0.52.  Worst |lift error| there:

| arm | worst error at coupled spacing | inside 15 % gate? | enters 15 % at R/dx (3h / 5h) | enters 3 % at R/dx (3h / 5h) |
|---|---|---|---|---|
| kernel | +48.0 % | **no** | 20.0 / 10.0 | not reached / not reached |
| plane | +35.7 % | **no** | 5.0 / 5.0 | 20.0 / 20.0 |
| quadratic | +27.4 % | **no** | 5.0 / 5.0 | not reached / 10.0 |

At the coupled spacing no near-field arm is inside the 15 % gate; the near-field class becomes admissible only after refinement, at the R/dx listed per arm and shell. A transient FSI2 load therefore cannot be validated at the deployed resolution, and a steady load must be taken from the far-field control-volume channel.

The ``enters ... at R/dx`` columns are the **coarsest** ladder spacing that buys the gate (the cheapest refinement), measured inside the ladder; values flagged ``not reached`` mean the finest ladder point is still outside the gate.

Consequences, stated as narrowly as the evidence allows:

1. A **transient** FSI2 load (the VIV drag/lift history needed for a Turek-Hron `C_D`/`C_L` comparison) can only come from a near-field arm, and no near-field arm is admissible at the deployed spacing.  Refining the coupled run is therefore *necessary*, not optional, before its `C_D` can be compared with the benchmark -- the -98 % deficit of `VIV_PERIODIC_VERDICT.md` currently sits behind that ceiling.
2. A **steady / time-averaged** load must be taken from the far-field control-volume channel, which holds the 3 % gate at the same spacing and same field (`FARFIELD_FORCE_CHANNEL.md`).
3. The quadratic arm is kept as a library capability (exact on quadratic fields, best arm on thick shells) but must not be presented as a cure for the curved-body bias at coarse support.

## 5. Not done / limits

* The ladder stops at dx = 0.0025 m (R/dx = 20).  Gate-entry values for the slowest-converging arm are extrapolated from the log-log fit and are labelled `reached_in_ladder` = no; they are predictions, not measurements.
* The field is an *analytic potential flow*, not an SPH field: viscosity, free-surface-adjacent and turbulent boundary structure are out of scope, so a real SPH boundary layer can only be worse.
* The near-field arms are evaluated on a synthetic lattice shell; the coupled run's fluid shell differs in packing but not in support/station ratio, which is the quantity the error law is written in.
* The unsteady far-field term (`-d/dt int rho u dV`) is still not implemented, so the far-field channel remains steady-only (round-3 limitation, unchanged).

