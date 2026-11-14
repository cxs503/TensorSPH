# SPHinXsys `test_2d_fsi2` reference records (verbatim archive)

Anchor: `XVAL-SPH-FSI2-BENCHMARK-AND-FROM-FLUID` — reference arm of the FSI2
cross-validation.  These are the *only* direct numbers the open-source
implementation (`Xiangyu-Hu/SPHinXsys`) publishes for its `test_2d_fsi2`, so
anything that claims to compare against "the SPHinXsys fsi2 result" has to be
judged against these files.  They are archived here, unmodified, so the numbers
in `docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md` are reproducible without
network access.

## Provenance

| field | value |
|---|---|
| upstream | `Xiangyu-Hu/SPHinXsys`, branch `master` |
| retrieved | 2026-11-14, GitHub contents API (`/repos/Xiangyu-Hu/SPHinXsys/contents/...`) |
| upstream dir | `tests/2d_examples/test_2d_fsi2/regression_test_tool/` |
| licence | Apache-2.0 (SPHinXsys); files are kept verbatim and unmodified |

| file | upstream blob sha (first 8) | sha256 |
|---|---|---|
| `BeamObserver_Position_Run_0_result.xml` | `e466fad9` | `d3a2b81c3594275c27b51fb987944424f6a44e5a04abc827604b1aa20cc57a86` |
| `BeamObserver_Position_Run_4_result.xml` | `5ca5d281` | `4dd9ea44f58b02bcdd1121d16d26e08379b936827a03a94ac744d2add3bd9083` |
| `BeamObserver_Position_Run_9_result.xml` | `70d1e6f9` | `707177d9e3e08651c3343fa0f9867dc07c3315ada8c72ac64319d8748232074d` |
| `InsertedBody_TotalViscousForceFromFluid_time_averaged_mean_variance.xml` | `1999cc23` | `45e0ca317aa287f044270f9908d22e133ed92c9d6dcdba13cb1c484499acaa3d` |

The case set-up the records were produced with is `tests/2d_examples/test_2d_fsi2/fsi2.h`
+ `fsi2.cpp`; the parameter block that matters for the audit is reproduced below
with upstream line numbers (values only, so the parameters can be checked
without shipping the whole C++ file).

```text
fsi2.h:16   Real DL = 11.0;                    // channel length  (scaled units)
fsi2.h:17   Real DH = 4.1;                     // channel height  (scaled units)
fsi2.h:18   Real global_resolution = 0.1;
fsi2.h:22   Real insert_circle_radius = 0.5;   // D = 1.0  -> length scale 10 vs the benchmark's D = 0.1 m
fsi2.h:23   Real bh = 0.4 * insert_circle_radius;   // beam thickness 0.2 = 10 x 0.02 m
fsi2.h:24   Real bl = 7.0 * insert_circle_radius;   // beam length 3.5 = 10 x 0.35 m
fsi2.h:28   Real rho0_f = 1.0;
fsi2.h:29   Real U_f = 1.0;
fsi2.h:30   Real c_f = 10.0 * U_f;
fsi2.h:31   Real Re = 100.0;
fsi2.h:32   Real mu_f = rho0_f * U_f * (2.0 * insert_circle_radius) / Re;   // = 0.01
fsi2.h:36   Real rho0_s = 10.0;                // solid/fluid density ratio 10  (benchmark: 1)
fsi2.h:37   Real poisson = 0.4;
fsi2.h:38   Real Ae = 1.4e3;                   // E / (rho_f U^2)  (benchmark FSI2: 5.6e3)
fsi2.h:39   Real Youngs_modulus = Ae * rho0_f * U_f * U_f;
fsi2.cpp:191 Real end_time = 200.0;            // scaled time (= 20 s physical)
fsi2.cpp:192 Real output_interval = end_time / 200.0;   // = 1.0 scaled = 0.1 s
```

## Record format

`BeamObserver_Position_Run_*.xml` is the standard SPHinXsys regression record:
`Snapshot_Element/number_of_snapshot_for_local_result_` gives the number of
stored samples and `Result_Element/Particle_0` carries one `snapshot_<k>="~[x,y]"`
pair per sample, in **scaled length units** (x is ~6.0 and y ~2.0 for the beam
tip, i.e. 10x the physical 0.6 m / 0.20 m).

`InsertedBody_TotalViscousForceFromFluid_time_averaged_mean_variance.xml` carries
the time-averaged total viscous force on the insert body in **scaled force
units** (`rho' U'^2 L'`), plus its variance.

`src/tensorsph/fsi2_reference.py` is the loader/statistics/acceptance-gate
module for these files; `tests/test_fsi2_reference.py` pins the parsed values.