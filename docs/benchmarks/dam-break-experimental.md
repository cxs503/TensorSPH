# Benchmark: B2 dam-break experimental validation

**Purpose:** compare the TensorSPH dam-break front trajectory against the
classical Martin & Moyce (1952) experiment, establishing **physical validation**
for the first time.

**Status:** `physical_validation` now **evaluated** (previously `not_evaluated`).
The experimental data is bundled with full provenance.

Run it:

```bash
# Full validation with plot
python benchmarks/dam_break/experimental_validation.py --sizes 0.02,0.01 --seconds 1.5 --plot

# Quick check at coarse resolution
python benchmarks/dam_break/experimental_validation.py --sizes 0.04 --seconds 0.5 --plot

# With adaptive time step
python benchmarks/dam_break/experimental_validation.py --sizes 0.02 --seconds 1.5 --adaptive-dt --plot
```

## Experimental reference

| quantity | value |
|---|---|
| source | Martin, J.C. & Moyce, W.J. (1952). *Phil. Trans. R. Soc. A* **244**(882), 312-324 |
| DOI | 10.1098/rsta.1952.0006 |
| digitization | From open-access review papers (Koshizuka & Oka 1996; Aureli et al. 2023) |
| data file | `benchmarks/dam_break/martin_moyce_1952.json` |
| uncertainty | ±0.05 in Z/a (estimated from figure resolution) |
| num points | 20 |

### Non-dimensionalization

Martin & Moyce measure the front position Z from the *initial column edge*
(x = a), so Z = 0 at t = 0. The simulation reports the absolute front position
x_max, so the conversion is:

```
Z/a = (x_max - a) / a
tau = t * sqrt(2g/a)
```

where a = 0.2 m is the column width and g = 9.81 m/s².

## Simulation setup

| quantity | value |
|---|---|
| column width `a` | 0.2 m |
| column height | 0.4 m (= 2a, matching Martin & Moyce aspect ratio) |
| tank | 2.0 × 0.75 m, static wall particles, free slip |
| water | `rho0 = 1000 kg/m³`, `c0 = 25 m/s`, `gamma = 7`, `nu = 0.01` |
| `dt` | from acoustic Courant limit 0.3 |
| initial state | at rest, masses from hydrostatic profile |

## Results

### Coarse resolution (dx = 0.04 m, N = 50 particles)

| metric | value |
|---|---|
| simulated time | 0.3 s (tau ≈ 2.97) |
| steps | 481 |
| max speed | 3.3 m/s |
| Mach number | 0.13 |
| peak compression | 2.3 % |
| **exp MAE (Z/a)** | **0.60** |
| exp RMSE | 0.78 |
| points compared | 14 / 20 |
| within uncertainty | 0 % |

**Analysis:** At 50 particles, the front trajectory is significantly behind the
experimental data. This is expected: SPH at coarse resolution has substantial
numerical diffusion, and the free-surface evolution is poorly resolved. The
existing convergence study (dam-break-2d.md) shows observed order 1.11, so
refinement should reduce the error.

### Expected behavior at finer resolution

From the existing three-level ladder (dam-break-2d.md):

| dx | particles | observed order | front L2 error |
|---|---|---|---|
| 0.04 | 50 | — | — |
| 0.02 | 200 | 1.11 | 4.83 % |
| 0.01 | 800 | 1.11 | 2.19 % |

Extrapolating the convergence trend, the experimental MAE should decrease
approximately as dx^1.11. At dx = 0.01, we expect MAE ≈ 0.15–0.20, which
would be within the experimental uncertainty band for most points.

## Gates

| gate | metric | measured | limit | result |
|---|---|---|---|---|
| `experimental_comparison` | MAE in Z/a vs Martin & Moyce | 0.60 (dx=0.04) | 0.15 | **fail** (coarse) |
| `mass_conservation` | mass drift | 0.0 kg | 0 | pass |
| `bounded_flow` | particles escaping | 0 | 0 | pass |
| `mach_number` | max \|v\| / c0 | 0.13 | 0.25 | pass |
| `density_compression` | max(rho/rho0 - 1) | 2.3 % | 10 % | pass |
| `ritter_comparison` | mean rel error vs Ritter | 0.51 | — | informational |

## Error sources

1. **Numerical diffusion:** SPH at coarse resolution smears the front, reducing
   the advance speed. This is the dominant error source at dx = 0.04.

2. **Wall friction:** The experiment had some wall friction; the simulation uses
   free slip. This would cause the simulation to advance slightly faster than
   the experiment, opposite to the observed trend.

3. **Weakly-compressible approximation:** c0 = 25 m/s gives Mach ≈ 0.13–0.20.
   The compressibility study (dam-break-2d.md) shows the front moves by less
   than 0.04 % of a between c0 = 25 and 60 m/s, so this is negligible.

4. **Initial condition:** The hydrostatic initialization gives a slightly
   compressed column, which may affect the early-time dynamics.

5. **Experimental uncertainty:** The digitized data has ±0.05 uncertainty in
   Z/a, estimated from figure resolution.

## Comparison with Ritter solution

The Ritter shallow-water solution (Z/a = tau) assumes a semi-infinite reservoir
on a dry bed. The Martin & Moyce experiment has a finite reservoir (aspect
ratio 2:1), so the Ritter solution overpredicts the front position. Both the
simulation and the experiment fall below the Ritter curve, as expected.

## Correction (measured, m2502): the resolution prediction is falsified

The section above predicts that refinement "should" bring the error inside the
experimental band (MAE 0.15-0.20 at `dx` = 0.01 m). That prediction has now been
measured on a three-rung ladder (`dx` = 0.04 / 0.02 / 0.01 m, 50 / 200 / 800
fluid particles, identical setup):

| `dx` (m) | exp MAE (`Z/a`) | RMSE | within uncertainty |
|---|---|---|---|
| 0.04 | 0.5988 | 0.6766 | 5 % |
| 0.02 | 0.5774 | 0.6817 | 5 % |
| 0.01 | 0.5728 | 0.6994 | 0 % |

The mean absolute error is flat (observed order 0.05 / 0.01, i.e. none) while the
RMSE *grows* with resolution: the residual is a resolution-independent bias, not
numerical diffusion of the front. The 15 % gate therefore stays **fail**, and the
`physical_validation` status of case B2 is *fails* at every resolution measured.
The bias the ladder leaves unexplained is separated into bed resistance and
time origin in [`dam-break-friction-attribution.md`](dam-break-friction-attribution.md)
(`benchmarks/dam_break/friction_sweep.py`, 10 runs over two spacings and five
wall-friction blends).

Machine-readable artifact: `benchmarks/dam_break/resolution_ladder.json`;
full write-up (generated from the artifact, no hand-copied numbers):
[`dam-break-resolution-ladder.md`](dam-break-resolution-ladder.md); figure:
`benchmarks/dam_break/resolution_ladder.png`.

## Next steps

1. **Run at finer resolution:** dx = 0.01 (800 particles) should give MAE
   within the experimental uncertainty band. This requires ~2 minutes of
   compute time.

2. **Wall friction study:** Compare free-slip vs no-slip walls to quantify
   the effect on the front trajectory.

3. **Density diffusion:** Test the Antuono et al. (2012) diffusive term to
   see if it improves the front trajectory.

4. **Adaptive time step:** The adaptive dt option caps the true Courant number
   at 0.300, which may improve accuracy at the cost of ~23 % more steps.

## Conclusion

The experimental validation infrastructure is now in place:
- ✅ Experimental data bundled with full provenance
- ✅ Non-dimensionalization matches Martin & Moyce convention
- ✅ Comparison metrics (MAE, RMSE, within-uncertainty fraction)
- ✅ Comparison plot generation
- ✅ Gate framework for pass/fail criteria

At coarse resolution (dx = 0.04), the error is large (MAE = 0.60), but this
is consistent with the known convergence behavior of SPH. The existing
convergence study shows observed order 1.11, so refinement to dx = 0.01
should bring the error within the experimental uncertainty band.

The `physical_validation` gate can now be changed from `not_evaluated` to
`evaluated` (with the caveat that finer resolution is needed for quantitative
agreement).
