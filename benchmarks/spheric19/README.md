# SPHERIC Test 19 · dam break against a flexible plate

A measured rigid gate rises at x = 0.20 m. The water column runs downstream
and loads a separate flexible cantilever at x = 0.60 m. The tank is 0.80 m
long and the water depth in this comparison is 0.30 m. Marker 1 is the
highest marker. This fixture compares the displacement histories at initial
heights **87.5, 65 and 40 mm** in the order used by Figure 1.

## Reproduce the comparison

From the TensorSPH checkout with Python 3.11+:

```sh
python -m pip install "scipy>=1.11" -e ".[test]"
python -m pip install --no-deps "git+https://github.com/cxs503/TensorFEM.git@d281d065d08d88573965226cce3c978243541054"
python benchmarks/spheric19/run.py --dx .01 --output out/test19-10mm
python benchmarks/spheric19/run.py --dx .005 --output out/test19-5mm
python benchmarks/spheric19/run.py --dx .01 --courant .11 --output out/test19-half-step
python benchmarks/spheric19/run.py --dx .005 --tank-contact --output out/test19-contact
python benchmarks/spheric19/plot.py out/test19-10mm out/test19-5mm out/test19-contact --output out/test19.png
python -m pytest -q tests/test_spheric19.py
```

Each run writes per-step marker displacements in metres, physical time in
seconds, and a JSON record with raw and peak-normalized marker RMSE, Mach
number, maximum particle excursion through the tank floor/sides, and the
horizontal force balance over all walls and fluid particles. Scores use linear
interpolation in physical time at the experimental sample times, with no phase
alignment or amplitude fitting.

## Model and interface

The default fluid method uses the TensorSPH single-phase WCSPH density-summation solver and
its hydrostatic mass initialization. It models an open water surface, free-slip
walls and a single row of moving plate stations on the centreline. The gate
lift is piecewise linear with a final hold. The plate stiffness and inertia
use the physical 4 mm thickness, Young's modulus 3.5 MPa, density 1161.54
kg/m³, and unit span. Its finite-rotation beam reduction uses uniaxial bending
stiffness EI, without a plane-strain Poisson correction.

At each acoustic step the fluid reads the plate positions and velocities,
evaluates the actual wall reaction for every station and sends both force
components to TensorFEM. The structure subcycles this load and returns its
updated shape. The exchange is explicit and staggered. Wall-pressure
extrapolation uses gravity and omits prescribed or structural wall acceleration.
Gas entrapment, wall-normal thickness resolution and strong coupling iterations
are outside this implementation. The pressure-only baseline uses no tank-position clamping. The optional
contact comparison enforces unilateral, frictionless, inelastic contact on the
fixed tank planes after each trial step, records the additional tank impulse
and raw penetration, and leaves the top open. This impulse stays separate
from the forces applied to the elastic plate.

## Experimental data provenance

Reference: [SPHERIC Test 19](https://www.spheric-sph.org/tests/test-19), prepared
by P. Sun. The original measurements are due to Liao, Hu & Sueyoshi (2015),
*Free surface flow impacting on an elastic structure: Experiment versus
numerical simulation*. Sun, Le Touzé & Zhang (2019) studied the multiphase
case, DOI [10.1016/j.enganabound.2019.03.033](https://doi.org/10.1016/j.enganabound.2019.03.033).

- `gate_fig2.dat` uses the centres of 40 green measurement symbols in
  [Figure 2](https://static.wixstatic.com/media/c24cbb_2cf2d7b6c3934113981582aca79e7f45~mv2.png)
  plus the physical start (0,0). SHA-256 of the source PNG is
  `dce218589c7f8a13801e841f63a92ece5ff1f7c7fa07531cb158dcb3a3ffadd9`.
  In image pixels the axes were calibrated with x=62.2 at 0 s and a 648 px
  span for 0.14 s, y=351.1 at 0 m and a 331 px span for 0.7 m.
  An alternate two-column SI record can be supplied via `--gate`.
- `markers_h030_fig3.csv` uses 62 sampled black experimental circles from
  the right-hand panels of [Figure 3](https://static.wixstatic.com/media/c24cbb_34ace7615efb48cf815cbe8e712cfac8~mv2.png).
  Source PNG SHA-256 is `eda0e560c7035173cfc111aed27478ead2654e4022c749f1e413187a69ebabc1`.
  Coordinates were calibrated against the ticks of each panel, converted from
  dimensionless time and displacement using H=0.30 m and g=9.81 m/s². One
  image pixel corresponds to approximately 0.93 ms and 0.51 mm. The numbers
  are figure digitizations, not the original ASCII records or the coloured
  SPH simulation curves.

## Measured CPU results

All three runs use 8 finite elements, sound speed 60 m/s and end at 0.46 s.
Each error is the RMSE in millimetres; parentheses give the error divided by
the largest measured displacement magnitude among that marker's samples.
The CFL column is the maximum Courant number used to select the fluid step.

| Particle spacing | CFL | Marker 1 | Marker 2 | Marker 3 | Peak fluid Mach |
|---|---:|---:|---:|---:|---:|
| 10 mm | 0.22 | 17.10 (23.75%) | 7.59 (14.71%) | 2.76 (9.43%) | 0.075 |
| 5 mm | 0.22 | 13.93 (19.34%) | 5.78 (11.21%) | 2.36 (8.07%) | 0.136 |
| 10 mm | 0.11 | 17.66 (24.52%) | 7.57 (14.67%) | 3.56 (12.15%) | 0.125 |

![Computed marker displacements compared with Liao's experimental symbols](markers.png)

At 5 mm spacing the top, middle and lower markers reach half of their measured
peak magnitudes **9.05, 6.81 and 3.41 ms early**. Refining the spatial grid
reduces all three RMSEs in this pair of runs. Halving the Courant limit at
10 mm does not uniformly reduce error. Two spacings do not establish a
convergence order. The SPHERIC page cites a 1 mm particle spacing; that level
has not been run here. The fixed tank boundary also allows excursions. These
histories are a development baseline; the coupled model is not qualified for
load prediction.

The largest envelope excursion in the pressure-only 5 mm run is **62.47 mm
in x at y = 0.820 m**, above the 0.60 m rim. This maximum is not a measure
of solid-wall penetration. There is also a genuine floor crossing before
the main bending response: at t = 0.26858 s a particle first exceeded 0.25
particle spacings below the floor near x = 0.620 m. The output records both
the first such envelope excursion and the location of the largest one.

### Fixed-wall contact comparison

With normal contact enabled at 5 mm spacing, the three marker errors become 12.96 mm (18.00%), 5.35 mm (10.37%), 3.10 mm (10.60%). The total absolute contact impulse on the fluid is 13.03 N·s per metre of span. The maximum trial excursion is 0.079 mm and the maximum after contact is 0.000 mm. The wall constraint changes the response at all markers. It does not establish experimental agreement. The [paired experimental and computed displacements](comparison.csv) include both spacings and the half-CFL control at their original experimental sample times.

## Delta-plus SPH comparison

Select Sun's single-resolution 2-D formulation via the fluid method switch.
The benchmark keeps the same gate history, plate geometry, FEM properties,
experimental data, sound speed of 60 m/s and end time of 0.46 s. The two
fluid methods differ in density evolution, kernel, stabilization and wall
treatment. This is a solver comparison. The shift has its own isolated
[operator and analytic free-drop checks](../delta_plus/README.md).

```sh
python benchmarks/spheric19/run.py --method delta-plus --dx .02 --output out/delta20
python benchmarks/spheric19/run.py --method delta-plus --dx .01 --output out/delta10
python benchmarks/spheric19/run.py --method delta-plus --dx .01 --courant .11 --output out/delta10-half
python benchmarks/spheric19/run.py --method delta-plus --dx .005 --output out/delta5
python benchmarks/spheric19/plot.py out/test19-5mm out/delta5 --output out/methods5.png
```

| Δx (mm) | CFL | Marker 1 RMSE (mm) | Marker 2 RMSE (mm) | Marker 3 RMSE (mm) | Peak Mach | Max density deviation |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.22 | 22.35 | 16.40 | 11.07 | 0.0312 | 0.529% |
| 10 | 0.22 | 14.70 | 8.61 | 6.03 | 0.0388 | 0.765% |
| 10 | 0.11 | 14.69 | 8.61 | 6.03 | 0.0389 | 0.768% |
| 5 | 0.22 | 7.95 | 4.91 | 4.06 | 0.0516 | 2.908% |

![Both fluid methods at 5 mm with the measured marker displacements](../delta_plus/test19_5mm.png)

At 10 mm the upper-marker RMSE drops from 17.10 to 14.70 mm. The middle
and lower errors rise from 7.59 to 8.61 mm and from 2.76 to 6.03 mm.
At this spacing halving the fluid CFL changes each RMSE by less than
0.01 mm. The computed half-rise times are late by 12.05, 16.16 and 14.95 ms
at CFL 0.22. The default WCSPH control produces identical per-step marker
histories to its previous 10 mm record.

At 5 mm the δ⁺-SPH errors normalized by the corresponding measured peaks are 11.04%, 9.52% and 13.87%. Signed half-rise timing errors (simulation minus experiment) are +7.48, +7.22 and +9.52 ms.

The JSON diagnostic checks peak fluid Mach ≤ 0.10 and maximum relative
density departure ≤ 1% at every δ⁺-SPH step. Exceedances retain their first
time and values. This regime check is independent of marker agreement.
The 5 mm run first crossed the density limit at t = 0.26432 s and reached
2.908% with peak Mach 0.0516. All three marker scores meet the 15%
repository tolerance, while the weak-compressibility gate fails.
All runs use pressure traction and omit the optional post-step tank contact.
[Machine-readable results](../delta_plus/spheric19_results.json) and
[all 62 paired experimental samples](comparison.csv) retain both methods.
