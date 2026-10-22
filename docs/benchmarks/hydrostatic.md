# Benchmark: B1 static pool / hydrostatic equilibrium

**Purpose:** verify that the pressure field of a pool at rest matches the analytical
hydrostatic profile `p(y) = rho0 * g * (H - y)`, and that the boundary treatment does not
distort it.

**Status:** numerical verification, not physical validation.

Run it:

```bash
python benchmarks/hydrostatic/run.py --seconds 4 --boundary wall
python benchmarks/hydrostatic/run.py --seconds 4 --boundary clamp   # pre-fix baseline
```

The script prints a JSON report and exits non-zero when a gate fails.

## Reading the result

Two effects have to be separated, otherwise the numbers look worse than the solver is:

1. **Acoustic oscillation.** A lattice initialised at *uniform* density is not in discrete
   equilibrium, so the pool sloshes. Once velocities have decayed to a few cm/s the
   *instantaneous* pressure profile still oscillates by tens of percent. The reported
   profile is therefore a **time average over the second half of the run**.
2. **Free-surface band.** In the topmost band the density summation is truncated by the
   free surface, so SPH pressure is not comparable to a hydrostatic reference. That band is
   labelled `free_surface` and excluded from the interior gate; it is reported so the
   limitation stays visible instead of being hidden by band selection.

## Measured result

`dx = 0.025 m`, `H = 0.35 m`, `rho0 = 1000 kg/m^3`, `c0 = 20 m/s`, 560 fluid particles,
`dt = 4.88e-4 s` (acoustic Courant 0.25), 4 simulated seconds, time-averaged over the last
2 seconds.

### Wall particles (`--boundary wall`, the default)

| band | particles | p time-averaged (Pa) | p analytical (Pa) | error |
|---|---|---|---|---|
| y = 0.025 | 80 | 3217.3 | 3191.1 | +0.8 % |
| y = 0.075 | 80 | 2711.4 | 2697.9 | +0.5 % |
| y = 0.126 | 80 | 2206.7 | 2201.1 | +0.3 % |
| y = 0.176 | 80 | 1703.2 | 1708.5 | -0.3 % |
| y = 0.225 | 80 | 1219.3 | 1227.6 | -0.7 % |
| y = 0.278 | 90 | 717.8 | 709.1 | +1.2 % |
| y = 0.326 (free surface) | 70 | 139.9 | 235.9 | -40.7 % |

- **Interior mean absolute relative error: 0.63 %** (gate: 5 %)
- Free-surface drop after 4 s: 2.6 % of the pool depth (gate: 5 %)
- Residual maximum speed: 0.055 m/s
- Particle masses are constants, so total mass is conserved exactly.

Note that the naive `sum_j W m_j` density summation at a wall under-counts kernel support by
one half. That is what the wall particles repair: without them the pressure near the floor
comes out roughly 30 % low.

### Coordinate clamp (`--boundary clamp`, retained only as a regression baseline)

The pre-fix behaviour is kept behind `Config(boundary="clamp")` so the improvement can be
re-measured rather than asserted. With flat clamping the pool also develops coincident
particles and loses the hydrostatic profile; see `docs/review-2026-10.md`.

## What this benchmark does *not* show

- It is not evidence that free-surface flows are accurate.
- It says nothing about wave impact, energy conservation, or splash morphology.
- Wall friction is free slip, so tangential momentum is not validated.

## Next steps for B1

1. Repeat at three resolutions and report the observed error decay instead of a single point.
2. Sweep `h/dx` independently of `dx` to separate kernel-width error from resolution error.
3. That is only honest if the initialisation is brought closer to equilibrium first
   (a hydrostatic density profile or a damped relaxation phase), otherwise each run
   averages a different transient.
4. Add a wall-force diagnostic so the boundary model can be checked against the integrated
   weight of the column, not only against local pressure.
