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
| y = 0.025 | 80 | 3215.3 | 3190.4 | +0.8 % |
| y = 0.075 | 80 | 2715.1 | 2697.9 | +0.4 % |
| y = 0.126 | 80 | 2191.2 | 2201.0 | -0.5 % |
| y = 0.175 | 80 | 1684.6 | 1708.6 | -1.4 % |
| y = 0.224 | 80 | 1206.4 | 1227.6 | -1.7 % |
| y = 0.275 | 90 | 703.1 | 715.6 | -1.7 % |
| y = 0.324 (free surface) | 70 | 168.9 | 235.9 | -28.4 % |

- **Interior mean absolute relative error: 1.09 %** (gate: 5 %)
- Free-surface drop after 4 s: 3.5 % of the pool depth (gate: 5 %)
- Residual maximum speed: 0.044 m/s

(These numbers are from version 0.4.0, which corrected the missing factor 2 in
the Morris viscous Laplacian. The solver therefore applies twice the viscous
damping that 0.2.0 applied at the same nominal `viscosity`; the interior error
moved from 0.63 % to 1.09 %, still far inside the gate.)
- Particle masses are constants, so total mass is conserved exactly.

Note that the naive `sum_j W m_j` density summation at a wall under-counts kernel support by
one half. That is what the wall particles repair: without them the pressure near the floor
comes out roughly 30 % low.

### Coordinate clamp (`--boundary clamp`, retained only as a regression baseline)

The pre-fix behaviour is kept behind `Config(boundary="clamp")` so the improvement can be
re-measured rather than asserted. With flat clamping the pool also develops coincident
particles and loses the hydrostatic profile; see `docs/review-2026-10.md`.

## Does a hydrostatic initial condition help?

`Config(hydrostatic_init=True)` assigns each pool particle the mass implied by the
analytical profile `rho(y) = rho0 (1 + rho0 g (H-y)/B)^(1/gamma)`, so the
discretised density field starts as a discrete equilibrium instead of uniform
`rho0`. Measured at `t = 0` on the impact case, the initial acceleration
deviation from hydrostatic balance drops from **g (the whole pool starts in free
fall) to 0.05 m/s²**, and the initial pressure profile error from **100 % to 5 %**.

It does **not** tighten B1:

| initialisation | initial mass | interior error | surface drop | t=0 profile error |
|---|---|---|---|---|
| uniform mass (default) | 350.00 kg | **1.09 %** | 3.51 % | 100 % |
| hydrostatic mass | 351.48 kg | 1.16 % | 2.33 % | 5 % |

Both pass the 5 % gate. The reason is that the residual error is not a startup
transient: a pool confined by free-slip walls rings in a sloshing mode whose
viscous damping time is `H^2 / nu ≈ 12 s`, so a 4 s run is still inside the same
oscillation regardless of how it was started. Time-averaging is what removes it.

The flag is therefore **off by default**. It is useful when a run must start from
a well-defined equilibrium (resolution studies, restart tests) and it halves the
free-surface band error (-40.7 % to -26.4 %), but it is not a fix for the slowly
damped sloshing mode and it changes the total mass by +0.4 % because a resting
column is compressed.

```bash
python benchmarks/hydrostatic/run.py --seconds 4 --hydrostatic-init
```

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
