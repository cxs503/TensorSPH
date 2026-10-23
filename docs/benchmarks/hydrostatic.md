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

| band | particles | p time-averaged (Pa) | p analytical (Pa) | error | fluctuation |
|---|---|---|---|---|---|
| y = 0.025 | 80 | 3166.0 | 3191.5 | -0.8 % | 6.1 % |
| y = 0.075 | 82 | 2678.3 | 2698.4 | -0.7 % | 6.7 % |
| y = 0.126 | 82 | 2173.7 | 2200.3 | -1.2 % | 8.1 % |
| y = 0.175 | 76 | 1699.3 | 1719.9 | -1.2 % | 8.5 % |
| y = 0.223 | 80 | 1218.7 | 1243.0 | -2.0 % | 9.5 % |
| y = 0.277 | 93 | 714.7 | 720.5 | -0.8 % | 15.2 % |
| y = 0.325 (free surface) | 67 | 169.2 | 245.3 | -31.0 % | 23.7 % |

- **Interior mean absolute relative error: 1.12 %** (gate: 5 %)
- Free-surface drop after 4 s: 3.15 % of the pool depth (gate: 5 %)
- Residual maximum speed: 0.013 m/s
- Fluctuation is the standard deviation of the pressure over the same 2 s
  averaging window, relative to the analytical pressure. A time average alone
  hides the ringing that a free surface produces, so both are reported.

Before 0.7.0 this table read 1.09 % interior, 3.5 % surface drop and 0.044 m/s
residual, with a fluctuation of 52–109 % per band. The difference is one term of
the wall pressure extrapolation: `p_w` was the kernel-weighted *average* of the
neighbouring fluid pressure, omitting `rho_f (g - a_w).(r_w - r_f) W` from Adami
et al. (2012) eq. (11). Without it the fluid next to the floor is
under-supported, the pool rings, and the time average still looks fine — the
fluctuation column is what makes the defect visible.
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

It tightens B1 once the wall pressure is right:

| initialisation | initial mass | interior error | surface drop | t=0 profile error |
|---|---|---|---|---|
| uniform mass (default) | 350.00 kg | 1.12 % | 3.15 % | 100 % |
| hydrostatic mass | 351.48 kg | **0.62 %** | 2.29 % | 5 % |

Both pass the 5 % gate. Before 0.7.0 the two rows read 1.09 % and 1.16 %, and
this section concluded that the initialisation made no difference, because the
residual error was a sloshing mode with a viscous damping time of
`H^2 / nu ≈ 12 s`: a 4 s run is still inside the same oscillation regardless of
how it was started. With the wall body-force term in place the pool rings far
less (residual speed 0.013 m/s instead of 0.044) and starting from a discrete
equilibrium now halves the interior error. The earlier "no difference" was
measured on a wall model that was itself injecting the imbalance.

The flag is still **off by default**, for the same reason as before: it changes
the total mass by +0.4 %, because a resting column is compressed. It is now the
right choice whenever a resolution study or a restart needs a well-defined
equilibrium to start from.

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
