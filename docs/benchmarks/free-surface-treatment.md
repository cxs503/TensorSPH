# Free-surface treatment: pressure clamp versus density diffusion

**Purpose:** the roadmap's open P1 item is "free-surface tensile instability is
fixed by clamping, which is first-order: it discards all rarefaction and cannot
represent capillary effects". This document adds a second, physically motivated
treatment, measures it against the clamp on the benchmarks that already exist,
and records the decision that followed.

**Status:** numerical verification. No physical validation is claimed: neither
treatment has been compared against measured data.

Run the comparison:

```bash
# both treatments on all three observables, ~2 min on two CPU cores
python benchmarks/free_surface/compare.py --deltas 0,0.1

# the same knob on the existing benchmarks
python benchmarks/hydrostatic/run.py --seconds 4 --density-diffusion 0.1
python benchmarks/dam_break/run.py --sizes 0.02,0.01 --seconds 0.8 --density-diffusion 0.1
```

## The two treatments

**Pressure clamp** (`Config.density_diffusion = 0`, the default).
Density summation under-counts kernel support at a free surface, so `rho < rho0`
there. The clamped Tait law,

```
p = max(B ((rho/rho0)^gamma - 1), 0),        B = rho0 c0^2 / gamma
```

returns zero instead of a large tensile pressure. That removes the tensile
instability but also discards every bit of rarefaction, so a free surface
cannot sustain any negative pressure at all.

**Diffusive density term** (`Config.density_diffusion = delta > 0`).
The term of Antuono, Colagrossi & Marrone (2012), *Numerical diffusive terms in
weakly-compressible SPH schemes*, Comput. Phys. Commun. 183:2570–2580:

```
drho_i/dt = ... - 2 delta c0 h sum_j V_j (rho_i - rho_j) * -(r_ij . grad W_ij) / r_ij^2
```

It is a diffusion on the density field: a particle denser than its neighbours
loses density. `V_j = m_j/rho_j`, and the pair contribution is applied as `-c`
to `i` and `+c V_i/V_j` to `j`, so `sum_i V_i drho_i/dt = 0` exactly.

### What this is *not*

It is **not delta-SPH proper**. delta-SPH integrates the continuity equation and
adds the diffusive term to `drho/dt`; this solver obtains density by summation
every step, so the term here is applied to the summation density over `cfg.dt`:

```python
rho = summation_density(pos)
rho = rho + dt * diffusion_rate(rho)          # only when delta > 0
p   = tait_pressure(rho)
```

It is also **not a surface-tension model**. It changes how density is smoothed;
it does not create a capillary pressure jump, and no curvature term exists.

### Invariants (pinned by tests)

| invariant | test |
|---|---|
| a uniform density field produces exactly zero rate | `test_a_uniform_density_field_is_left_untouched` |
| the volume-weighted mean density is conserved | `test_the_volume_weighted_mean_density_is_conserved` |
| the term reduces density variance on a perturbed field | `test_the_term_reduces_density_variance` |
| the pair implementation matches the dense reference | `test_the_pair_implementation_matches_the_dense_reference` |
| `delta = 0` is bit-identical to not having the term | `test_zero_diffusion_does_not_touch_the_solution` |
| `delta` has its own explicit-step limit `0.125 h/(delta c0)` | `test_diffusion_has_its_own_step_limit` |
| only fluid-fluid pairs are diffused (wall density is extrapolated) | `test_the_term_only_sees_fluid_neighbours` |

## Measured

`python benchmarks/free_surface/compare.py --deltas 0,0.1`, two CPU cores.
Impact case at `dx = 0.025` to `t = 0.3 s`; hydrostatics at `dx = 0.025` over
4 s averaged over the second half (the B1 protocol); dam break at `dx = 0.02` to
`t = 0.8 s`.

These numbers were re-measured in 0.7.0, after the wall pressure extrapolation
was completed with the body-force term of Adami et al. (2012) eq. (11). The 0.6.0
measurement of the same script is kept below for comparison, because the wall fix
changed what the hydrostatic metric can see.

### 1. Hydrostatics (B1 protocol, 7 bands)

| delta | interior pressure error | pressure fluctuation | residual max &#124;v&#124; | wall clock |
|---|---|---|---|---|
| 0.0 | 1.119 % | 9.0 % | 0.013 m/s | 51 s |
| 0.1 | **1.057 %** | **6.8 %** | 0.013 m/s | 57 s |

The fluctuation falls by a quarter. That is the quantity this term is supposed to
address, and it now holds up: at 5, 6, 7 and 8 bands the fluctuation improves at
every band count (2.6 → 2.2 %, 3.0 → 2.4 %, 9.0 → 6.8 %, 3.3 → 2.7 % on a
separate 0.6 x 0.3 m pool, so the levels are not comparable across geometries but
the sign is).

The interior pressure error is **not** robust. At 7 bands it improves (1.119 →
1.057 %); at 5, 6 and 8 bands it gets worse (1.623 → 1.719 %, 1.133 → 1.150 %,
1.315 → 1.855 %). Reporting only the 7-band number would be picking the band set
that flatters the term, so the interior error is treated as unchanged.

### 2. Dam break (B2 geometry, `dx = 0.02`)

| delta | bulk compression | peak compression | front L2 vs `delta = 0` | wall clock |
|---|---|---|---|---|
| 0.0 | 2.66 % | 8.98 % | — | 4.8 s |
| 0.1 | 2.63 % | 8.89 % | **0.20 %** | 5.6 s |

At `dx = 0.01` the same comparison gives 0.08 % of `a`, so the front difference
converges away under refinement and the trajectory is unaffected at the level the
benchmark resolves. Compression is unchanged to two digits.

### 3. Droplet impact — clumping

| delta | min separation | pairs closer than `dx` | KE at t = 0.3 s | wall clock |
|---|---|---|---|---|
| 0.0 | 0.053 `dx` | 10.1 % | 13.40 J | 4.3 s |
| 0.1 | **0.088 `dx`** | 10.1 % | 13.29 J | 4.7 s |

The worst pair separation improves by 67 %, from 5 % of a particle spacing to 9 %,
while the kinetic energy is unchanged to 1 %. Ten percent of all neighbour pairs
are still closer than one spacing in both cases, so the term mitigates
near-coincident particles but does not remove them.

Cost through `run(..., plot=False)` with the Verlet cache and artifacts:
1.78 s to 2.32 s, i.e. **+30 %** for the default 750-step impact run.

### What changed from the 0.6.0 measurement

| observable | 0.6.0 (incomplete wall) | 0.7.0 (complete wall) |
|---|---|---|
| B1 fluctuation, δ = 0 | 67.9 % | 9.0 % |
| B1 fluctuation, δ = 0.1 | 65.6 % | 6.8 % |
| δ effect on the fluctuation | −3 % | **−24 %** |
| B1 residual speed | 0.044 m/s | 0.013 m/s |

In 0.6.0 the pool was ringing (the wall pressure was the kernel average of the
neighbouring fluid pressure, not its extrapolation to the wall face), and that
ringing swamped the quantity the diffusive term acts on. The 0.6.0 conclusion —
"the improvement is 3 %, too small to matter" — was correct for the code it was
measured on and wrong about the term.

## Decision

**The default stays `density_diffusion = 0`.** The case for turning it on is
better than it was, but it is still a cost with no gate to justify it:

- the only robust gain is the residual pressure fluctuation on an
  already-quiet pool (9.0 % → 6.8 %, i.e. the pool is quieter but it was not
  noisy to begin with);
- the interior pressure error moves within the spread of band choices, so there
  is no defensible claim that B1 improves;
- B2 is unchanged, including the peak-compression gate;
- the trade is +30 % wall clock for a 67 % better worst-case particle separation.

Use `Config.density_diffusion = 0.1` when near-coincident particles matter —
splash morphology, high-speed impact, anything where `min separation / dx` below
0.1 is a problem. Do not expect it to fix the free-surface density deficit: at
the top B1 band the error moves from −31.0 % to −35.6 %, i.e. diffusion pulls the
surface band slightly further from the hydrostatic value, because smoothing a
truncated kernel support is not the same as restoring it.

## What would change this conclusion

- **A gate that measures what the term improves.** The fluctuation is now the
  only robust gain (9.0 % → 6.8 %), and it is reported but not gated. A criterion
  such as "relative pressure fluctuation below X % in the last second of B1"
  would give the default decision something to rest on other than judgement.
- **A capillary case.** Without surface tension (`B3` in the roadmap is not
  started) neither treatment can be judged on the physics that usually motivates
  free-surface work.
- **A genuinely quiet free surface.** B1 still rings a little even with the
  complete wall condition (0.013 m/s residual, 9 % fluctuation). A drop
  oscillating in zero gravity, or B1 with a damping ramp before the averaging
  window, would remove the last of the sloshing mode and make the fluctuation a
  cleaner measurement of surface noise.
