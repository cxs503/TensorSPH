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

### 1. Hydrostatics (B1 protocol, 7 bands)

| delta | interior pressure error | pressure fluctuation | residual max &#124;v&#124; | wall clock |
|---|---|---|---|---|
| 0.0 | 1.086 % | 67.9 % | 0.044 m/s | 54 s |
| 0.1 | **0.943 %** | **65.6 %** | 0.052 m/s | 59 s |

The interior error improves by 13 % relative and the temporal fluctuation by
3 %. Both are small, and the interior-error improvement is **not robust**: with
six bands instead of seven the same pair of runs reads 0.978 % and 1.073 %
(`--bands 6`), i.e. the sign flips. The fluctuation, which is the quantity a free-surface treatment
is supposed to address, moves in the same direction under both band choices but
only by a few percent.

The reason is visible in the B1 table itself: this pool never comes to rest
(free slip, no damping), so the fluctuation is dominated by a sloshing mode, not
by free-surface noise. `delta` at 0.1 cannot calm a mode that is not the one it
targets.

### 2. Dam break (B2 geometry, `dx = 0.02`)

| delta | bulk compression | peak compression | front L2 vs `delta = 0` | wall clock |
|---|---|---|---|---|
| 0.0 | 1.79 % | 10.26 % | — | 4.7 s |
| 0.1 | 1.60 % | 10.33 % | **0.26 %** | 5.6 s |

At `dx = 0.01` the same comparison gives 0.083 % of `a`, so the front difference
converges away under refinement and the trajectory is unaffected at the level
the benchmark resolves. Bulk compression drops by 0.2 percentage points; the
wall-impact peak does not move.

### 3. Droplet impact — clumping

| delta | min separation | pairs closer than `dx` | KE at t = 0.3 s | wall clock |
|---|---|---|---|---|
| 0.0 | 0.052 `dx` | 10.0 % | 13.54 J | 4.4 s |
| 0.1 | **0.089 `dx`** | 9.9 % | 13.39 J | 4.7 s |

This is the clearest effect: the worst pair separation improves by 71 %, from
5 % of a particle spacing to 9 %, while the kinetic energy is unchanged to 1 %.
Ten percent of all neighbour pairs are still closer than one spacing in both
cases, so the term mitigates near-coincident particles but does not remove them.

Cost: +7 % in this script's bare integration loop (4.4 s to 4.7 s) and **+30 %**
through `run(..., plot=False)` with the Verlet cache and artifacts (1.78 s to
2.32 s, best of three), where the extra mask-indexing is a larger share of a
cheaper step.

## Decision

**The default stays `density_diffusion = 0`.** The measured benefits do not
justify changing the physics every existing number was produced with:

- hydrostatics: the interior error improvement is inside the spread caused by
  choosing a different band set, and the fluctuation improves by 3 %;
- dam break: the front moves by less than 0.3 % and converges away under
  refinement;
- impact: the real gain is in the smallest pair separations, at a 27 % cost.

Use `Config.density_diffusion = 0.1` when near-coincident particles matter —
splash morphology, high-speed impact, anything where `min separation / dx`
below 0.1 is a problem. Do not expect it to fix the free-surface density
deficit: at the top B1 band the error goes from −28.4 % to −35.6 %, i.e. the
diffusion pulls the surface band slightly further from the hydrostatic value,
because smoothing a truncated kernel support is not the same as restoring it.

## What would change this conclusion

- A damping or relaxation phase in B1, so the sloshing mode stops masking the
  free-surface noise. The current metric cannot see what `delta` targets.
- A capillary case: without surface tension (`B3` in the roadmap is not started)
  neither treatment can be judged on the physics that usually motivates
  free-surface work.
- A pressure-noise metric on a genuinely quiet free surface, such as a drop
  oscillating in a zero-gravity box, rather than a sloshing pool.
