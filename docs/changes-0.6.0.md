# 0.6.0 — free-surface treatment: a measured alternative to the clamp

**Version:** 0.5.0 → 0.6.0
**Date:** 2026-10-23
**Physics of the default path:** unchanged. `Config.density_diffusion` defaults to
`0.0`; a 400-step trajectory at the default is `torch.equal` to the 0.4.0
baseline (pos and velocity), and a test asserts the term is never *called* when
it is disabled, so the disabled branch cannot drift.

## Why

The gap analysis calls the free-surface treatment "first-order": the Tait law is
clamped at zero, so the solver cannot represent any rarefaction. Clamping is
what makes the solver stable at a surface where the summation density
under-counts kernel support, but it is a switch, not a model.

The proposed next step was to add the density-diffusion term of Antuono,
Colagrossi & Marrone (2012), measure it *next to* the clamp rather than
substituting it, and let the measurement pick the default. This release does
that.

## What was added

| Area | Change |
|---|---|
| Config | `density_diffusion: float = 0.0` — δ of the diffusive term. `0.0` disables it |
| Operator | `density_diffusion_rate(pos, mass, rho, cfg, pairs, fluid_count)`; wired into `rhs` and its dense reference `rhs_all_pairs` |
| Stability | `diffusion_dt_limit(cfg) = 0.125 h/(δ c0)`; `validate_config` rejects δ < 0 and a step that violates the limit |
| Manifest | `derived` now records `viscous_dt_limit`, `gravity_dt_limit` and `diffusion_dt_limit` (null when the term is off — `inf` is not JSON) |
| Benchmarks | `--density-diffusion` on `hydrostatic/run.py` and `dam_break/run.py`; new `free_surface/compare.py` |
| B1 diagnostics | pressure **fluctuation** per band (second moment over the averaging window), because a time average hides the noise a free-surface treatment targets |
| Tests | `tests/test_free_surface.py`, 8 tests → suite 82 → **90**, ~10 s |

## The operator

```
rate_i = -2 δ c0 h sum_j V_j (rho_i - rho_j) . -(r_ij·∇W_ij)/r_ij^2 ,   V_j = m_j/rho_j
```

applied as `-c` to `i` and `+c V_i/V_j` to `j`, so the volume-weighted mean
density is conserved. It only sees **fluid-fluid** pairs: the wall density is an
extrapolation, not a state variable, and diffusing into it would drain mass.
`delta = 0` short-circuits before any indexing, so the default path costs
nothing — a test spies on the operator to prove it is not merely adding zeros.

Verified invariants: a uniform field gives exactly zero rate; `Σ V_i rate_i = 0`;
the variance of a perturbed field falls in one step (20.6 → 18.7); the pair form
matches the dense reference to 5.6e-17.

**It is not delta-SPH.** delta-SPH integrates the continuity equation; this
solver obtains density by summation, so the term is applied to the summation
density over `cfg.dt`. **It is not surface tension** either, and the docs say so.

## The measurement

`python benchmarks/free_surface/compare.py --deltas 0,0.1`, two CPU cores.

| Observable | δ = 0 (clamp) | δ = 0.1 | Verdict |
|---|---|---|---|
| B1 interior pressure error | 1.086 % | 0.943 % | −13 %, but the sign flips with `--bands 6` (0.978 / 1.073 %) — inside the band-selection spread |
| B1 temporal pressure fluctuation | 67.9 % | 65.6 % | the intended effect, but only −3 %: the pool sloshes, and a sloshing mode is not what the term targets |
| B1 top (free-surface) band | −28.4 % | −35.6 % | **worse**; smoothing a truncated kernel support is not restoring it |
| B2 front, L2 vs clamp | — | 0.26 % (dx = 0.02), 0.083 % (dx = 0.01) | converges away; trajectory unaffected |
| B2 bulk compression | 1.79 % | 1.60 % | small improvement |
| B2 wall-impact peak | 10.26 % | 10.33 % | unchanged — the failing gate stays failing |
| Droplet impact, worst pair separation | 0.052 `dx` | **0.089 `dx`** | the clearest effect: +71 % |
| Droplet impact, pairs closer than `dx` | 10.0 % | 9.9 % | near-coincident particles are mitigated, not removed |
| Kinetic energy at t = 0.3 s | 13.54 J | 13.39 J | −1 % |
| Cost, bare loop / `run(plot=False)` | 4.4 s / 1.78 s | 4.7 s / 2.32 s | +7 % / **+30 %** |

## Decision

**Default stays 0.0.** The gains are real but narrow: pair separations improve
substantially, the hydrostatic error does not improve robustly, the free-surface
band gets worse, and the one gate that already fails still fails. Nothing here
justifies changing the physics behind every published number.

The knob is documented for the case where it does matter — high-speed impact
and splash morphology, where `min separation / dx < 0.1` is a problem — and the
document says plainly what would change the conclusion: a quiet free-surface
case (oscillating drop in zero gravity) instead of a sloshing pool, and a
capillary case, for which no surface-tension model exists yet.

## Honest gaps

- No physical validation for either treatment; this is a numerical comparison.
- The fluctuation metric cannot separate free-surface noise from the sloshing
  mode, which is why the conclusion rests on the clumping statistic as well.
- The wall boundary density is still extrapolated rather than time-integrated,
  so the diffusive term is blind to it by construction.
- `torch.compile` remains unmeasured (no C++ toolchain in this environment).
