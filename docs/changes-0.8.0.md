# 0.8.0 — an adaptive step, and a gate that can see ringing

**Version:** 0.7.0 → 0.8.0
**Date:** 2026-10-23
**Default physics:** unchanged. `Config.adaptive_dt` defaults to `False`, and the
fixed path produces the same numbers as 0.7.0 (verified: the dam-break ladder
re-measures 8.977 / 8.982 / 8.613 %, order 1.11, L2 4.83 / 2.19 %).

Two loose ends from the gap analysis, both closed.

## 1. Adaptive time step

### The problem

`validate_config` checks the step against four limits — acoustic, viscous,
gravity, diffusion — but it checks them **at rest**, and then the step never
changes. For the acoustic limit that is not the same thing: the signal speed in a
moving fluid is `c0 + |v|`, so a step that satisfies `dt c0 / h <= 0.3` can
exceed 0.3 once the flow accelerates.

The dam-break case does exactly that, and the repository never said so, because
the recorded Courant number was computed from `c0` alone:

| `dx` | nominal `dt c0 / h` | measured `dt (c0+|v|) / h` |
|---|---|---|
| 0.04 | 0.300 | 0.344 |
| 0.02 | 0.300 | 0.347 |
| 0.01 | 0.300 | 0.359 |

### The policy

```python
stable_dt(cfg, max_speed) = min(acoustic_dt_limit(cfg, max_speed),   # (c0 + |v|)
                                viscous_dt_limit(cfg),
                                gravity_dt_limit(cfg),
                                diffusion_dt_limit(cfg))

next_dt(cfg, dt, max_speed) = min(cfg.dt_safety * stable_dt(cfg, max_speed),  # 0.9
                                  dt * cfg.dt_growth,   # grow slowly, default 1.1
                                  cfg.dt)               # the configured step is the cap
```

Shrink immediately (the next step is already safe), grow by at most
`dt_growth` per step, never exceed the step that start-up validation approved.

`dt_safety = 0.9` is not decoration. The limits are evaluated with the speed at
the **start** of the step, and a particle can accelerate during it: a stress case
with `gravity = 200 m/s^2` measured a peak Courant number of **0.979 against a 0.9
limit** without the headroom, and 0.872 with it. That discovery is a test
(`test_an_accelerating_flow_cannot_exceed_the_limit`) so it cannot be removed by
accident.
`rhs` takes the step it is actually taking, so the diffusive density term is not
applied for a step that never happened.

Two consequences that are easy to get wrong:

- **The loop runs to a time, not to a step count.** `cfg.steps` at `cfg.dt` is
  the requested simulated time; an adaptive run needs more iterations for the
  same physics. Hitting the 4x safety cap is reported as a *failure*, not
  silently truncated.
- **Courant is a per-step product.** It is now accumulated as
  `dt_n (c0 + |v|_n) / h` inside the loop instead of being reconstructed from the
  maxima of two independent series, which would overestimate it.

### Measured

| | fixed | adaptive |
|---|---|---|
| steps (`dx` 0.04 / 0.02 / 0.01) | 1 282 / 2 564 / 5 128 | 1 579 / 3 164 / 6 347 (+23 %) |
| measured Courant | 0.344 / 0.347 / 0.359 | **0.300 / 0.300 / 0.300** (never above) |
| front L2 vs fixed | — | 0.05 / 0.03 / 0.02 % |
| peak compression | 8.98 / 8.98 / 8.61 % | 8.58 / 8.92 / 7.93 % |
| observed order | 1.11 | 1.08 |

The adaptive maximum is 0.300 because the configured cap gives exactly that at
rest; the operating point once the surge accelerates is 0.27. The property that
matters is that the measured number never exceeds the nominal one.

The trajectory is unchanged at the level the benchmark resolves (0.02–0.05 %), for
23 % more steps. The wall-impact peak moves by up to 0.7 points because it is a
millisecond-scale transient whose capture depends on where the steps land — the
same reason its margin was already flagged as fragile.

On the default droplet-impact case adaptation is a **no-op**: its configured step
is 41 % of what the acoustic limit allows, so the step never leaves the cap and
the run is bit-identical. A test asserts that, so the reference numbers cannot
drift silently if the policy changes.

## 2. Pressure-fluctuation gate

B1 had one gate on the time-averaged pressure profile, and a time average is
exactly what a ringing pool passes. The 0.6.0 wall code — missing the body-force
term of Adami et al. (2012) eq. (11) — scored 1.09 % and 3.51 % on the two
existing gates while the pool oscillated at 52–109 % of the hydrostatic pressure.

`benchmarks/hydrostatic/run.py` now also gates the **temporal standard deviation**
of the pressure over the averaging window, averaged over the interior bands:

| code | interior error | surface drop | fluctuation | gates |
|---|---|---|---|---|
| 0.6.0 (incomplete wall) | 1.09 % | 3.51 % | **67.9 %** | fail |
| 0.8.0 | 1.12 % | 3.15 % | **9.0 %** | pass |

The limit is 15 %: a statement that a resting surface should not wander by more
than about a seventh of the hydrostatic pressure. The current run has 6 points of
margin, and the number the limit was checked against is the real defect above,
not the current code. CI runs B1 at the full 4 s protocol and additionally runs a
deliberately over-constrained case (`--max-fluctuation 0.001`) to prove the gate
can fail.

## Where the two changes meet

The fluctuation gate makes the density-diffusion comparison in
[`benchmarks/free-surface-treatment.md`](benchmarks/free-surface-treatment.md)
actionable: `density_diffusion = 0.1` measures 6.8 % against 9.0 %, i.e. it buys
25 % of the quantity that is now gated. It still does not change the default —
+30 % cost for a gain that a quiet pool does not need — but the trade-off is now
stated in the currency of a gate instead of in prose.

## Honest gaps

- Adaptation bounds the step by stability, not by accuracy. Nothing here measures
  a convergence rate with respect to `dt`; the front L2 comparison against the
  fixed run (0.42 %, 0.05 %) is a sanity bound.
- The wall-impact compression peak remains transient-dominated and remains the
  least trustworthy number in the ladder.
- `torch.compile` is still unmeasured (no C++ toolchain here).
- The hydrostatic fluctuation gate costs 51 s per CI job; it is the only
  criterion in the repository that needs a settled flow to mean anything.
