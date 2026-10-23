# What changed in 0.5.0

**Scope:** no new physics. This release makes the existing gates measure what
they claim to measure, removes three interfaces that lied about their state,
and takes a measured 1.4x off the force evaluation. Everything below was
measured in this repository; commands are given.

Environment for every number: Debian, Python 3.11.2, `torch 2.14.1+cu130`,
CPU only, `torch.set_num_threads(2)`.

---

## 1. Gates that could not fail

### 1.1 `mass_conserved` was a constant

`benchmarks/dam_break/run.py` compared `mass_drift_kg == 0.0`, where
`mass_drift_kg` was written as a literal `0.0` three lines earlier. The gate
tested a Python constant, not the simulation.

`DamBreakRun` now carries `initial_mass`, `final_mass` and a `mass_drift`
property, and the gate compares `|m(t_end) - m(0)|` against
`--max-mass-drift` (default 0). It still reports 0.0 kg — particle masses are
constants — but the number now comes from two tensor sums.

### 1.2 Stability gates only saw the diagnostic rows

`rho_max`, `max_speed` and the Courant number were sampled every
`output_every` steps (default 50), and `evaluate_acceptance` judged the run from
those samples alone. An overshoot between two output points was invisible.

The integration loop now accumulates `peak_compression` and `peak_speed` with
`torch.maximum` on every step — two elementwise ops, no host round-trip — and
the criteria are evaluated from those. Every criterion records the population
it was measured over:

| criterion | population |
|---|---|
| `finite_state` | `sampled_rows` |
| `density_compression`, `mach_number`, `acoustic_courant` | `all_steps` |
| `boundary_escape` | `all_steps` |

The effect on the default impact case is not cosmetic:

| quantity | sampled rows (16) | every step (751) |
|---|---|---|
| max density compression | 2.00 % | **3.64 %** |
| max speed | 0.704 m/s | **1.424 m/s** |
| max Mach | 0.035 | **0.071** |
| max Courant | 0.256 | 0.264 |

The recorded peak speed was less than half the true peak. On the dam-break case
the difference is larger and is discussed in
[`benchmarks/dam-break-2d.md`](benchmarks/dam-break-2d.md): the sampled
compression was 3.8 % at `dx = 0.01`, the step-wise maximum is 10.0 %, and the
`density_compression` gate now fails.

The manifest records this explicitly:

```json
"sampling": {
  "diagnostic_rows": 16,
  "output_every": 50,
  "extrema_population": "all_steps",
  "extrema": {"peak_rho_compression": 0.0364, "peak_speed": 1.424, "peak_courant": 0.264}
}
```

CI asserts the population is `all_steps` for the three stability criteria.

### 1.3 The time step was validated against the acoustic limit only

`validate_config` checked `dt * c0 / h <= max_courant` and nothing else. The
explicit Morris Laplacian is not bounded by that condition, so a configuration
could pass validation and diverge:

| `viscosity` | viscous limit `0.125 h²/ν` | accepted before | outcome |
|---|---|---|---|
| 0.01 (default) | 1.3e-2 s | yes | fine |
| 0.5 | 2.6e-4 s | yes | marginal, 0.7 J kinetic energy |
| 1.0 | 1.3e-4 s | **yes** | 53 % density compression, Mach 1.18 |
| 2.0 | 6.6e-5 s | **yes** | 1.6e4 % compression, non-finite state |

`validate_config` now also enforces `dt <= viscous_dt_limit(cfg)` and
`dt <= gravity_dt_limit(cfg)`, both exposed as functions so a caller can read
the limit instead of parsing the error message. The failing configuration is
rejected before the first step instead of after 800.

---

## 2. Interfaces that lied

| symptom | cause | fix |
|---|---|---|
| `DamBreakConfig(device="auto")` raised `RuntimeError: Expected one of cpu, cuda, … device type …` | `dam_break` called `torch.device(cfg.device)` directly | both cases go through `resolve_device` |
| `DamBreakConfig(boundary="clamp")` ran as if it were `"wall"` | `run_case` never read `cfg.boundary` | `validate_case` rejects anything but `"wall"` |
| `rhs` accepted 3-D input and applied the 2-D kernel normalisation | the `10/(7 pi h²)` constant was not tied to a dimension | `rhs` rejects `pos.shape[1] != 2` |
| `--include-host` could not be turned off | `store_true` with `default=True` | replaced by `--skip-host` |
| `--observed-order` did nothing | parsed, never read | removed; the order is always reported when three spacings are given, and `--max-mass-drift` took its place |
| `--dt 1e-2` printed a traceback | the CLI caught only `SimulationUnstableError` | `ValueError` is reported as a configuration failure with exit status 2 |

The 3-D check matters because `neighbor_search` supports 1-D, 2-D and 3-D; only
the solver is 2-D. Silently producing a wrong density in 3-D was the one place
where the modules disagreed about their contract.

---

## 3. Performance

### 3.1 The hotspot was mask indexing, not `index_add`

Profiling `rhs` at 608 fluid + 291 wall particles attributes 30 % of the time
to `aten::index`, 13 % to `aten::nonzero` and 13 % to `aten::copy_` — all of it
producing the boolean masks that select fluid/wall pairs. Mask-indexing calls
dominated the 702 ATen calls the function made per evaluation.

The wall pressure extrapolation now gathers with a clamped index and zeroes the
weight of non-crossing pairs, and the viscous blend is built as a float
multiplier instead of selecting the pairs first. The kernel value and gradient
come from one shared helper (`pair_kernels`) instead of recomputing the radius
and the branch twice.

Result: **bit-identical** output, **702 → 515 ATen calls** and
**60 → 17 mask-indexing calls** per evaluation (profiler counts at 608 + 291
particles).

### 3.2 22 % of the pairs could not contribute

In the default impact case 1 823 of 7 918 pairs are wall-wall pairs. They enter
the density sum of wall particles and the force sum of static particles, and
both results are discarded: the wall pressure is extrapolated *from* the fluid,
and a static particle's acceleration is never integrated.

`drop_wall_wall_pairs(pairs, fluid_count)` removes them, and
`PairListCache(..., fluid_count=n)` prunes at build time so the saving survives
list reuse. A test asserts the pruned list produces bit-identical `rho`,
`pressure` and `acc`.

### 3.3 Measured

| metric | before | after | factor |
|---|---|---|---|
| `rhs`, 608 particles + 291 wall, cached pairs (median of 7×20) | 2.62 ms | **1.85 ms** | 1.42x |
| full integration step (rhs + pair reuse + boundary), 400 steps | 2.99 ms | **2.17 ms** | 1.38x |
| pairs fed to `rhs` | 9 342 | 7 323 | −22 % |
| default 750-step run, with plot | 5.02–5.18 s | **4.50–4.70 s** | 1.10x |
| default 750-step run, `--no-plot` | — | **3.51–3.59 s** | 1.45x vs the 5.05 s baseline |
| dam-break ladder `dx = 0.01`, 5 128 steps | 32.3 s | **22.6 s** | 1.43x |
| dam-break `dx = 0.02` | 6.8 s | **4.9 s** | 1.39x |
| dam-break `dx = 0.04` | 2.0 s | **1.6 s** | 1.25x |

Physics is unchanged: 400 integration steps of the impact case with the old and
the new code produce bit-identical positions and velocities, and the B1/B2
numbers that depend on the trajectory are unchanged (interior pressure error
1.086 %, front L2 2.584 % / 1.312 %, observed order 0.9604).

### 3.4 Where the remaining time goes

`run` still renders the figure on the way out: matplotlib import plus save is
about 1.5 s, i.e. **a third of the default run**. `run(cfg, out, plot=False)`
and `--no-plot` skip it. Snapshots are now written as they are produced and the
plot reads back at most 12 of them, so memory no longer grows with `steps`
(previously every snapshot was cloned and held until the end).

The force evaluation is still dispatch-bound: 515 ATen calls for 7 323 pairs
means roughly 2.7 µs of Python/dispatch per pair *group*, not per pair. On this
2-core box a single elementwise op on 8 000 elements costs 1.8 µs and an
`index_add` costs 20 µs, so the next meaningful step is fusion (`torch.compile`)
rather than more arithmetic. It could not be measured here — the sandbox has no
C++ toolchain, so Inductor fails to build — and no compiled path was added
without a measurement.

---

## 4. Packaging and CI

- `matplotlib` moved to the `plot` extra; the solver degrades with a
  `RuntimeWarning` that names the extra. A base install no longer pulls a
  plotting stack to compute a density field.
- `ruff` with `select = ["E", "F", "W"]` and `line-length = 100` is a CI gate.
  Only two lines needed changing to make the tree clean at that setting.
- CI now also runs Python 3.11 (the matrix skipped it), and asserts the manifest
  contract including the `all_steps` extrema population.
- Tests: 67 → 82. The additions pin the mass-drift measurement, the all-step
  extrema, both CFL rejections, the 3-D rejection, plot opt-out, wall-wall pair
  pruning neutrality, the `fluid_count` cache, and the dam-break boundary
  rejection and impact-spike separation.

---

## 5. Still open

- **No LICENSE file.** Discussed in the gap analysis, still absent. Choosing one
  is the maintainer's call, not a refactor.
- **Experimental reference data.** Unchanged: B2 still has no digitized
  Martin & Moyce table, and the gate stays `not_evaluated`.
- **Free-surface treatment.** Still the zero-pressure clamp. Delta-SPH or
  artificial stress remains the next scientific increment, and the honest way
  to land it is a side-by-side comparison against the clamp.
- **Adaptive time step.** The step is now validated against three limits at
  start-up but stays fixed during the run.
- **CUDA.** The pruning and mask changes are device-agnostic, but nothing here
  was run on a GPU. `PairListCache.displacement` and the diagnostics still sync
  to the host once per step.
