# Benchmark: B2 dam-break free-surface flow

**Purpose:** exercise the solver on a genuinely free-surface problem whose
observable — the leading-edge position — has a classical non-dimensional form,
and establish **resolution convergence** before any comparison with experiment.

**Status:** numerical verification. `physical_validation` remains
`not_evaluated`: no digitized experimental table is bundled with this
repository.

Run it:

```bash
python benchmarks/dam_break/run.py --sizes 0.02,0.01 --seconds 0.8
python benchmarks/dam_break/run.py --sizes 0.04,0.02 --seconds 0.35 --quick
```

## Case

| quantity | value |
|---|---|
| column width `a` | 0.2 m |
| column height | 0.4 m (= 2a, the Martin & Moyce aspect ratio) |
| tank | 2.0 x 0.75 m, static wall particles, free slip |
| water | `rho0 = 1000 kg/m^3`, `c0 = 25 m/s`, `gamma = 7`, `nu = 0.01` |
| `dt` | from the acoustic Courant limit 0.3 |
| initial state | at rest, masses from the hydrostatic profile |

The leading edge `Z(t)` is the maximum x of the water particles, normalised as
`Z/a` against `tau = t sqrt(2 g / a)`, which is the convention of the classical
dam-break experiments.

## Measured

| `dx` | particles | steps | wall clock | max &#124;v&#124; | Mach | compression | escapes | front hits wall |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 200 | 2 564 | 7.3 s | 4.63 m/s | 0.19 | 7.4 % | 0 | t = 0.649 s |
| 0.01 | 800 | 5 128 | 33.9 s | 4.92 m/s | 0.20 | 5.7 % | 0 | t = 0.649 s |

Front trajectory at `dx = 0.01`:

| `tau` | `Z/a` | `Z / Ritter` |
|---|---|---|
| 0.00 | 0.97 | — |
| 0.79 | 1.30 | 0.82 |
| 1.58 | 2.03 | 0.64 |
| 2.37 | 3.02 | 0.64 |
| 3.16 | 4.19 | 0.66 |
| 3.96 | 5.50 | 0.69 |
| 4.75 | 6.86 | 0.72 |
| 6.33 | 9.69 | 0.77 |

The front advances monotonically, stays below the Ritter shallow-water front
(ratio 0.63–0.77), and the two resolutions agree well.

## Gates

| id | metric | measured | limit | result |
|---|---|---|---|---|
| `mass_conserved` | mass drift | 0.0 kg | 0 | pass |
| `bounded_flow` | particles clamped beyond the tank | 0 | 0 | pass |
| `mach_number` | max &#124;v&#124; / `c0` | 0.197 | 0.25 | pass |
| `density_compression` | max(rho/rho0 - 1) | 7.4 % | 10 % | pass |
| `resolution_convergence` | relative L2 of `front(t)/a`, `dx` 0.02 vs 0.01 | **1.68 %** | 10 % | pass |
| `experimental_reference` | agreement with Martin & Moyce (1952) | — | — | **not_evaluated** |

The convergence window stops when the front reaches the far wall (t = 0.649 s),
because after impact the leading edge is pinned by the boundary and no longer
measures the flow.

## Why the experimental gate is `not_evaluated`

Three separate reasons, all of which must be resolved before this case can be
called validated:

1. **No data.** The repository does not ship a digitized Martin & Moyce table.
   Inventing or hand-transcribing numbers would make the comparison
   untraceable, which is the exact failure mode the roadmap is written to avoid.
   The gate names the reference so the work has a target.
2. **Wrong wall condition.** The experiments have no-slip floors; these walls are
   static particles with free-slip viscosity. A faster front than the experiment
   is therefore expected, independent of resolution. Adding wall friction is its
   own change with its own verification.
3. **Compressibility.** Mach 0.2 and 5–7 % density compression are at the edge of
   the weakly-compressible assumption. A production run should raise `c0`, with
   the corresponding `dt` cost, and show that the front trajectory stops
   changing with `c0`.

## What this case does establish

- The free-surface evolution is bounded: nothing leaves the tank, nothing
  diverges, and the front is monotone.
- Mass is conserved exactly (particle masses are constants and are never
  rescaled).
- The observable converges at 1.68 % L2 between `dx = 0.02` and `dx = 0.01`,
  which is the prerequisite for any later comparison: a reference comparison on
  an unconverged trajectory would measure discretization noise.

## Next steps, in order

1. Bundle a digitized reference table with full provenance (source, figure,
   digitization method, uncertainty) and add the comparison plot.
2. Add wall friction and re-run; report both free-slip and no-slip curves.
3. Sweep `c0` (25, 40, 60 m/s) at fixed `dx` and show the front trajectory
   converges with compressibility.
4. Extend the resolution ladder to three spacings and report the observed order.
