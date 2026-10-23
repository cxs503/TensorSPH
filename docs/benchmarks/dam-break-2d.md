# Benchmark: B2 dam-break free-surface flow

**Purpose:** exercise the solver on a genuinely free-surface problem whose
observable — the leading-edge position — has a classical non-dimensional form,
and establish **resolution convergence** before any comparison with experiment.

**Status:** numerical verification. `physical_validation` remains
`not_evaluated`: no digitized experimental table is bundled with this
repository.

Run it:

```bash
# three-level ladder + observed order + compressibility sweep (~4 min)
python benchmarks/dam_break/run.py --sizes 0.04,0.02,0.01 --seconds 0.8 \
    --sound-speeds 25,40,60

# quick CI-sized check
python benchmarks/dam_break/run.py --sizes 0.04,0.02 --seconds 0.4 --quick

# no-slip walls instead of free slip
python benchmarks/dam_break/run.py --sizes 0.02 --seconds 0.8 --wall-friction 1.0
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

Three-level ladder, `c0 = 25 m/s`, free-slip walls:

| `dx` | particles | steps | wall clock | max &#124;v&#124; | Mach | peak compression | bulk compression | escapes | front hits wall |
|---|---|---|---|---|---|---|---|---|---|
| 0.04 | 50 | 1 282 | 1.6 s | 3.64 m/s | 0.15 | 8.977 % | 2.35 % | 0 | t = 0.678 s |
| 0.02 | 200 | 2 564 | 5.3 s | 3.89 m/s | 0.16 | 8.982 % | 2.66 % | 0 | t = 0.670 s |
| 0.01 | 800 | 5 128 | 24.1 s | 4.91 m/s | 0.20 | 8.613 % | 1.93 % | 0 | t = 0.668 s |

**Peak versus bulk compression.** "Peak" is the maximum of `rho/rho0 - 1` over
**every integration step**; "bulk" is the same maximum up to the step where the
front reaches the far wall. The peak is a single impact transient recorded a few
milliseconds after the surge arrives, while the bulk flow never exceeds 2.7 %.

Both numbers moved down in 0.7.0, when the wall pressure extrapolation was
completed with the body-force term of Adami et al. (2012) eq. (11); the peak went
from 9.7 / 10.3 / 10.0 % to 8.98 / 8.98 / 8.61 % (8.977 / 8.982 / 8.613 to three
decimals -- the first two agree to two). The limit was not touched: at
0.5.0 the same gate was reported as failing on these numbers, and it is reported
as passing now because the wall condition changed, not the metric.

Earlier revisions of this document reported 3.8 % at `dx = 0.01`. That number
came from diagnostics sampled every 0.01 s, which stepped over the spike. The
step-wise maximum is 8.6 %, so the `density_compression` gate at a 10 % limit
**passes**, with about one percentage point of margin. The metric did not change;
the measurement did, and then the physics did.

**Observed order of accuracy**: the relative L2 difference of `front(t)/a` is
4.83 % between `dx` = 0.04 and 0.02, and 2.19 % between 0.02 and 0.01, giving

    p = log2(4.83 / 2.19) = 1.11

i.e. first order, which is what a semi-implicit Euler integration with a
first-order boundary treatment should give. Before 0.7.0 the same ladder read
2.58 % and 1.31 % (order 0.96). Completing the wall pressure with its body-force
term *increases* the coarse/medium difference, because that term is proportional
to the spacing: it is a first-order correction to a first-order boundary
condition, so it changes the coarse solution more than the fine one, and it moves
the ladder onto a cleaner first order as it does so.

Front trajectory at `dx = 0.01`:

| `tau` | `Z/a` | `Z / Ritter` |
|---|---|---|
| 0.00 | 0.97 | — |
| 0.69 | 1.20 | 0.87 |
| 1.38 | 1.78 | 0.64 |
| 2.08 | 2.57 | 0.62 |
| 2.77 | 3.51 | 0.63 |
| 3.46 | 4.54 | 0.66 |
| 4.15 | 5.64 | 0.68 |
| 4.85 | 6.78 | 0.70 |
| 5.54 | 7.97 | 0.72 |
| 6.23 | 9.18 | 0.74 |

The front advances monotonically and stays below the Ritter shallow-water front
(ratio 0.62–0.88), which is expected for a finite reservoir.

## Compressibility study

The same geometry at `dx = 0.01` with three sound speeds. A reference
comparison is only meaningful once the observable stops moving with `c0`.

| `c0` (m/s) | Mach | peak compression | wall clock | front L2 vs next level |
|---|---|---|---|---|
| 25 | 0.20 | 8.61 % | 23 s | 0.014 % (25 → 40) |
| 40 | 0.12 | 4.76 % | 33 s | 0.014 % (40 → 60) |
| 60 | 0.08 | 5.15 % | 46 s | — |

The leading-edge trajectory is converged with respect to compressibility at the
percent level: changing `c0` by a factor of 2.4 moves the front by less than
0.02 % of `a`.

Compression is a different story: the peak at `c0 = 25` is an impact transient of
8.61 %, it falls to 4.76 % at `c0 = 40`, and then it rises again to 5.15 % at
`c0 = 60`. That last step is not monotone and is not explained by the Mach number
alone; the impact transient is a short event whose capture depends on where a
step lands relative to the surge hitting the wall, and none of these runs resolve
it well. The honest reading is that raising `c0` buys compression margin at the
gate but the residual peak is not a clean function of `c0`.

## Gates

| id | metric | measured | limit | result |
|---|---|---|---|---|
| `mass_conserved` | \|m(t_end) - m(0)\| | 0.0 kg | 0 | pass |
| `bounded_flow` | particles clamped beyond the tank | 0 | 0 | pass |
| `mach_number` | max &#124;v&#124; / `c0` | 0.197 | 0.25 | pass |
| `density_compression` | max(rho/rho0 - 1) over every step | **8.98 %** | 10 % | pass (first time since 0.5.0) |
| `bulk_compression` | same, before the front reaches the wall | **2.66 %** | 10 % | pass |
| `resolution_convergence` | relative L2 of `front(t)/a`, `dx` 0.04 vs 0.02 | **4.83 %** | 10 % | pass |
| `sound_speed_convergence` | relative L2 of `front(t)/a`, `c0` 40 vs 60 | **0.014 %** | 5 % | pass |
| `experimental_reference` | agreement with Martin & Moyce (1952) | — | — | **not_evaluated** |

The convergence window stops when the front reaches the far wall (t = 0.669 s),
because after impact the leading edge is pinned by the boundary and no longer
measures the flow.

The `density_compression` gate now passes at every resolution, with 1.0–1.4
percentage points of margin. It passed at 0.7.0 for the first time because the
wall pressure extrapolation was completed, not because the threshold moved; in
0.5.0 and 0.6.0 the same ladder read 9.7–10.3 % and the gate was reported as
failing. The margin is small enough that it should be re-checked whenever the
wall or the free-surface treatment changes, and it is not an argument that the
peak is resolved: the impact transient is a few milliseconds long and 0.04 m
particles cannot resolve it.

The CI-sized configuration (`--sizes 0.04,0.02 --seconds 0.4`) never reaches the
wall and stays at 2 %, so the peak does not appear in CI at all.

## Why the experimental gate is `not_evaluated`

One reason remains, and two have been addressed:

1. **No data — still open.** The repository does not ship a digitized Martin &
   Moyce table. Transcribing numbers from memory or from a figure would make the
   comparison untraceable, which is the exact failure mode the roadmap is written
   to avoid. The gate names the reference so the work has a target. What is
   needed: the table with its source (figure or table number), the digitization
   method, and an uncertainty per point.

   Searched, without success: the original *Phil. Trans. R. Soc. A* **244**
   paper (DOI `10.1098/rsta.1952.0006`, paywalled), the Aureli et al. (2023)
   dam-break review in *Water* **15**(6):1229 (no table), and GitHub code search
   for `"Martin Moyce"` (0 hits). Later papers replot the `Z/a` against
   `tau = t sqrt(2 g / a)` curve but do not publish the tabulated points. So the
   data is not merely missing from this repository — it is not freely
   redistributable in tabulated form, which is why the gate stays
   `not_evaluated` rather than being filled with numbers of unknown provenance.
2. **Wall condition — addressed.** `--wall-friction 1.0` now gives a no-slip
   floor by extending the viscous stencil into the static wall particles. It is
   verified separately: a quarter-wave shear over a no-slip floor decays at
   0.84x (dx = 0.02) and 0.91x (dx = 0.01) of the analytic Stokes rate
   `nu (pi/2H)^2`, converging under refinement. Free slip conserves tangential
   momentum to machine precision in the same test. The normal direction is the
   Adami et al. (2012) eq. (11) extrapolation, both terms.
3. **Compressibility — addressed.** See the table above: the front moves by less
   than 0.04 % of `a` when `c0` goes from 25 to 60 m/s.

So the remaining blocker for B2 is data provenance, not the solver.

## What this case does establish

- The free-surface evolution is bounded: nothing leaves the tank, nothing
  diverges, and the front is monotone.
- Mass is conserved exactly (particle masses are constants and are never
  rescaled).
- The observable converges at observed order 1.11 over a three-level ladder,
  which is the prerequisite for any later comparison: a reference comparison on
  an unconverged trajectory would measure discretization noise.
- The trajectory no longer depends on the artificial sound speed (0.014 % of
  `a` between `c0` = 40 and 60 m/s), so the weakly-compressible approximation is
  not what would limit the comparison.
- A no-slip floor is available and verified, so the wall condition no longer
  stands between this case and the experiments.

## Next steps

1. **The only blocker:** bundle a digitized Martin & Moyce table with full
   provenance (source, figure or table number, digitization method,
   uncertainty per point) and add the comparison plot. Until that exists the
   gate stays `not_evaluated`; reconstructing numbers from a plotted curve
   without recording the method would defeat the purpose of the gate.
2. Afterwards, decide the default for `wall_friction` on the basis of that
   comparison rather than on a guess.
