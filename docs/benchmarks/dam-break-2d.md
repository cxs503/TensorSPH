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
| 0.04 | 50 | 1 282 | 1.6 s | 5.28 m/s | 0.21 | 9.7 % | 2.4 % | 0 | t = 0.658 s |
| 0.02 | 200 | 2 564 | 4.9 s | 4.44 m/s | 0.18 | 10.3 % | 1.8 % | 0 | t = 0.661 s |
| 0.01 | 800 | 5 128 | 22.6 s | 4.50 m/s | 0.18 | 10.0 % | 2.0 % | 0 | t = 0.662 s |

**Peak versus bulk compression.** "Peak" is the maximum of `rho/rho0 - 1` over
**every integration step**; "bulk" is the same maximum up to the step where the
front reaches the far wall. The two differ by a factor of five because the peak
is a single impact transient: at `dx = 0.02` the surge arrives at
t = 0.661 s and the maximum is recorded at t = 0.678 s, seventeen milliseconds
later, while the bulk flow never exceeds 2.4 %.

Earlier revisions of this document reported 3.8 % at `dx = 0.01`. That number
came from diagnostics sampled every 0.01 s, which stepped over the spike. The
step-wise maximum is 10.0 %, so the `density_compression` gate at a 10 % limit
now **fails** at all three resolutions. The metric did not change; the
measurement did.

**Observed order of accuracy**: the relative L2 difference of `front(t)/a` is
2.58 % between `dx` = 0.04 and 0.02, and 1.31 % between 0.02 and 0.01, giving

    p = log2(2.58 / 1.31) = 0.96

i.e. essentially first order, which is what a semi-implicit Euler integration
with a first-order boundary treatment should give.

Front trajectory at `dx = 0.01`:

| `tau` | `Z/a` | `Z / Ritter` |
|---|---|---|
| 0.00 | 0.97 | — |
| 0.69 | 1.21 | 0.88 |
| 1.38 | 1.79 | 0.65 |
| 2.08 | 2.58 | 0.62 |
| 2.77 | 3.52 | 0.64 |
| 3.46 | 4.57 | 0.66 |
| 4.15 | 5.71 | 0.69 |
| 4.85 | 6.88 | 0.71 |
| 5.54 | 8.07 | 0.73 |
| 6.23 | 9.28 | 0.74 |

The front advances monotonically and stays below the Ritter shallow-water front
(ratio 0.62–0.88), which is expected for a finite reservoir.

## Compressibility study

The same geometry at `dx = 0.01` with three sound speeds. A reference
comparison is only meaningful once the observable stops moving with `c0`.

| `c0` (m/s) | Mach | peak compression | wall clock | front L2 vs next level |
|---|---|---|---|---|
| 25 | 0.18 | 10.0 % | 24 s | 0.034 % (25 → 40) |
| 40 | 0.10 | 6.1 % | 34 s | 0.019 % (40 → 60) |
| 60 | 0.07 | 4.0 % | 48 s | — |

The leading-edge trajectory is converged with respect to compressibility at the
percent level: changing `c0` by a factor of 2.4 moves the front by less than
0.04 % of `a`.

Compression is a different story, and it is why the density gate now fails:
the 10 % peak at `c0 = 25` is an impact transient, and it falls to 6.1 % at
`c0 = 40` without moving the front. So the choice of `c0` is settled by the
density gate rather than by the trajectory: raise it if the peak compression is
the quantity being quoted.

## Gates

| id | metric | measured | limit | result |
|---|---|---|---|---|
| `mass_conserved` | \|m(t_end) - m(0)\| | 0.0 kg | 0 | pass |
| `bounded_flow` | particles clamped beyond the tank | 0 | 0 | pass |
| `mach_number` | max &#124;v&#124; / `c0` | 0.211 | 0.25 | pass |
| `density_compression` | max(rho/rho0 - 1) over every step | **10.3 %** | 10 % | **fail** |
| `bulk_compression` | same, before the front reaches the wall | **2.4 %** | 10 % | pass |
| `resolution_convergence` | relative L2 of `front(t)/a`, `dx` 0.04 vs 0.02 | **2.58 %** | 10 % | pass |
| `sound_speed_convergence` | relative L2 of `front(t)/a`, `c0` 40 vs 60 | **0.019 %** | 5 % | pass |
| `experimental_reference` | agreement with Martin & Moyce (1952) | — | — | **not_evaluated** |

The convergence window stops when the front reaches the far wall (t = 0.669 s),
because after impact the leading edge is pinned by the boundary and no longer
measures the flow.

The `density_compression` gate fails at every resolution by 0–0.3 percentage
points, and it should be reported that way: the peak is a wall-impact transient
of about 10 % that does not converge away with resolution, while the bulk flow
stays at 2 %. Two ways to make the run pass without touching the metric:

```bash
# stop before the surge reaches the far wall (the convergence window already does)
python benchmarks/dam_break/run.py --sizes 0.04,0.02,0.01 --seconds 0.65

# or raise the artificial sound speed: 6.1 % peak at c0 = 40, front unchanged
python benchmarks/dam_break/run.py --sizes 0.04,0.02,0.01 --seconds 0.8 --sound-speed 40
```

The CI-sized configuration (`--sizes 0.04,0.02 --seconds 0.4`) never reaches the
wall and passes at 2.3 %, so this failure does not appear in CI. It is a real
property of the case, not a flaky test.

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
   momentum to machine precision in the same test.
3. **Compressibility — addressed.** See the table above: the front moves by less
   than 0.04 % of `a` when `c0` goes from 25 to 60 m/s.

So the remaining blocker for B2 is data provenance, not the solver.

## What this case does establish

- The free-surface evolution is bounded: nothing leaves the tank, nothing
  diverges, and the front is monotone.
- Mass is conserved exactly (particle masses are constants and are never
  rescaled).
- The observable converges at observed order 0.96 over a three-level ladder,
  which is the prerequisite for any later comparison: a reference comparison on
  an unconverged trajectory would measure discretization noise.
- The trajectory no longer depends on the artificial sound speed (0.019 % of
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
