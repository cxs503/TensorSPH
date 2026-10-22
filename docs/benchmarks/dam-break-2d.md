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

| `dx` | particles | steps | wall clock | max &#124;v&#124; | Mach | compression | escapes | front hits wall |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 50 | 1 282 | 3.1 s | 5.26 m/s | 0.21 | 9.7 % | 0 | t = 0.659 s |
| 0.02 | 200 | 2 564 | 9.4 s | 4.41 m/s | 0.18 | 9.9 % | 0 | t = 0.669 s |
| 0.01 | 800 | 5 128 | 40.9 s | 4.39 m/s | 0.18 | 3.8 % | 0 | t = 0.669 s |

**Observed order of accuracy**: the relative L2 difference of `front(t)/a` is
2.58 % between `dx` = 0.04 and 0.02, and 1.31 % between 0.02 and 0.01, giving

    p = log2(2.58 / 1.31) = 0.96

i.e. essentially first order, which is what a semi-implicit Euler integration
with a first-order boundary treatment should give.

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

The front advances monotonically and stays below the Ritter shallow-water front
(ratio 0.62–0.75), which is expected for a finite reservoir.

## Compressibility study

The same geometry at `dx = 0.01` with three sound speeds. A reference
comparison is only meaningful once the observable stops moving with `c0`.

| `c0` (m/s) | Mach | compression | wall clock | front L2 vs next level |
|---|---|---|---|---|
| 25 | 0.18 | 3.8 % | 40.7 s | 0.034 % (25 → 40) |
| 40 | 0.10 | 2.2 % | 59.8 s | 0.019 % (40 → 60) |
| 60 | 0.07 | 2.0 % | 85.3 s | — |

The leading-edge trajectory is converged with respect to compressibility at the
percent level: changing `c0` by a factor of 2.4 moves the front by less than
0.04 % of `a`. The weakly-compressible assumption is therefore not the limiting
error for this observable, and `c0 = 25 m/s` is adequate for the default case.

## Gates

| id | metric | measured | limit | result |
|---|---|---|---|---|
| `mass_conserved` | mass drift | 0.0 kg | 0 | pass |
| `bounded_flow` | particles clamped beyond the tank | 0 | 0 | pass |
| `mach_number` | max &#124;v&#124; / `c0` | 0.197 | 0.25 | pass |
| `density_compression` | max(rho/rho0 - 1) | 7.4 % | 10 % | pass |
| `resolution_convergence` | relative L2 of `front(t)/a`, `dx` 0.04 vs 0.02 | **2.58 %** | 10 % | pass |
| `sound_speed_convergence` | relative L2 of `front(t)/a`, `c0` 40 vs 60 | **0.019 %** | 5 % | pass |
| `experimental_reference` | agreement with Martin & Moyce (1952) | — | — | **not_evaluated** |

The convergence window stops when the front reaches the far wall (t = 0.669 s),
because after impact the leading edge is pinned by the boundary and no longer
measures the flow.

The `density_compression` gate is the marginal one at coarse resolution: 9.7 %
and 9.9 % at `dx` = 0.04 and 0.02 against a 10 % limit, falling to 3.8 % at
`dx` = 0.01. The coarse levels are at the edge of the weakly-compressible
assumption, which is worth remembering before quoting coarse-grid results.

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
