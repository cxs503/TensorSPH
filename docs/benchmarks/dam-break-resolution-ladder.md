# B2 dam-break experimental gate: resolution ladder (measured)

**Generated file** -- every number below is injected from
`benchmarks/dam_break/resolution_ladder.json` by
`benchmarks/dam_break/make_ladder_doc.py`. Do not edit by hand; rerun the
generator instead.

## Why this measurement exists

`docs/benchmarks/dam-break-experimental.md` reported the Martin & Moyce
(1952) comparison at one coarse resolution only (`dx` = 0.04 m, 50 fluid
particles, mean absolute error 0.60 in `Z/a`) and then *predicted* that
refinement would fix it:

> "Extrapolating the convergence trend, the experimental MAE should decrease
> approximately as dx^1.11. At dx = 0.01, we expect MAE ~ 0.15-0.20."

That prediction was never measured. This artifact measures it on a
three-rung ladder with everything else held fixed.

## Setup (identical for every rung)

| quantity | value |
|---|---|
| experimental reference | Martin & Moyce (1952) |
| digitized points | 20 |
| column width `a` | 0.2 m |
| column height | 0.4 m (2:1, as in the experiment) |
| tank width | 2.0 m |
| sound speed `c0` | 25.0 m/s |
| wall friction | 0.0 (free slip) |
| density diffusion | 0.0 (off) |
| non-dimensionalisation | `tau = t sqrt(2g/a)`, `Z/a = (x_max - a)/a` |

## Measured results

| `dx` (m) | fluid particles | steps | `dt` (s) | wall clock (s) | exp MAE (`Z/a`) | RMSE | max abs err | points | within uncertainty | Mach | max compression |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | 50 | 801 | 6.240e-04 | 20.9 | 0.5988 | 0.6766 | 1.2020 | 20 | 5 % | 0.132 | 2.35 % |
| 0.02 | 200 | 1603 | 3.120e-04 | 199.4 | 0.5774 | 0.6817 | 1.4862 | 20 | 5 % | 0.133 | 2.66 % |
| 0.01 | 800 | 3205 | 1.560e-04 | 787.7 | 0.5728 | 0.6994 | 1.6182 | 20 | 0 % | 0.134 | 1.93 % |

Experimental uncertainty used for the `within uncertainty` column: +-±0.05 in Z/a (estimated from figure resolution and scatter in original data).

## Error decay: what the ladder actually shows

| rung pair | observed order `p` in `MAE ~ dx^p` |
|---|---|
| 0.04 -> 0.02 m | +0.052 |
| 0.02 -> 0.01 m | +0.012 |

Mean observed order: **+0.032**.

Interpretation, stated plainly:

* The first-order solver converges at ~1 in the *front* trajectory
  (`docs/benchmarks/dam-break-2d.md`: order 0.96-1.11 versus the Ritter
  shallow-water solution).
* Against the *experiment* the same ladder converges at +0.032, i.e. almost not at all: the mean absolute error moves
  from 0.5988 to 0.5728 while the resolution improves
  by a factor 4.
* Therefore the residual is **not** numerical diffusion of the front.
  It is a resolution-independent bias, so the prediction quoted above
  is falsified: refinement to `dx` = 0.01 m leaves the error at
  0.5728, not 0.15-0.20.

## Gate verdict

| gate | metric | measured | limit | result |
|---|---|---|---|---|
| `experimental_comparison` | mean abs error in `Z/a` | 0.5728 | 0.15 | **fail** |
| `within_uncertainty` | fraction inside +-0.05 of `Z/a` | 0 % | 100 % | **fail** |

The benchmark's own gate reports `fail` at the finest
rung; this document records the same verdict and adds the ladder
evidence that the failure is structural rather than under-resolved.

## Where the bias can come from (ranked, not yet separated)

1. **Two-dimensional model of a three-dimensional experiment.** The
   published experiment is a finite-width channel; the simulation is a 2-D
   cross-section with free-slip walls and no side-wall drag. That changes
   the front speed at leading order and does not vanish with `dx`.
2. **Symmetry plane at the back wall.** The 2-D column starts against the
   tank wall with a free-slip/no-penetration condition, while the
   experiment's gate is removed instantly; the release transient differs.
3. **Front definition.** `x_max` is the leading particle, which in SPH
   trails the true continuum front by a fraction of `h`; that lag is
   resolution dependent but small compared with the observed 0.58.
4. **Digitisation uncertainty** (+-0.05 in `Z/a`) is an order of magnitude
   smaller than the measured error, so it cannot explain the gap.

Items 1-3 have to be separated by measurement before any claim of
physical validation of this case is made. The honest status of B2 is
therefore: *numerical verification passes, physical validation fails.*

## Reproduce

```bash
python benchmarks/dam_break/experimental_validation.py \
    --sizes 0.04,0.02,0.01 --seconds 0.5 \
    --json-out benchmarks/dam_break/resolution_ladder.json --plot
python benchmarks/dam_break/make_ladder_doc.py
```

