# Design: Stokes' second problem (oscillating plate)

Anchor: `CASE-SPH-STOKES-2ND-PROBLEM` (feature ledger), acceptance
`<5 %` amplitude-decay error against the analytic solution inside the Stokes
layer with at least 5 particle rows in that layer.

## Why this case

`benchmarks/couette` verifies the steady, straight profile and
`benchmarks/poiseuille` the steady, curved one.  Both are time-invariant, so a
wall model can be wrong in the tangential direction and still match a steady
profile once the interior is pinned.  The oscillating plate loads the wall with a
known time-dependent shear and asks two questions that a wrong tangential
condition cannot answer simultaneously:

* does the amplitude decay with the correct length `delta = sqrt(2 nu / omega)`,
* is the phase lag `y/delta`?

The answer is the finite-gap Stokes solution; it needs no experiment, no tuning,
and the run starts from rest so the initial condition is not the answer.

## What enters the solver

Nothing new.  The case drives the existing production path
(`tensorsph.droplet_impact.rhs`) with `wall_vel` -- the same input the wave-flume
wavemaker uses -- updated from the analytic plate velocity each step, and
`wall_friction = 1`.  `gravity = 0` so the problem is pure shear.

## Metrics, and why these ones

| metric | definition | gate |
|---|---|---|
| `G1` amplitude L2 | RMS of `A_meas - A_exact` over rows `0 < y <= delta` (first row excluded) / `U` | `< 5 %` |
| `G2` decay rate | least-squares `d ln A / dy` times `delta`, compared with the exact slope | `|x + 1| < 5 %` |
| `G3` phase slope | least-squares `d phi / dy` times `delta` | `|x - 1| < 5 %` |
| `G4` free-slip control | first-row amplitude of a `wall_friction = 0` run / analytic | `< 20 %` |
| `G5` resolution | rows strictly inside `0 < y <= delta`, first row excluded | `>= 5` |

`G2`/`G3` are slopes and therefore insensitive to a constant wall offset (the
effective position of the no-slip plane), which is the quantity that most SPH
wall models get wrong by a fraction of `dx`.  `G1` keeps the absolute level
honest, so a profile that decays correctly but is scaled wrong cannot pass both.
`G4` makes the case falsifiable: if the no-slip blend were ineffective, the
amplitude would collapse and `G1` would fail for the wrong reason.

## Runtime and cost

`H/dx = 40` at `delta/dx = 7.1` costs ~40 s of CPU per 4 periods on the
reference platform; `H/dx = 30` (below the five-row requirement, reported as the
coarseness side of the ladder) ~20 s.  Both fit inside the routine test budget,
and the CI test only runs `H/dx = 20`, one discarded and one averaged period.

## Known limitations

* The first fluid row is not gated (one-sided kernel, extrapolated wall
  pressure); it is reported in `out/results.json` and plotted.
* The wall/fluid viscosity blend raises the effective viscosity in the first
  row above `nu`; the slope metrics are the parts of the answer that do not
  depend on it.
* The weakly compressible solver is not acoustic-resolving at this `c0`
  (`Mach = 4e-3`); only the fundamental component is compared, which is all the
  exact solution contains.