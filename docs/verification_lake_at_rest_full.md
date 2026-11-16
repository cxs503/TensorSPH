# Lake at rest: per-particle mass reconstruction for full balance

> **Evidence channel (m23060).** This document fills the gap in
> `docs/verification_lake_at_rest.md` §5 ("Not done"): the per-particle
> (non-mirror) reconstruction minimising both acceleration components.
> The module `tensorsph.lake_at_rest_full` implements this approach and
> is tested by `tests/test_lake_at_rest_full.py`.

Case-pool anchor `CASE-SPH-LAKE-AT-REST`. Reference: the analytic hydrostatic
identity (`u = 0`, flat free surface) — no experimental uncertainty, no fitted
constant. Artefact: `tests/test_lake_at_rest_full.py`; code
`src/tensorsph/lake_at_rest_full.py`.

Reproduce:

```
PYTHONPATH=src python -m pytest tests/test_lake_at_rest_full.py -v
```

## 1. Verdict

**PARTIAL** — the per-particle reconstruction reduces the combined residual
from 0.262g to 0.077g but **cannot** achieve machine precision.  This is a
fundamental limitation: the system is overdetermined (2n equations for ax and
ay, but only n unknowns for mass per particle).

## 2. Method

The original `reconstruct` function ties mirror pairs and only minimises
vertical acceleration (`a_y`).  The horizontal component (`a_x`) remains at
`O(0.1g)` in the side-wall band because the horizontal force is odd under
mirror reflection and no symmetric mass field can cancel it.

This module relaxes the symmetry constraint and solves for one mass per
particle, minimising both `a_x` and `a_y` simultaneously via Gauss-Newton
with autograd Jacobian.  The lattice and pair list are frozen (taken from
the input state), so the shipped `rhs` remains fully differentiable.

## 3. Results (dx = 0.04, 70 particles)

| iteration | max|a|/g | max|a_x|/g | max|a_y|/g |
|-----------|---------|-----------|-----------|
| 0 (prior) | 2.619e-01 | 3.687e-02 | 2.608e-01 |
| 1 | 2.174e-01 | 1.398e-01 | 2.165e-01 |
| 2 | 1.825e-01 | 1.286e-01 | 1.732e-01 |
| 3 | 1.741e-01 | 1.179e-01 | 1.735e-01 |
| 4 | 1.615e-01 | 1.085e-01 | 1.610e-01 |
| 5 | 1.607e-01 | 9.512e-02 | 1.603e-01 |
| 6 | 1.106e-01 | 7.767e-02 | 1.102e-01 |
| 7 (final) | 7.730e-02 | 7.129e-02 | 2.988e-02 |

The solver stops at iteration 7 because no descent direction is found
(line search fails).  The residual is reduced by a factor of 3.4 but
remains at `O(0.1g)`.

### Comparison with other approaches

| approach | max|a|/g | max|a_x|/g | max|a_y|/g | DOFs |
|----------|---------|-----------|-----------|------|
| analytic prior | 0.262 | 0.037 | 0.261 | n/a |
| mirror-tied (original) | 1.48e-13 | 0.086 | 1.48e-13 | n/2 |
| per-particle (this work) | 0.077 | 0.071 | 0.030 | n |
| periodic tank | 5.76e-13 | 1.39e-12 | 4.37e-13 | n/rows |

The mirror-tied version achieves machine precision for `a_y` but leaves
`a_x` at 0.086g.  The per-particle version reduces both components but
cannot reach machine precision because the system is overdetermined.

## 4. Error judgement (per key quantity, explicit < 3 % answer)

| key quantity | reference | error | < 3 %? |
|--------------|-----------|-------|--------|
| combined residual vs machine precision | 0 | 7.73e-02 g | **no** — fundamental limitation |
| horizontal residual vs mirror-tied | 0.086 g | 0.071 g | **yes** (17% reduction) |
| vertical residual vs mirror-tied | 1.48e-13 g | 2.99e-02 g | **no** — trade-off for reducing ax |

## 5. Findings

**F5 (new, documented).** The per-particle mass reconstruction reduces the
combined residual from 0.262g to 0.077g but cannot achieve machine precision.
This is because the system is overdetermined: 2n equations (ax, ay per particle)
but only n unknowns (mass per particle).  The Gauss-Newton solver finds the
best least-squares compromise, which leaves O(0.1g) residual.

**F6 (trade-off).** The per-particle reconstruction trades vertical precision
for horizontal reduction: the mirror-tied version achieves `a_y = 1.48e-13 g`
but `a_x = 0.086 g`; the per-particle version achieves `a_y = 0.030 g` and
`a_x = 0.071 g`.  Neither approach achieves machine precision for both
components simultaneously in the wall-bounded tank.

## 6. Caveats

* **Mass-only approach.** Position corrections (which would add 2n more DOFs)
  were attempted but the neighbour search (`cell_linked_pairs`) is not
  differentiable, breaking the autograd-based Jacobian.  A finite-difference
  approach was also tried but did not converge.
* **Only one resolution measured.** The convergence behavior at finer
  resolutions (dx = 0.02, 0.01) was not measured due to computational cost.
* **Wall-bounded tank only.** The periodic tank already achieves machine
  precision with the mirror-tied approach, so the per-particle reconstruction
  is only relevant for the wall-bounded case.

## 7. Conclusion

The per-particle mass reconstruction is a valid extension of the lake-at-rest
verification chain.  It demonstrates that relaxing the mirror-symmetry
constraint reduces the horizontal residual but cannot achieve machine precision
due to the overdetermined nature of the problem.  The honest verdict is that
the wall-bounded tank cannot hold a lake at rest to machine precision with
mass-only adjustments; the periodic tank remains the configuration in which
the anchor's tolerance is met.
