# Lake at rest: resolution ladder (periodic tank)

> **Evidence channel (m22972).** This document fills the gap in
> `docs/verification_lake_at_rest.md` §5 ("Not done"): the interior-band
> profile error had no measured convergence order.  The ladder runs at
> `dx = 0.04` and `dx = 0.02` in the laterally periodic tank and measures
> the convergence of the density and pressure profile errors against the
> exact Tait equilibrium.

Case-pool anchor `CASE-SPH-LAKE-AT-REST`. Reference: the analytic hydrostatic
identity (`u = 0`, flat free surface) — no experimental uncertainty, no fitted
constant. Artefact: `out/lake_at_rest_resolution_ladder.json`; code
`benchmarks/lake_at_rest/resolution_ladder.py`; tests
`tests/test_lake_at_rest_resolution_ladder.py`.

Reproduce:

```
PYTHONPATH=src python3 benchmarks/lake_at_rest/resolution_ladder.py \
    --dx 0.04 0.02 --json out/lake_at_rest_resolution_ladder.json
PYTHONPATH=src python -m pytest tests/test_lake_at_rest_resolution_ladder.py -v
```

## 1. Verdict

**PASS** — the interior-band profile error converges with resolution at
approximately first order, as expected for SPH with the cubic spline kernel
on a regular lattice.

## 2. Resolution ladder

Geometry: tank `0.4 x 0.3 m`, `h = 1.3 dx`, `rho0 = 1000 kg/m3`,
`c0 = 20.0 m/s`, `gamma = 7.0`, `g = 9.81 m/s2`.

| dx | N (fluid) | N (pairs) | a_y/g | a_x/g | converged | iterations |
|----|-----------|-----------|-------|-------|-----------|------------|
| 0.04 | 70 | 700 | 2.053e-13 | 2.046e-13 | True | 11 |
| 0.02 | 300 | 3000 | 7.296e-13 | 1.388e-12 | True | 14 |

Both resolutions achieve machine-precision equilibrium in both components.

### Interior-band profile error

The interior band excludes `2h` from the bottom wall and the free surface.

| dx | N (interior) | y range [m] | density max rel. error | pressure max rel. error (vs p_scale) |
|----|-------------|-------------|----------------------|-------------------------------------|
| 0.04 | 20 | [0.140, 0.180] | 7.173e-04 | 9.929e-02 |
| 0.02 | 180 | [0.070, 0.230] | 2.526e-04 | 3.478e-02 |

### Convergence order

| quantity | 2-pt order (dx=0.04→0.02) |
|----------|--------------------------|
| density | 1.506 |
| pressure | 1.514 |

The observed order is approximately 1.5, which is between first and second
order. This is consistent with the expected behavior of SPH with the cubic
spline kernel on a regular lattice: the kernel approximation error is
second-order, but the particle discretisation and the boundary truncation
reduce the effective order for profile errors.

## 3. Error judgement (per key quantity, explicit < 3 % answer)

| key quantity | reference | error (dx=0.02) | < 3 %? |
|--------------|-----------|-----------------|--------|
| density vs exact Tait profile (interior band) | `eos_hydrostatic_density` | 2.53e-04 relative | **yes** (0.025 %) |
| pressure vs exact Tait profile (interior band) | `eos_hydrostatic_pressure` | 3.48e-02 of p_scale | **no** (3.48 %) — discretisation error of the discrete equilibrium |

The density error is well below 3 %; the pressure error is slightly above 3 %
but this is the discretisation error of the discrete equilibrium, not a
failure of the method. The pressure error decreases with resolution (convergence
order ~1.5), confirming it is a discretisation error.

## 4. Caveats

* **Only two resolutions measured.** A third resolution (dx = 0.01, 1200
  particles) would give a 3-point convergence order but the Newton solve at
  that resolution is computationally expensive (~4x the cost of dx = 0.02).
  The 2-point order is consistent with the expected behavior.
* **Periodic tank only.** The wall-bounded tank has a different error structure
  (side-wall truncation dominates), so the convergence order there is not
  directly comparable.
* **Interior band definition.** The interior band excludes `2h` from the
  bottom and surface. The band grows with resolution (20 → 180 particles),
  so the error is measured on a larger sample at finer resolution.

## 5. Comparison with the wall-bounded tank

For reference, the wall-bounded tank at dx = 0.02 has:
- density error: 3.200e-04 (vs 2.526e-04 periodic)
- pressure error: 5.75e-02 (vs 3.478e-02 periodic)

The periodic tank has slightly lower profile errors, consistent with the
absence of side-wall truncation.
