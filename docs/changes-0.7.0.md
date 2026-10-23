# 0.7.0 — the wall pressure was missing a term

**Version:** 0.6.0 → 0.7.0
**Date:** 2026-10-23
**Default physics:** changed, deliberately, and measured before and after.

## What was wrong

`wall_lattice` and the `rhs` wall treatment cited Adami, Hu & Adams (2012),
*J. Comput. Phys.* 231:7057–7075, but implemented only the first half of eq. (11):

```
p_w = ( sum_f p_f W + sum_f rho_f (g - a_w).(r_w - r_f) W ) / sum_f W
       ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       implemented     missing
```

Without the body-force term, `p_w` is the kernel-weighted *average* of the
neighbouring fluid pressure rather than its extrapolation to the wall face. At the
bottom of a pool the two differ by about `rho g dx / 2`, so the fluid nearest the
floor was under-supported: the pool never settled, and every free-surface
diagnostic was contaminated by the ringing it caused.

The gap analysis had carried this as P1-4, "the citation is stronger than the
code", with the note that it "only matters a little for hydrostatics". That note
was wrong, and 0.7.0 measures why.

## What changed

| Area | Change |
|---|---|
| Wall pressure | both terms of eq. (11) are evaluated. The walls are static (`a_w = 0`), so the new term is `-gravity * (y_w - y_f) * rho_f` folded into the same `index_add`, costing one gather and one multiply |
| Extrema precision | the all-steps `rho/rho0 - 1` extremum was computed elementwise in float32 near 1.0, losing ~5e-4 of relative precision. It is now reduced first and rescaled in float64. A test that compares the extremum against the sampled rows caught it |
| `rho_max_relative_error` | same cancellation, now computed from the float64 extrema |
| Docstrings | the wall docstring quotes eq. (11) in full and says both terms are evaluated |
| Tests | `tests/test_wall_pressure.py`, 3 tests → suite 90 → **93** |
| License | the repository had none, while its documentation discussed other projects' licence obligations. MIT, chosen by the maintainer: `LICENSE`, SPDX `license = "MIT"` and `license-files` in `pyproject.toml`; the wheel now carries `License-Expression: MIT` and the file itself |

## Measured

### B1 hydrostatics (`dx = 0.025`, `c0 = 20`, 4 s, averaged over the second half)

| quantity | 0.6.0 | 0.7.0 |
|---|---|---|
| interior pressure error | 1.09 % | 1.12 % |
| pressure fluctuation (mean over interior bands) | 67.9 % | **9.0 %** |
| worst interior band fluctuation | 90.2 % | 15.2 % |
| free-surface band error | −28.4 % | −31.0 % |
| surface drop after 4 s | 3.51 % | 3.15 % |
| residual maximum speed | 0.044 m/s | **0.013 m/s** |

The interior error is unchanged to two digits. What changed is that the pool is
quiet: it had been ringing because the wall was not holding the hydrostatic
equilibrium.

### B1 with `hydrostatic_init=True`

| initialisation | interior error 0.6.0 | interior error 0.7.0 |
|---|---|---|
| uniform mass | 1.09 % | 1.12 % |
| hydrostatic mass | 1.16 % | **0.62 %** |

0.6.0 concluded that a hydrostatic initial condition did not help, because the
residual error was a sloshing mode that outlived the run. With the wall fixed it
halves the interior error. The old conclusion was measured on a wall model that
was itself injecting the imbalance.

### B2 dam break (three-level ladder, 0.8 s, `c0 = 25`)

| `dx` | peak compression 0.6.0 | peak compression 0.7.0 | bulk 0.7.0 |
|---|---|---|---|
| 0.04 | 9.71 % | **8.98 %** | 2.35 % |
| 0.02 | 10.26 % | **8.98 %** | 2.66 % |
| 0.01 | 10.00 % | **8.61 %** | 1.93 % |

**The `density_compression` gate passes for the first time since it started
failing in 0.5.0**, and the 10 % limit was not touched. Observed order moved from
0.96 to 1.11; the coarse/medium L2 grew from 2.58 % to 4.83 % because the new
term is proportional to the spacing, so it corrects the coarse solution more —
which is what a first-order boundary correction should do.

### Compressibility sweep (`dx = 0.01`)

| `c0` | Mach | peak compression 0.6.0 | peak compression 0.7.0 | front L2 vs next |
|---|---|---|---|---|
| 25 | 0.20 | 10.0 % | 8.61 % | 0.014 % |
| 40 | 0.12 | 6.1 % | 4.76 % | 0.014 % |
| 60 | 0.08 | 4.0 % | 5.15 % | — |

The last step is no longer monotone. It is reported as measured: the impact
transient is a few milliseconds long, none of these runs resolve it, and letting
it stand as a clean function of `c0` would be a nicer story than the data
supports.

### Free-surface comparison (re-measured, see the benchmark document)

Fixing the wall made the density-diffusion term measurable for the first time:
its effect on the B1 pressure fluctuation went from −3 % to **−24 %** (9.0 % →
6.8 %), while its interior-error effect remains within the band-choice spread and
its B2 effect remains negligible. **The default stays `density_diffusion = 0`**;
the reasoning is in [`benchmarks/free-surface-treatment.md`](benchmarks/free-surface-treatment.md).

## Honest gaps

- The default path is no longer bit-identical to 0.6.0. That is the point of the
  release, and the before/after numbers above are the record.
- The B2 density gate now passes with about one percentage point of margin. The
  impact transient is not resolved at any of the three ladder resolutions, so the
  margin is a property of the discretisation, not of the physics.
- `torch.compile` remains unmeasured (no C++ toolchain in this environment).
- The peak-compression transient still has no experimental reference; the
  `experimental_reference` gate stays `not_evaluated`.
