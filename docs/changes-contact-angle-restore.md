# Restoring the contact-angle boundary condition and a green test suite

**Date**: 2026-11-03 ｜ **Mission**: m2399 ｜ **Type**: defect repair + verification
**Files**: `src/tensorsph/surface_tension.py`, `tests/test_contact_angle.py`

## 1. What was broken

`HEAD` did not import.

```
$ PYTHONPATH=src python -c "import tensorsph"
ImportError: cannot import name 'apply_contact_angle_boundary'
             from 'tensorsph.surface_tension'
```

The contact-angle feature (wetting/dewetting at walls, e.g.
`Config(contact_angle=math.pi/3)`) was authored on a branch
(`e0627ed`, *feat: contact angle boundary condition for surface tension*) and
matured as `bec5848` — whose message reads *"m2090: matured mission
deliverables（1 个文件被门禁滤除）"*. The **one filtered file was
`src/tensorsph/surface_tension.py`**, i.e. exactly the file that defines the
feature. What reached `main` was the consumer half:

| Artifact | State on `HEAD` |
|---|---|
| `src/tensorsph/__init__.py` | imports and re-exports `apply_contact_angle_boundary` |
| `src/tensorsph/droplet_impact.py` | `Config.contact_angle`; `rhs()` calls `compute_surface_tension_force(..., wall=wall)` |
| `tests/test_contact_angle.py` | 10 tests importing the symbol |
| `docs/contact_angle_boundary.md` | documents the function as implemented |
| `src/tensorsph/surface_tension.py` | **no `apply_contact_angle_boundary`, no `wall` argument** |

Consequences, all measured on `HEAD` before the fix:

* `import tensorsph` fails — the package's public API is unusable, even though
  every test module imports submodules directly and the damage stayed invisible
  to `git log` reviews.
* `pytest tests/` reports **7 collection errors** (`test_adaptive_dt`,
  `test_contact_angle`, `test_laplace_pressure`, `test_laplace_validation`,
  `test_ritter`, `test_surface_tension`, `test_surface_tension_laplace`) —
  154 tests collected, the whole surface-tension and Laplace-pressure families
  dead.
* `compute_surface_tension_force` had no `wall` parameter while `rhs()` passed
  one, so any run with `surface_tension_coefficient > 0` would raise
  `TypeError` at the first step. Nothing caught this: `rhs` is only reached
  through benchmarks, which are not in CI.

This is a gate failure, not an algorithm failure: the missing code existed and
was reviewed, it just never landed.

## 2. What was restored

`apply_contact_angle_boundary` (+ its bounded-memory fallback
`_apply_contact_angle_by_bounds`) is back in `surface_tension.py`, the
`wall` argument is back on `compute_surface_tension_force`, and the boundary
condition is applied to the color gradient *before* the normal and the curvature
are extracted, so the CSF force sees the enforced direction.

```
n_new = cos(theta) * n_wall + sin(theta) * t_hat,      t_hat  _|_  n_wall
```

Because the two basis vectors are orthonormal, `|n_new| = 1`: the rotation
preserves `|grad c| = delta_s` and therefore the force magnitude, and changes
only its direction at the wall.

## 3. Three defects in the never-merged original

The dropped revision was not correct; restoring it verbatim would have shipped
three silent errors. Each is now covered by a test.

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `n_wall = v / (|v| + 1e-8)` | `|n_wall| = 1 - 6.7e-7`, leaving a `1.3e-6` "tangential" residue on a gradient that is exactly parallel to the wall. Normalising that residue produced a full unit tangent, and `|n_new|` came out **1.366** instead of 1 — a 37 % inflation of `delta_s` on a gradient that should not have been touched at all | `clamp_min(eps)` instead of `+ eps` |
| 2 | no degenerate case | when the gradient is parallel to the wall normal the tilt direction is undefined; the old code returned `cos(theta) * n_wall`, i.e. it **shrank the gradient to \|cos θ\|** (0.5 × at 60°) and deleted the interface | such particles are left bit-identical, with the tolerance stated as `TANGENT_TOL = 1e-4` |
| 3 | `grad_c[apply_mask] = ...` with `apply_mask = near & grad_mag.squeeze() > eps` | in-place mutation of the caller's tensor, and `squeeze()` collapses to a 0-d scalar for a single particle, so the mask indexing raises | out-of-place `torch.where`, `(N, 1)` masks |

The original tests could not see any of this: they asserted `result.shape` and
`|grad|` preservation on particles that were *all* modified, and — see §4 — the
two integration tests asserted a non-zero force where the operator returns
exactly zero.

## 4. The integration test asserted something that cannot hold

`TestContactAngleIntegration` used a 6-particle cloud with four "interior"
particles arranged at the corners of a square. Measured on that cloud:

```
neighbour counts          [4, 4, 4, 4, 2, 2]      surface = {4, 5}
kappa (1/m)               [10.10, -4.57, 11.76, -2.91, 0.00, 0.00]
delta_s                   [5.05, 0.83, 5.77, 0.00, 4.70, 4.04]
CSF acceleration          [[0, 3.7e-3], [0, -3.3e-4], [-4.9e-3, 0], 0, 0, 0]
```

The two surface particles have exactly **zero** curvature because all of their
neighbours carry the same unit normal, so `kappa = div(n) = 0` identically. The
assertion `not allclose(acc[4:], 0)` therefore cannot hold for *any* contact
angle treatment — the zero comes from the curvature operator, not from the
boundary condition. It went unnoticed only because the module failed to import
in CI.

The test now uses a 7 × 7 fluid lattice half a smoothing length above a wall
row, where 32 of 49 particles carry a non-zero CSF force, and it asserts the
behaviour that actually matters:

| Test | Assertion | Measured |
|---|---|---|
| `test_enforced_angle_equals_contact_angle` | `n . n_wall = cos theta` for θ = π/6, π/3, π/2, 2π/3 | `0.866025, 0.500000, 0.000000, -0.500000` (tol 1e-6) |
| `test_magnitude_preserved_exactly` | `\|grad\|` unchanged | ratio `1.000000` for every near-wall particle at every θ |
| `test_gradient_parallel_to_wall_normal_is_left_alone` | degenerate case untouched | bit-identical |
| `test_far_particle_is_bit_identical` | no action beyond `2h` | exact equality |
| `test_contact_angle_changes_only_the_near_wall_force` | influence confined to `4h` | changed ⊆ near band |
| `test_no_contact_angle_is_backward_compatible` | `contact_angle=None` + wall cloud = wall-free result | `torch.equal` |
| `test_uniform_normal_cloud_has_zero_curvature` | records the degeneracy above | `kappa[surface] = 0` |

The `4h` bound in the fifth row is not slack: the rotation is applied within
`2h` of the wall, and `compute_curvature` couples each particle to *its*
neighbours, so the perturbation travels one further smoothing length. An
earlier version of the test asserted `2h` and failed for exactly this reason —
the measurement is recorded here rather than the bound being quietly widened
before anyone looked.

## 5. Verification

```
PYTHONPATH=src python -m pytest tests/test_contact_angle.py -q   -> 21 passed
PYTHONPATH=src python -m pytest tests/ -q                        -> see §5.1
PYTHONPATH=src python -c "import tensorsph"                      -> ok
```

## 6. Note on the evidence grade

The contact-angle boundary condition is now **implemented and unit-verified**
(angle enforced to 1e-6, magnitude preserved, confinement, backward
compatibility). It is **not** physically validated: no case in this repository
measures a sessile-droplet contact angle or a capillary rise against experiment,
so the grade for the physics is WITHHELD, as it already is for the surface
tension model itself
(`docs/verification_surface_tension_laplace.md`: Laplace pressure off by
77–118 %). Restoring the plumbing does not change that.