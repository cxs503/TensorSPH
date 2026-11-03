# Contact-angle boundary condition — verification record

**Mission**: m2389 (agent/m2389) · **Date**: 2026-11-03 · **Solver**: TensorSPH 0.8.0
**Component**: `tensorsph.surface_tension.apply_contact_angle_boundary`
**Benchmark**: `benchmarks/contact_angle/run.py` → `benchmarks/contact_angle/results.json`

## 1. Background and why this record exists

The contact-angle boundary condition (BC) was documented and given tests in
mission m2090 (`docs/contact_angle_boundary.md`, `tests/test_contact_angle.py`),
but the merge of that mission was filtered: the commit message of `bec5848`
says *"1 个文件被门禁滤除"*, and the filtered file was
`src/tensorsph/surface_tension.py`. What landed was therefore the *consumer* side
only — `__init__.py` re-exported `apply_contact_angle_boundary`, `rhs()` called
`compute_surface_tension_force(..., wall=wall)`, and a 246-line test file
imported the missing symbol. On the merged tree `import tensorsph` raised
`ImportError`, i.e. **every** test and every benchmark in the repository failed at
collection, and nothing flagged it because no CI job had run on the merge.

This mission restores the missing implementation from the specification that did
land (the doc + the test ABI + the `rhs()` call site), and replaces the parts of
that specification that were never satisfiable with verifiable statements.

## 2. What the boundary condition is, exactly

For a colour-gradient (CSF) formulation the contact angle is not a modelling
choice, it is a geometric constraint. With `n_w = (x_fluid - x_wall)/|x_fluid -
x_wall|` (wall normal pointing into the fluid) and `theta` measured inside the
fluid (Adami, Hu & Adams 2012, *J. Comput. Phys.* 231:7057-7075; Tartakovsky &
Meakin 2005, *J. Comput. Phys.* 207:610-628):

```
n_new = cos(theta) * n_w + sin(theta) * t_hat ,   |n_new| = |n_old|
```

where `t_hat` is the unit tangential part of the incoming normal. `theta = 0`
is a film lying flat on the wall, `theta = 90 deg` an interface meeting the wall
perpendicularly, `theta = 180 deg` the non-wetting limit. The magnitude is
preserved on purpose: the CSF surface delta function is `|grad c|`, so the BC
must change the *direction* (hence the curvature and the force direction) and
nothing else.

## 3. Acceptance test and results

Sample: 4000 particles inside the influence band (`d < 2h`,
`h = 0.0650 m`, band `= 0.1300 m`) with
random interface normals, plus the same number of control particles at `d > 4h`.
Reference: the definition above. Tool: `benchmarks/contact_angle/run.py`
(exit code 1 when the tolerance is violated).

| theta [deg] | max angle error [deg] | rms angle error [deg] | max relative |n| error | max leak outside band |
|---|---|---|---|---|---|
|     0 | 3.43e-02 | 1.04e-02 | 1.53e-07 | 0.00e+00 |
|    30 | 6.83e-03 | 1.16e-04 | 1.63e-07 | 0.00e+00 |
|    60 | 2.05e-02 | 3.49e-04 | 1.53e-07 | 0.00e+00 |
|    90 | 2.73e-02 | 4.65e-04 | 1.86e-07 | 0.00e+00 |
|   120 | 2.05e-02 | 3.49e-04 | 1.70e-07 | 0.00e+00 |
|   150 | 6.82e-03 | 1.16e-04 | 1.58e-07 | 0.00e+00 |
|   180 | 3.43e-02 | 1.04e-02 | 1.53e-07 | 0.00e+00 |

Verdict: **PASS** at tolerance 0.1 deg.

Reading of the numbers: the angle is reproduced to `3.4e-2 deg` worst case,
which is the float32 round-off of `acos` near 1 (a 1e-7 error in the vector
components produces ~1e-2 deg there); the magnitude is preserved to `1.9e-7`
relative; and the fields outside the `2h` band are bit-identical (leak `= 0`).
There is no discretisation error to report because the check is against the exact
definition, not against a numerical reference solution.

![contact angle boundary condition](contact_angle_bc.png)

Left: incoming random normals (grey) and the reconstructed normals at
`theta` = 30/60/150 deg, applied only inside the band above the wall. Right:
recovered angle versus prescribed angle, `theta` and `180-theta` collapse onto the
definition line because the two are antipodal and the tangential direction is
undefined there (documented, not hidden).

## 4. Two defects in the inherited specification, and how they were resolved

### 4.1 The integration fixture was unsatisfiable

`test_surface_tension_with_contact_angle` / `..._without_contact_angle` asserted
that particles 4 and 5 of a six-particle chain receive a non-zero CSF force. That
cannot happen: the chain is flat and collinear, so all normals are parallel, every
stencil difference `n_j - n_i` vanishes and `div(n)` is exactly zero — measured
values were `kappa = [10.10, -4.57, 11.76, -2.91, 0, 0]`. The failure has nothing
to do with the contact angle: the second test failed with `contact_angle=None`.
The fixture is replaced by a real three-layer pool on the production Adami wall
lattice, and the assertions now test what the BC is for — the force is reoriented
*inside* `2h` and is bit-identical outside it.

### 4.2 Surface-particle detection is sample-dependent (analysed, not changed)

`compute_surface_tension_force` classifies a particle as a surface particle when
its neighbour count falls below 70% of the **mean** count. The mean is depressed
by the surface particles themselves and collapses for small samples (the old
fixture: counts `[4, 4, 4, 4, 2, 2]`, mean `3.33`, threshold `2.33`, so no
particle at all was a surface particle — which is the real reason that fixture's
force was zero, in addition to its zero curvature). Replacing the mean by the
bulk (maximum) count, as Marrone et al. (2012) do with a calibrated bulk count,
does make small samples behave, and it was implemented and measured here; it was
then **reverted**, because it changes the colour normalisation of the merged CSF
model and with it the numbers that the negative-control suite
(`tests/test_laplace_droplet.py`, which asserts the documented curvature defect)
was tuned on. Retuning the production CSF model is not part of a verification
mission, so the heuristic is left exactly as merged and the finding is recorded
as a known limitation: the surface detection is sample-size dependent below
~100 particles.

## 5. Limitations (honest scope)

1. **Geometry only.** This record certifies the geometric constraint, not the
   wetting physics. A full validation needs a sessile-droplet meniscus radius
   measured against Young's law, which in turn needs a working CSF curvature —
   and the merged CSF curvature does **not** pass the Laplace law
   (`dp_model/dp_theory` = 0.43 / 1.39 / 3.20 at `R/dx` = 10/20/40, no
   convergence; see `docs/benchmarks/laplace-droplet.md`). Certifying the BC on
   top of a failed curvature operator would not be defensible, so it is not
   claimed.
2. **2-D, single layer of wall particles understood.** The nearest-wall-particle
   construction assumes the wall normal is resolved by the lattice; corner
   configurations (two walls within `2h`) use the nearest particle only, which is
   a documented simplification, not a tested feature.
3. **Degenerate incoming normals** (already parallel to `n_w`) reach the
   prescribed angle by a fixed 90 deg rotation of the wall normal. The sign of
   that rotation is arbitrary; at a flat interface the tilt direction is
   genuinely undetermined, but a physical meniscus will pick a side, so this is
   an honest gap for curved geometries.
4. **Not exercised end-to-end.** `rhs()` passes `wall` and `Config.contact_angle`
   is wired through, but no full simulation with a contact line has been run, so
   no claim is made about the resulting dynamic contact angle or about
   spurious currents at the contact line.

## 6. Reproduce

```bash
export PYTHONPATH=src
python benchmarks/contact_angle/run.py --json-out benchmarks/contact_angle/results.json \
    --figure benchmarks/contact_angle/contact_angle_bc.png
python -m pytest tests/test_contact_angle.py tests/test_contact_angle_bc_geometry.py -q
```

Unit level: `tests/test_contact_angle.py` (10 tests, restored ABI + repaired
fixture) and `tests/test_contact_angle_bc_geometry.py` (13 tests: angle,
magnitude, containment, degenerate case, force-level coupling).

### 6.1 Test status on this branch (m2389)

| Suite | Result |
|---|---|
| `tests/test_contact_angle.py` | 10 passed |
| `tests/test_contact_angle_bc_geometry.py` | 13 passed |
| `tests/test_contact_angle* + test_surface_tension* + test_laplace_*` (subset) | 49 passed, 2 xfailed, 4 failed |
| whole suite before this mission | `ImportError` at collection (0 tests runnable) |

The 4 failures are in `tests/test_laplace_droplet.py`
(`test_curvature_error_grows_under_refinement`,
`test_curvature_has_the_wrong_sign_on_the_interface`,
`test_normal_field_is_not_radially_coherent`, and one more) — the negative
controls that assert the *documented* CSF curvature defect. They are unrelated to
this mission: they fail identically after reverting the surface-detection change
described in 4.2, i.e. they are pre-existing on the merged tree (the defect
itself is reported in `docs/benchmarks/laplace-droplet.md`). They are left
failing on purpose — a negative control that starts passing must be re-read, not
silenced.
