# Contact Angle Boundary Condition for Surface Tension

**Rewritten**: m2403 (2026-11-03) — the previous revision of this file documented
a feature that did not exist in the code. See *Audit finding* below.

## Physical background

When a fluid interface meets a solid wall the **contact angle** θ fixes how the
interface leaves the wall. It is a property of the three surface tensions at the
contact line (Young, 1805):

```
σ_sv = σ_sl + σ_lv · cos(θ)
```

- θ < 90°: hydrophilic (wetting) — the liquid spreads on the wall
- θ = 90°: neutral — the interface meets the wall perpendicularly
- θ > 90°: hydrophobic — the liquid beads up

## Audit finding (m2403)

`tests/test_contact_angle.py` imports `apply_contact_angle_boundary`, this
document described it in five numbered steps and listed ten passing tests, and
`Config.contact_angle` was a documented knob — but

1. `apply_contact_angle_boundary` **did not exist**, so the test module failed at
   *collection*: the repository's test suite was red;
2. `src/tensorsph/__init__.py` re-exported the missing name, so
   `import tensorsph` worked only because `surface_tension.py` never imported it —
   the export line was dead;
3. `rhs()` has always called
   `compute_surface_tension_force(pos, rho, mass, fluid_pairs, cfg, wall=wall)`,
   while the function was defined as `(pos, density, mass, pairs, cfg)`: **every**
   wall-bounded run with `surface_tension_coefficient > 0` raised `TypeError`;
4. two of the ten tests quoted here (`test_hydrophilic_angle_tilts_toward_wall`,
   `test_hydrophobic_angle_tilts_away_from_wall`) did exist but asserted only the
   array shape, not the angle, so nothing scored the physics.

So the merge reported "contact angle boundary condition: implemented" was a
phantom feature: documentation and export list ahead of code, and no runnable
path from `Config` to the solver. This is the same failure class as the
`mass_conserved` gate that compared `0.0` with itself (`changes-0.5.0.md`).

## Implementation (now real)

Module: `src/tensorsph/wetting.py`, re-exported from
`tensorsph.surface_tension` (the documented import path).

With `w` the wall normal pointing **into** the fluid and `n` the outward
interface normal (`n = -∇c/|∇c|`, colour field 1 inside the liquid), Young's
condition is

```
n · w = cos θ                                                   (1)
```

Check on a sessile circular cap (centre at height `y_c`, radius `R`): at the
contact point `n · w = -y_c/R`, which gives `θ = 90°` for the hemisphere
(`y_c = 0`), `θ → 0` for a flat puddle (`y_c → -∞`, `R ≈ |y_c|`) and `θ → 180°`
for a non-wetting sphere (`y_c ≈ R`). Every limit matches `cos θ`.

The near-wall colour gradient is replaced by

```
∇c ← -|∇c| · (cos θ · w + sin θ · t̂),      t̂ = normalise(n - (n·w) w)     (2)
```

so the magnitude is preserved exactly (the surface delta function `δ_s = |∇c|`
and the force balance are untouched), θ = 90° reproduces the tangential
component of the incoming normal, and the original azimuth around `w` is kept —
the model rotates, it does not invent a direction.

`w` is **measured from the wall geometry**, not assumed: the kernel-weighted mean
of the fluid-minus-wall offsets inside two smoothing lengths. A particle enclosed
by wall particles on opposite sides cancels to zero and is reported as "no wall
normal" and left unchanged, rather than given an arbitrary axis.

Steps:

1. `wall_normals(pos, wall, h)` — cell-linked search, `w` and the nearest-wall
   distance for every fluid particle (O(N) with cells, not O(N·M));
2. band = 2 h, the support of the cubic-spline kernel, where a wall particle can
   still influence the colour field;
3. project the outgoing normal onto the wall plane, renormalise, rebuild with
   (2), restore the magnitude;
4. particles outside the band, with a vanishing gradient, or with a normal
   parallel to `w` are unchanged — the condition is undefined there and is not
   faked.

Wiring: `rhs()` already passed `wall=` into the force; the force now accepts it
and applies `apply_contact_angle_boundary` when `cfg.contact_angle is not None`.

## Usage

```python
from tensorsph import Config
cfg = Config(surface_tension_coefficient=0.072,   # water, N/m
             contact_angle=math.pi / 3)           # 60 degrees, wetting
```

`contact_angle=None` (default) leaves the wall boundary untouched, bit for bit.

## Verification

`benchmarks/contact_angle/run.py` → `benchmarks/contact_angle/validation_report.json`:
11 contact angles × 9 wall orientations (0°…170°), one particle at 0.5 h above a
planar wall lattice, float64, incoming normal oblique to the wall.

| quantity | analytic | measured | verdict |
|---|---|---|---|
| θ from `acos(n·w)` | prescribed θ | max abs. error **5.7e-14 deg** (worst case of 99) | pass, `< 3 %` for every case (worst relative error 5.7e-13) |
| `|∇c|` before/after | preserved | max abs. error **4.4e-16** | pass, `< 1e-12` |
| wall normal recovered | `R(φ)·ŷ` | max component error **< 1e-16** over 9 orientations | the normal is measured, not `+y` |
| tangential azimuth | preserved | cos = 1 to 1e-12 | the condition rotates only |

The error is round-off, i.e. the construction is exact and the 3 % platform bar
is met by 12 orders of magnitude (`abs_error/θ < 5.7e-13` in every case).

`tests/test_wetting.py` — 50 cases pinning the above (angle sweep × rotations,
magnitude, azimuth, wall-normal recovery, inert-when-undefined, far field, empty
wall, and the `wall=` keyword on the force path).

## What is *not* verified, and why

The end-to-end equilibrium contact angle of a sessile drop under the force field
is **not** verified here: that requires the CSF curvature and colour field to
reproduce the Laplace law, and they do not —
`docs/verification_surface_tension_laplace.md` measures `Δp` errors of −77 %…−118 %
and curvature errors of +322 %…+1148 %. With a defective curvature field, any
measured drop shape would score the curvature defect, not the wetting condition.
The two integration tests in `tests/test_contact_angle.py` that demand a non-zero
surface-tension acceleration on a synthetic six-particle cloud are therefore
`xfail(strict=True)`: on that cloud the interior colour gradient vanishes, so the
curvature — and hence the force — is zero regardless of θ. They turn into a
failure the moment the force path starts producing a non-zero field, which is the
intended reminder.

Also not implemented: dynamic contact angle (velocity dependence), contact-line
pinning/hysteresis, and 3-D wall normals (the solver is 2-D; `wall_normals` is
written for arbitrary dimension, but nothing exercises D = 3).

## References

1. Young, T. (1805). An essay on the cohesion of fluids. *Phil. Trans. R. Soc.* 95, 65–87.
2. Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992). A continuum method for modeling surface tension. *J. Comput. Phys.* 100(2), 335–354.
3. Tartakovsky, A. M., & Meakin, P. (2005). Modeling of surface tension and contact angles with smoothed particle hydrodynamics. *Phys. Rev. E* 72, 026301.
4. Adami, S., Hu, X. Y., & Adams, N. A. (2012). A generalized wall boundary condition for smoothed particle hydrodynamics. *J. Comput. Phys.* 231(21), 7057–7075.
5. Adami, S., Ihmsen, M., & Adams, N. A. (2012). A transport-velocity formulation for smoothed particle hydrodynamics. *J. Comput. Phys.* 231, 7057.