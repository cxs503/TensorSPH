# Contact angle on a wall: does `Config.contact_angle` give Young's law?

**Status: WITHHELD — the boundary condition is implemented and the operator is
correct, but the emergent macroscopic angle does not follow the prescribed
angle.**  This document is the measurement that supports that verdict.

## 1. What is being verified

`src/tensorsph/csf_consistent.py::apply_contact_angle_boundary` rotates the
interface normal near a wall so that `n . n_wall = cos(theta)` (Adami, Hu &
Adams 2012).  The capability has been advertised since `docs/contact_angle_boundary.md`
on the strength of *operator* tests: the rotation formula, applied to
hand-built gradient fields.  The physical claim a user reads into
`Config(contact_angle=...)` is different: *a droplet resting on that wall will
show that contact angle*.  Young's law is then the reference:

    sigma_sv = sigma_sl + sigma_lv cos(theta)

For a 2-D drop (a cylinder in 3-D) the equilibrium shape on a flat wall is a
circular cap, so the angle is recoverable from geometry alone.

## 2. Case and method

| item | value |
|---|---|
| geometry | droplet on a flat wall, `gravity = 0` (the wall is a BC only) |
| wall | 3 static layers, Adami pressure extrapolation |
| initial state | **half disc, `theta_init = 90 deg` — identical for every case** |
| radius / box | `R = 0.06 m`, `pool_width = 0.30 m` |
| spacing / smoothing | `dx = 0.005 m`, `h = 1.3 dx`, 263 fluid particles |
| surface tension | CSF `consistent` scheme, `csf_curvature = mls`, `sigma = 0.072 N/m` |
| viscosity / sound speed | `nu = 0.02` (code units), `c0 = 10 m/s` |
| step / duration | `dt = 1.63e-4 s` (acoustic Courant 0.25), 2154 steps, `t = 0.35 s` |
| sweep | `contact_angle` in {30, 60, 90, 120, 150} deg **+ control (BC off)** |

Why the design matters: if each case started *at* its target angle, a solver
that did nothing at all would score zero error.  Every case therefore starts
from the same 90 deg half disc, and only `Config.contact_angle` changes, so the
control run (`contact_angle=None`) is the causal baseline.

The angle is measured from the particle cloud, never from the CSF field (that
would be circular — it is what the BC writes into):

* `r = H^2 / A`, with `H` the drop height and `A = N dx dy` (area is conserved
  by the solver to machine precision: measured change `0.0 %`);
* for a circular cap, `(1 - cos t)^2 / (t - sin t cos t) = r` is monotone in
  the contact angle `t`, so `r` inverts to one angle;
* the inversion table is built from synthetic caps of **known** angle on the
  same lattice and `dx`, smoothed and made monotone (a 2 deg cap grid is not
  monotone without that: lattice-cut noise put `r(58) > r(60) > r(62)`, which
  silently reported 53 deg for a known 60 deg cap), then anchored on the case's
  own initial state, whose angle is 90 deg by construction.  Measured zero
  point: **88.0 deg** (residual -2 deg);
* arrangement sensitivity of the estimator, from jittered and lattice-shifted
  reference caps: 2.4 / 4.2 / 4.7 / 2.0 / 18.2 deg at 30 / 60 / 90 / 120 / 150 deg
  — above ~135 deg the cap closes onto a disc and the height stops
  discriminating, which is why 150 deg is only weakly constrained.

A trimmed circle fit of the colour-threshold interface shell is reported as a
cross-check; it carries a known inward bias of about +6 deg at 90 deg and is not
used for the verdict.

## 3. Results

| prescribed | emergent | circle-fit cross-check | error | Drift over last 0.2 s |
|---|---|---|---|---|
| 30 deg | 107.2 deg | 114.1 deg | **+77.2 deg (+257 %)** | +7.5 deg |
| 60 deg | 107.3 deg | 106.9 deg | **+47.3 deg (+79 %)** | +7.6 deg |
| 90 deg | 108.5 deg | 115.5 deg | **+18.5 deg (+20.6 %)** | +7.6 deg |
| 120 deg | 109.4 deg | 117.1 deg | -10.6 deg (-8.8 %) | +3.2 deg |
| 150 deg | 110.3 deg | 118.4 deg | **-39.7 deg (-26.5 %)** | +3.7 deg |
| **control (no BC)** | **110.7 deg** | 118.7 deg | — | +4.0 deg |

Peak Mach 0.005, peak bulk density error < 0.1 %, area change 0.0 %, zero
boundary events — the *solver* is healthy; it is the boundary condition that
does not deliver.

Reading the table:

1. **Every prescribed angle lands within 3.1 deg of every other.**  The BC has
   a real but tiny effect, and its sign is right: the emergent angle decreases
   monotonically as the prescribed angle decreases, and all BC cases sit below
   the 110.7 deg control.  The *size* is two orders of magnitude too small
   (1.4 deg of shift for a 30 deg request).
2. **Only the 120 deg case is inside the 15 deg tolerance, and that is a
   coincidence**: the control sits at 110.7 deg, so any request near 110 deg
   "passes" whatever the BC does.
3. The dominant error is a **common drift of about +19 deg from the 90 deg
   initial state**, present with the BC off as well.  It is not settled at
   `t = 0.35 s` (still +7.5 deg over the last 0.2 s), so the run is a
   transient, not an equilibrium.

## 4. Root causes found while measuring

**(a) The BC was being averaged away by the model's own normal smoothing.**
`csf_fields` rotated `grad_c` *before* the two-pass Shepard smoothing of the
normal field.  That average spans the whole 2h support, so the rotated normals
were mixed with unrotated interface neighbours.  Measured on the resting
90 deg drop at `dx = 0.004`: for a prescribed 30 deg, the normal at the contact
corner came out at `n . n_wall = -0.78` (~141 deg) instead of `+0.87`.  Fixed in
this branch by rotating the *smoothed* normal (`contact_angle_normals`), which
preserves the documented invariants (`|grad c|` untouched, gradient rebuilt from
the rotated normal) and leaves every `contact_angle=None` path bit-identical.
After the fix the corner normal does meet the prescribed angle within the
operator's linear blend; `tests/test_contact_angle_validation.py` pins that.

**(b) Even with (a) fixed, the rotation is spread over the whole wetted base.**
The colour function of a single-phase cloud cannot tell the free surface from
the wetted base: the bottom row under the drop is "interface" (its kernel sum is
~0.5 `rho0`) with a normal pointing straight into the wall, and there the
tangent of the rotation is degenerate (any in-wall direction is perpendicular to
the normal), so the operator picks one by sign convention rather than by the
local interface orientation.  For a 90 deg drop the target normal is exactly the
tangent, so the BC's rotation is nearly orthogonal to the field it must control.
What survives is a small, correctly signed tilt — the 1.4 deg effect above.

**(c) Weakly-compressible SPH cannot equilibrate a static drop at these
settings.**  The Laplace pressure `sigma/R` is 1.2 Pa, which in the Tait EOS is
a density signal of `1.7e-6` — the same unresolvability the repo's own Laplace
benchmark documents (it integrates the force rather than evolving it).  The
remaining free-surface density deficit (kernel sum ~0.5 `rho0`) drives spurious
currents of ~0.05 m/s, which is 7x the capillary velocity scale `sigma/(rho R^2)
x t`, so the drop's shape is spurious-current-driven and the common +19 deg
drift follows.

## 5. Verdict and what it would take

| criterion | required | measured | result |
|---|---|---|---|
| emergent vs prescribed angle, 30 deg | within 15 deg | +77.2 deg | **fail** |
| emergent vs prescribed angle, 60 deg | within 15 deg | +47.3 deg | **fail** |
| emergent vs prescribed angle, 90 deg | within 15 deg | +18.5 deg | **fail** |
| emergent vs prescribed angle, 120 deg | within 15 deg | -10.6 deg | pass (coincidental: control is at 110.7 deg) |
| emergent vs prescribed angle, 150 deg | within 15 deg | -39.7 deg | **fail** |
| BC effect distinguishable from control | > 5 deg | 3.5 deg (largest) | **fail** |
| solver health (area, Mach, escapes) | conserved / Mach < 0.1 / 0 | 0.0 % / 0.005 / 0 | pass |

`Config.contact_angle` should therefore be advertised as **experimental /
unverified** until (b) and (c) are addressed.  The concrete next steps, in order:

1. Apply the contact condition only at the contact line, using an explicit
   contact-line force (`f = sigma (n_wall +/- cos theta n) delta_s`) instead of
   rotating normals across the whole wetted base — this removes the degenerate
   tangent of (b);
2. give the model a colour function that includes the solid phase, so the
   wetted base is not mistaken for free surface;
3. re-run this benchmark on a scheme that can hold a static drop (higher `c0`
   with a longer run, or a pressure-Poisson/isobaric SPH variant), and only then
   re-test the 15 deg tolerance.

## 6. Reproduction

    PYTHONPATH=src python benchmarks/contact_angle/run.py \
        --dx 0.005 --seconds 0.35 --write-json benchmarks/contact_angle/results.json \
        --figures-dir benchmarks/contact_angle

    PYTHONPATH=src python -m pytest -q tests/test_contact_angle_validation.py

Artifacts: `benchmarks/contact_angle/results.json` (per-case rows + estimator
calibration + per-angle uncertainty), `contact_angle_parity.png`,
`contact_angle_clouds.png`.

Note on cost: set `torch.set_num_threads(1)`.  With the 128 OpenMP threads of
this host one `rhs` call on 406 particles took 4.49 s against 0.0225 s
single-threaded; the whole sweep is ~5 min single-threaded.
