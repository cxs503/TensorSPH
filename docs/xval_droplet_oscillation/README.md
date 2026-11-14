# Dynamic reading of the surface-tension force: a 2-D drop capillary oscillation
(XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION)

**Repository** TensorSPH · mission m20175 · every number below is injected from
`benchmarks/surface_tension/out_droplet_oscillation/oscillation.json`
(generator: `benchmarks/surface_tension/_make_doc.py`), zero hand-copy.

## 1. Why this run exists

All previous rounds of this anchor are **static**: the colour gradient, the
stress tensor and the force field are integrated across a *fixed* interface and
compared with Delta p = sigma/R. A static reading cannot see whether the force
that is actually integrated in time converts surface energy into kinetic energy
at the right rate. Upstream SPHinXsys' own acceptance for its surface-tension
model is a shape regression of a relaxing square droplet, which is insensitive
to |sigma|, so the dynamic reading is missing on both sides.

## 2. Exact reference

Inviscid oscillation of a 2-D drop (Rayleigh 1879; Lamb, *Hydrodynamics*, 6th ed.
arts. 313-315): `omega_m^2 = sigma m (m^2-1) / (rho R^3)`; its `m = 1` branch is
the neutral translation mode, so the relation is anchored at zero independently
of any measurement. `m = 2` is used because it preserves the area to first
order. Reference values for this case (sigma = 0.072 N/m,
rho0 = 1000.0 kg/m^3, R = 0.05 m):
**T_2 = 3.3798 s**, and Delta p = sigma/R =
**1.4400 Pa** for the EOS-carried jump.

## 3. Setup

production CSF (cfg.csf_scheme='consistent'), free surface, WCSPH Tait EOS c0=0.5 m/s, gravity off, adaptive dt (CFL 0.25), density_calibration on; eps = 0.05; free surface (vacuum outside, no walls);
particle masses from `density_calibration` so the initial state carries no
startup pressure; dt from the CFL limit recomputed every step. Readings: period
of the mode-2 order parameter `q2` from its zero-crossing spacing (immune to the
damping envelope), and the mean pressure of the core (`r < R/2`).

**Containment audit** (this is the reading the earlier attempts at this same
benchmark in `benchmarks/oscillating_droplet/` did not gate on):
15.0 % is irrelevant if particles leave the drop.

| quantity | definition |
|---|---|
| max r/R | largest particle radius during the run, divided by R |
| escaped | number of particles that ever went beyond 1.6 R |
| t_esc | time of the first escape |

## 4. Results (single-resolution (R/dx = 10) diagnostic matrix over one Rayleigh period; resolution ladder submitted as cluster job cj2559, still queued when this matrix was produced)

| config | R/dx | N | steps | T measured [s] | T exact [s] | period err | p_in [Pa] | dp err | max r/R | escaped | t_esc [s] | contained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 10 | 315 | 1040 | 3.0499 | 3.3798 | -9.76 % | 1.0303 | -28.45 % | 4.261 | 39 | 1.729 | NO |
| viscous_5e-4 | 10 | 315 | 1040 | nan | 3.3798 | +nan % | 0.4061 | -71.80 % | 1.020 | 0 | nan | yes |


## 5. Verdict

**NOT ESTABLISHED -- honest FAIL of the dynamic reading, and the failure is in
the free-surface treatment, not in the surface-tension magnitude.**

* `baseline` is the configuration a reader would call "the production setup".
  It loses containment at t = 1.729 s (about
  51 % of one Rayleigh period):
  39 particles are thrown beyond 1.6 R, the largest reaching
  4.26 R. Its period reading is
  -9.76 % and it *appears* to pass the 15 % gate -- but a
  quarter of the oscillation it would have to describe happens after the drop has
  already disintegrated, so the number is an artefact of the escape transient,
  not a measurement of the CSF force. This is exactly the failure mode recorded
  in `benchmarks/oscillating_droplet/` (particle left the box at 0.44 T, errors
  -32 % ... -62 %): the sign and size of the recorded error are set by the
  escape, which is why two setups of the same physics disagree in *both* signs.
* `viscous_5e-4` keeps containment perfectly (max r/R =
  1.020, zero escapes) but then the drop does not
  perform a capillary oscillation at all within one period (0 zero crossings of
  `q2`, order parameter drifting from 0.0853 to
  0.1208). Containment was bought at the price of the signal.
* Neither row can therefore be used to claim that TensorSPH's surface-tension
  force reproduces the Rayleigh-Lamb frequency within 15 %; and neither can be
  used to claim that it does not. The recorded **readings are not measurements**.
* The only defensible dynamic-adjacent evidence in the repository remains the
  linear-response (modal stiffness) reading of
  `benchmarks/oscillating_droplet/validation_report.json`, which deliberately
  avoids time integration.

**What has to be fixed before this benchmark can decide anything**: the free
surface needs a treatment that keeps the interface particles attached
(transport-velocity / particle shifting, or the upstream SPHinXsys wall +
contact-relation construction), and the containment audit above must be a gate
of every dynamic surface-tension benchmark from now on.

## 6. Pitfalls found while doing this

1. **A plausible number is the most dangerous outcome.** The baseline period
   error (-9.76 %) is inside the gate and is *wrong*. Only the containment audit
   showed it.
2. **Do not compare total numbers across setups**: the previous attempt
   (`benchmarks/oscillating_droplet/`, R = 0.02 m, sigma = 7.2 N/m, fixed dt,
   hex lattice, eps = 0.1) reports errors of -32 % ... -62 %; this one reports
   -9.8 %; both are escape artefacts, and neither is grid-converged.
3. **Viscosity as a stabiliser destroys the signal**: nu = 5e-4 m^2/s (500x
   water) suffices to keep the drop intact, and removes the oscillation
   entirely -- the drop relaxes instead of oscillating.

## 7. Artefacts

| path | content |
|---|---|
| `src/tensorsph/capillary_oscillation.py` | reusable case/exact-reference/estimator/containment driver |
| `benchmarks/surface_tension/droplet_oscillation.py` | the matrix driver |
| `benchmarks/surface_tension/out_droplet_oscillation/oscillation.json` | every number above |
| `benchmarks/surface_tension/_make_doc.py` | doc + figure generator (this file is generated) |
| `tests/test_droplet_oscillation.py` | 14 invariants (Rayleigh reference, order parameter linearity, damping-immune period estimator, EOS pressurisation) |
| `docs/xval_droplet_oscillation/fig_oscillation.png` | figure: q2(t) vs the exact cosine, and gate status |
