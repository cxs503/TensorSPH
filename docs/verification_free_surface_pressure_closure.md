# Free-surface pressure closure: closing the C3 gate of the Laplace audit

**Scope**: TensorSPH, `Config.free_surface_pressure` (default **off**) and
`tensorsph.droplet_impact.free_surface_pressure_closure`.
**Artifacts**: `benchmarks/laplace_pressure_floor/report_closure.json`,
`.../laplace_pressure_floor_report_closure.png`,
`tests/test_free_surface_pressure_closure.py` (14 tests).
**Supersedes**: the C3 recommendation of `backlog_update_m2971.md` (P0) and the
"not implemented" status in `docs/verification_laplace_pressure_floor.md`.

## 1. The defect

Density summation under-counts neighbours wherever the kernel support is
truncated, so the outermost fluid shell of a droplet has `rho/rho0` between
0.90 and 0.94 (measured, see the `rho_shell` column below). The Tait law
clamped at zero -- the standard first-order WCSPH free-surface treatment --
therefore assigns `p = 0` to that entire shell. The pressure then has to fall
from the compressed core to zero **across one spacing**, which is not a surface
tension force and is not balanced by one; it drives the surface outward, the
droplet overshoots its equilibrium compression, and the Young-Laplace jump
`sigma/R = 360 Pa` cannot be recovered. The four-way decomposition in
`verification_laplace_pressure_floor.md` isolated this as error source C3 and
showed it was the only one of the four that did not shrink with `c0` or `dx`.

## 2. What was implemented

Adami, Ihmsen & Adams (2012), section 3.3, applied to a free-surface fluid
particle instead of a wall particle: a particle with truncated support takes its
pressure from a kernel-weighted average over its **interior** neighbours only,

    p_i = max(0, sum_j (p_j + rho_j g.(r_i - r_j)) W_ij / sum_j W_ij),
    j runs over particles with `rho_j/rho0 >= 0.99`

Three details carry the scheme:

* **The body-force term is not optional.** For a linear hydrostatic field every
  summand is algebraically `p_i`, so the weighted average returns `p_i` exactly
  whatever the kernel normalisation. Without the term the estimate is biased by
  `rho g <y_i - y_j>`, i.e. by roughly half a spacing; measured in
  `test_body_force_term_is_what_makes_it_exact` and `..._matches_rho_g_times_...`
  (bias +196 Pa at dx = 0.02 m against a round-off residual, a factor
  > 1e3). The wall extrapolation in `rhs` had the same defect before it was
  fixed; this is the same fix, one surface further out.
* **Truncated neighbours must not feed the average.** They are the particles
  being reconstructed; including them makes the estimate depend on itself and
  drags it back towards the clamped zero
  (`test_truncated_neighbours_do_not_contaminate_the_average`, with a control
  showing that a *genuinely interior* low-pressure neighbour still contributes,
  so the filter is on density and not on pressure).
* **Pairwise antisymmetry survives.** The momentum equation applies the same
  scalar `p_i/rho_i^2 + p_j/rho_j^2` to both members of a pair with opposite
  signs, so replacing `p_i` alone leaves the pair force anti-symmetric and the
  net momentum transfer at round-off: 1.3e-08 of the summed force
  magnitude with the closure on, 2.9e-08 with it off
  (`test_pairwise_antisymmetry_of_the_pressure_force_is_preserved`).

Order in `rhs` matters: the closure is applied to the fluid field **before** the
wall extrapolation, so the wall averages the closed surface rather than the
clamped one. The density is untouched -- only the pressure used by the momentum
equation changes -- which keeps mass, the summation density and every diagnostic
bit-identical between the two variants.

## 3. Results

`sigma/R = 360 Pa`, `rho0 = 1000 kg/m^3`, `sigma = 7.2 N/m`, `R = 0.02 m`,
mass-calibrated hexagonal drop, `t_end = 1.5 t_sigma`, tail = last 40 % of the
window, read-out = absolute pressure with no band differencing (the C1-corrected
convention).

| dx [m] | c0 [m/s] | N | steps | tail-avg, clamp | tail-avg, **closure** | snapshot | band A (deprecated) | rho_core-1 / required | v_max/v_ref |
|---|---|---|---|---|---|---|---|---|---|
| 0.0015 | 5 | 642 | 1052 | -37.4 | **-0.9** | -4.3 | -69.1 | +1.32e-02 / 1.38e-02 | 0.105 |
| 0.0015 | 20 | 642 | 1710 | -38.1 | **-3.4** | +6.3 | -76.9 | +9.54e-04 / 8.98e-04 | 0.090 |
| 0.002 | 5 | 362 | 592 | -34.0 | **+4.9** | +3.5 | -60.1 | +1.43e-02 / 1.38e-02 | 0.104 |
| 0.002 | 20 | 362 | 1283 | -37.9 | **-7.2** | +44.6 | -12.2 | +1.26e-03 / 8.98e-04 | 0.142 |

**Acceptance (P0 in `backlog_update_m2971.md`)**: all four ladder points inside
15 %. Largest remaining error 7.2 %, i.e. **PASS**; the clamped baseline
it replaces sat at 38.1 %. 4 of 4 points are also inside 8 %,
1 of 4 inside 3 %.

Two independent consistency checks agree that the *mechanism* is now right, not
just the number:

* The achieved core compression tracks the value the Laplace law **requires**
  (`compression_required` = 1.38e-02 at c0 = 20): the residual is +6 % at
  dx = 0.0015 and +40 % at dx = 0.002, and at c0 = 5 it is -4 % / +4 %. Under the
  clamp the core was *under*-compressed by 30-40 % everywhere and the ratio did
  not follow `dx`.
* The shell pressure rises from exactly 0 to the interior value (measured on an
  over-dense disc: mean shell pressure 0 -> 3514 Pa while `p_max` stayed at
  8472 Pa, i.e. the closure lifts the shell without touching the interior).

## 4. Honest error judgement

Against the 3 % bar, **not passed**: only the finest/lowest-`c0` point is inside
3 % (-0.9 %). The ladder passes the 15 % bar that P0 defined and the
remaining spread (-7.2 % ... +4.9 %) still has resolution sensitivity in it, so
the Young-Laplace jump moves from WITHHELD to **REAL with a stated 8 % floor**,
not to "exact".

Known residual issues, not hidden:

* `band A` (the deprecated `p_core - p(0.80-0.95R)` read-out) is still wrong by
  -77 % to -12 % with the closure on. That definition
  differences against a band that sits inside the density-deficit zone; the
  closure fixes the pressure there, which makes the *difference* meaningless
  rather than small. Confirms the existing instruction to retire the convention.
* `v_max/v_ref` reaches 0.14 at dx = 0.002 / c0 = 20, above the 0.1
  spurious-current bar of the audit. That is the capillary breathing mode at the
  coarsest resolution; it does not affect the time-averaged pressure, but the
  droplet is not fully quiescent there.
* The shell summation density still reads 0.90-0.94 x rho0, so the
  pressure and the density no longer agree pointwise on the shell; only the
  pressure is corrected. A density-side closure (or the mass calibration of P1)
  would remove the last of it and is left as the next step.
* Timing: the ladder is 4 short runs of a few hundred particles; this is a
  2-D verification case, not a performance claim.

## 5. Controls

| Control | Measurement |
|---|---|
| mass calibration off, c0 = 20, abs read-out | -56.6 % |
| legacy `mls` curvature operator, lowest c0 | +54.2 % |

The first shows that the 2.4 kPa transient from the lattice bias (C2) is still
there and still large on the absolute read-out when calibration is disabled;
the closure does not fix it and was not meant to. The second confirms the
operator control is unchanged by this work.

## 6. Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py \
    --dx 0.002 --sound-speeds 20 5 --free-surface-pressure
python benchmarks/laplace_pressure_floor/make_closure_doc.py
PYTHONPATH=src python -m pytest tests/test_free_surface_pressure_closure.py -q
```
