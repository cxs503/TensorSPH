# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE, round 3: a well-balanced capillary force

*Anchor:* `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` (TensorSPH), acceptance `< 15 %` on the
Laplace jump. *What this round adds:* the reason the round-3 time loop needed a viscosity
knob to hold a drop still, a formulation that removes it, and the evidence that removing it
does not touch the physics.

## ① Case, geometry, parameters

| quantity | value |
|---|---|
| configuration | 2-D circular drop in a periodic-free carrier block, hydrostatic |
| radius `R` | 20.0 mm |
| carrier padding | 12 mm on every side |
| lattice | triangular (hex), `dx` per level |
| surface tension `sigma` | 0.0720 N/m (water/air) |
| density `rho0` | 1000 kg/m^3 (both phases, single-density carrier) |
| analytic reference | `Delta p = sigma/R` = **3.6000 Pa** |
| velocity scale | `v_ref = sqrt(sigma/(rho R))` = 0.06000 m/s |
| kernel | cubic spline, `h = 1.3 dx` |
| time step | `dt = 0.1 t_sigma`, `t_sigma = sqrt(rho h^3/sigma)` |
| projection solve | adjoint pair `G = -D^T`, gauge-fixed Jacobi-PCG, `rel_tol = 1e-06` |
| viscosity | **none** (`nu = 0`) for every run on this page |

The physics is the Young-Laplace law: a static drop has a pressure jump `sigma/R` between
its interior and the surrounding fluid, and a correct solver must produce that jump *with
the interface at rest*.

## ② Method: what was changed, and why

Rounds 1-2 (`xval_projection_laplace_oss.md`, `xval_projection_laplace_robustness.md`)
measured the jump statically and mapped its knob sensitivity.  The dynamic round
(`verification_laplace_sph_isph_dynamic.md`) made the time loop stable by pairing the
divergence and gradient operators as exact adjoints, but the drop never came to rest:
`max|u|/v_ref = 0.179` against Adami et al. (2012)'s `0.1`
admissibility line (that round's own gate reports *pass = False*), and it was
held at bay by a viscosity knob `nu_f` that is a damping device, not a modelled property.

The module header of `src/tensorsph/projection_adjoint.py` already named the cause: *"even a
consistent projection cannot remove the solenoidal part of the lattice CSF force"*.  A
projection removes only what its own gradient can represent, so a rotational part of the
capillary acceleration is re-injected every step and only viscosity can bleed it away.

Round 3 turns that statement into a construction.  For a circular interface the continuum
CSF acceleration is a pure gradient (`curl(sigma kappa grad c) = (sigma/rho) grad kappa x
grad c = 0` because `kappa` and `c` both vary radially), so on the lattice the solenoidal
part is a discretisation residue, not physics.  `helmholtz_split` separates the two parts
with *the projection's own operator*:

```
q  solves   G^T G q = G^T f      with G = -D^T   =>   (D D^T) q = -D f
f  =  G q  +  f_sol              (L2-orthogonal split, CG-limited)
```

so `G q` is exactly annihilated by the projection and `f_sol` becomes a *measured number*
(`solenoidal_fraction`) instead of an invisible momentum source.  `relax_projection_balanced`
then marches with `G q` only.

Two independent instruments come out of this, and they must agree:

* `dp` from the pressure solve (as in rounds 1-3);
* `dp = rho [q]` from the potential alone -- **it never touches the pressure field**, yet
  after the projection the pressure *is* `rho q`, so agreement is a statement about the
  formulation, not a tautology about one array.

## ③ Results

### Resolution ladder (well-balanced, inviscid)

| R/dx | particles | max\|u\|/v_ref | mean\|u\|/v_ref | dp / Pa | dp err | rho[q] / Pa | rho[q] err | area ratio | solen. frac. | steps | wall / s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1203 | 3.481e-07 | 6.462e-08 | 3.6010 | +0.029 % | 3.6010 | +0.029 % | 1.000000 | 0.203 | 50 | 284 |
| 13 | 2150 | 1.422e-07 | 4.435e-08 | 3.5871 | -0.358 % | 3.5871 | -0.358 % | 1.000000 | 0.251 | 50 | 392 |
| 20 | 4773 | 3.642e-07 | 7.192e-08 | 3.5547 | -1.257 % | 3.5547 | -1.257 % | 1.000000 | 0.382 | 50 | 898 |

### Controls (same loop, one knob moved)

| control | max|u|/v_ref | dp err | area ratio | diverged |
|---|---|---|---|---|
| no split, inviscid (control) | 1.183e-01 | -0.165 % | 1.000000 | no |
| dt_factor=0.05 | 1.506e-07 | +0.029 % | 1.000000 | no |
| dt_factor=0.20 | 6.052e-07 | +0.029 % | 1.000000 | no |

### Irrotationality defect vs kernel support (`R/dx = 20`)

| h/dx | solenoidal fraction | rho[q] / Pa | vs sigma/R | |<Gq,f_sol>| | reconstruction |
|---|---|---|---|---|---|
| 1.3 | 0.3820 | 3.5547 | -1.257 % | 1e-10 | 2e-17 |
| 1.5 | 0.2963 | 3.6225 | +0.625 % | 8e-09 | 8e-18 |
| 1.7 | 0.2333 | 3.6182 | +0.507 % | 1e-09 | 4e-18 |
| 1.9 | 0.1873 | 3.5428 | -1.590 % | 2e-08 | 3e-18 |

### Round-3 comparison, same quantity, same gate

| route | viscosity | max\|u\|/v_ref | Adami 0.1 gate | dp err | area ratio |
|---|---|---|---|---|---|
| round 3, damped march (`results_dynamic.json`) | `nu_f = 0.01` | 0.179 | **fail** | +2.018 % | 1.00431 |
| round 3, inviscid control | 0 | see file (loses the lattice) | fail | — | — |
| **round 3, well-balanced (this page)** | **0** | **3.64e-07** | **pass** | +1.257 % | 1.000000 |

## ④ Figure

![fields and series](benchmarks/laplace_sph/results_balanced/balanced.png)

(a) `max|u|/v_ref` vs `t/t_sigma`: the round-3 damped run climbs to and past the Adami line,
the well-balanced runs sit six decades below it; (b) `dp` error from both instruments against
the 3 % claim line and the 15 % acceptance band; (c) the solenoidal fraction of the CSF force
falls as the kernel support grows -- the signature of a smoothing artefact; (d) the final
velocity field (log scale) shows no recirculation cells at the interface; (e) the pressure
field; (f) the radial pressure profile against the analytic step `sigma/R` at `R`.

## ⑤ Notes, limits, and what this does **not** say

* **Scope is the equilibrium branch.** The claim is: a static drop is a fixed point of this
  discretisation to solver tolerance, and the jump it produces is the Laplace jump.  Neither
  capillary waves nor droplet oscillation are covered, and the single-density carrier still
  forbids interface advection with a real density jump.
* **The split is a formulation choice and it changes the discrete force.** What makes it
  admissible is measured, not asserted: the discarded part moves `dp` by less than the run-to-run
  resolution spread (see the `rho[q]` column, which is computed *without* the discarded part
  in the pressure at all), and it shrinks with kernel support. Across the `h/dx` sweep the discarded fraction varies by a factor 2.0 (0.382 -> 0.187) while `rho[q]` stays within 1.59 % of `sigma/R` with no monotone trend: the part the projection cannot see does not carry the jump.
* **The resolution trend is not a clean convergence.** The jump error along the ladder
  (0.03 %, -0.36 %, -1.26 %) drifts to `-1.3 %` at the finest level instead of decaying; it stays inside the
  3 % claim line, but a formal order of accuracy for `dp` should not be claimed from this
  ladder (the readout band and the `h/dx` effect of round 2 act at the same magnitude).
* **The operator pair is not gauge-free.** `G` applied to a constant does not vanish on this
  lattice (measured in `tests/test_projection_balanced.py`), so the potential is only defined
  up to the pinned value; the Laplace jump is a *difference* across the interface, hence immune.
* **Cost is roughly doubled**: one extra CG solve per step (the potential) on top of the
  pressure solve, with the same operator and preconditioner.
* **Still not a full two-phase solver**, still 2-D here, still no ghost-particle boundary --
  the carrier layer is velocity-pinned, a boundary condition rather than physics.

## ⑥ Error judgement (required)

Key quantity: the Laplace jump against the analytic `sigma/R` = 3.6000 Pa.

| gate | threshold | measured | verdict |
|---|---|---|---|
| jump, worst level | `< 15 %` (owner acceptance) | +1.257 % | PASS |
| jump, worst level | `< 3 %` (claim level) | +1.257 % | PASS |
| jump from the potential `rho[q]` | `< 3 %` | +1.257 % | PASS |
| spurious currents | `< 0.1` (Adami et al. 2012) | 3.64e-07 | PASS |
| area drift | `< 2 %` | 0.0000 % | PASS |

**Is every key quantity within 3 %?**  Yes for the jump on both
instruments at every level run.  The spurious-current gate is met by construction of the
formulation (the projected force has no rotational part left to inject), so it is reported as
an inequality, not as a fitted number, and the control run shows what it is without the split.

## ⑦ Reproduce

```bash
PYTHONPATH=src python benchmarks/laplace_sph/run_balanced.py --level 1   # sharded: level 2, 3
PYTHONPATH=src python benchmarks/laplace_sph/run_balanced.py --merge --figure
PYTHONPATH=src python benchmarks/laplace_sph/make_doc_balanced.py
PYTHONPATH=src python -m pytest tests/test_projection_balanced.py -q
```

| artefact | role |
|---|---|
| `src/tensorsph/projection_adjoint.py` | `helmholtz_split`, `irrotationality_defect`, `relax_projection_balanced` |
| `benchmarks/laplace_sph/run_balanced.py` | sharded harness (JSON + figure) |
| `benchmarks/laplace_sph/results_balanced/results_balanced.json` | every number on this page |
| `tests/test_projection_balanced.py` | split exactness, physics gate, Adami gate, anti-drift |
