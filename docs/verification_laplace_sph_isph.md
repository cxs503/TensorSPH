# CASE-LAPLACE-SPH — Young-Laplace jump `Δp = σ/R` on a projection-SPH drop

**Verdict: the Laplace gate is met at every resolution; the spurious-current
gate is NOT met and is reported as an honest FAIL.**

| gate | measured | limit | result |
|---|---|---|---|
| `laplace_error_finest` | 0.0021 | 0.0500 | PASS |
| `laplace_error_all_levels` | 0.0097 | 0.0500 | PASS |
| `gauge_self_test` | 0.0029 | 0.0300 | PASS |
| `cross_check_flux` | 0.0060 | 0.0300 | PASS |
| `bulk_balance` | 0.0058 | 0.0500 | PASS |
| `negative_control_sigma0` | 0.0000 | 1.00e-10 | PASS |
| `spurious_currents_adami_0p1` | nan | 0.1000 | FAIL |

---

## 1. Problem setting and reference

A circular interface in mechanical equilibrium carries the jump

```
Δp = σ (1/R1 + 1/R2) = σ/R        (circle: one curvature radius)
```

Reference value: **Δp = σ/R = 3.6000 Pa** for σ = 0.072 N/m and R = 0.02 m, ρ0 = 1000.0 kg/m³.

Sources: Brackbill, Kothe & Zemach (1992), *J. Comput. Phys.* **100**:335
(CSF formulation and the `δs = |∇c|` surface delta function); Cummins & Rudman
(1999), *J. Comput. Phys.* **152**:584 (projection/incompressible SPH).

Dimensionless parameters of the case:

| parameter | value |
|---|---|
| R/dx (ladder) | 10.0, 13.3, 20.0 |
| R/h (ladder) | 7.69, 10.26, 15.38 |
| Bond number | 0.0 (no gravity) |
| Weber number | 0.0 (statics: no inertia) |
| Δp / bulk modulus of the WCSPH EOS (c0 = 20 m/s) | 6.300e-05 |

## 2. Method — why the pressure must come from a projection, not from an EOS

The shipped droplet is weakly compressible: `p = B((ρ/ρ0)^γ − 1)`, `B = ρ0 c0²/γ`.
For this case that route is closed *by arithmetic*: carrying `σ/R = 3.6 Pa`
through a bulk modulus `B = ρ0 c0²/γ = 57142.9 Pa` needs a
density ratio of `Δp/B = 6.30e-05`, while a
WCSPH density field carries ~1e-3 relative noise — the signal is more than a
decade under the noise even at this small radius (for the R = 0.1 m geometry
of `benchmarks/laplace_droplet/` the same audit gives 1.8e-6). No kernel or
curvature fix moves that, which is exactly what the previous rounds measured:
relaxed WCSPH with the consistent CSF gives -78.6 %
and the best admissible free-surface closure variant -20.1 %.

So the pressure is taken from a Lagrange multiplier instead (ISPH):

```
u* = u + dt a_CSF                        (CSF traction only)
−∇²p = −ρ ∇·a           ,  p = 0 on the outer carrier layer
u  = u* − dt ∇p/ρ
```

The jump then lives in the Poisson *source*, not in a 1e-5 density
difference, and the question becomes the honest one: does the discrete CSF
traction integrate to σ/R across the interface?

Discretisation (all operators already shipped in this repository):

| ingredient | operator | code |
|---|---|---|
| colour function | Shepard-smoothed phase indicator, `∫|∇c| dn = 1` | `laplace_isph.smooth_indicator` |
| gradient | first-order consistent MLS | `csf_consistent.mls_gradient` |
| normal | `n = −∇c/|∇c|`, Shepard-smoothed twice | `csf_consistent.shepard_smooth` |
| curvature (default) | `κ = ∇·n` from the smoothed normal | `csf_consistent.mls_divergence` |
| curvature (alternative) | local quadratic surface fit | `csf_curvature.fit_curvature` |
| surface delta | `δs = |∇c|` (Brackbill) | `csf_consistent` |
| Poisson operator | Cummins & Rudman pair Laplacian, matrix-free CG | `laplace_isph.pressure_poisson` |

The vapour phase is filled with particles of the *same* density. That is a
static-verification device, not a multiphase model: `Δp = σ/R` contains σ and
R only, and filling the void is what completes the colour transition `1 → 0`
(a one-sided carrier keeps only ~75 % of the delta normalisation,
`docs/verification_effective_sigma.md`). Here the measured normal integral of
`δs` is 0.9932 on the finest level.

## 3. Parameters

| parameter | value |
|---|---|
| `radius_m` | 0.02 |
| `sigma_N_per_m` | 0.072 |
| `rho0` | 1000.0 |
| `outer_carrier_m` | 0.012 |
| `h_factor` | 1.3 |
| `curvature_default` | mls |
| `tolerance` | 0.05 |

Particle counts on the ladder:

| R/dx | N |
|---|---|
| 10.0 | 1203 |
| 13.3 | 2150 |
| 20.0 | 4773 |

## 4. Results — reference vs measured

### 4.1 Resolution ladder (`κ = ∇·n` of the smoothed normal)

| R/dx | R/h | N | Δp measured [Pa] | Δp theory [Pa] | ratio | error |
|---|---|---|---|---|---|---|
| 10.0 | 7.69 | 1203 | 3.6350 | 3.6000 | 1.0097 | +0.97 % |
| 13.3 | 10.26 | 2150 | 3.6218 | 3.6000 | 1.0061 | +0.61 % |
| 20.0 | 15.38 | 4773 | 3.5923 | 3.6000 | 0.9979 | -0.21 % |

### 4.2 Ladder with the quadratic surface fit

| R/dx | κ error | Δp ratio | Δp error |
|---|---|---|---|
| 10.0 | -28.73 % | 0.7699 | -23.01 % |
| 13.3 | -28.37 % | 0.7631 | -23.69 % |
| 20.0 | -29.84 % | 0.7554 | -24.46 % |

### 4.3 Ablation — where the residual error sits

| variant | κ error (finest) | δs normal integral | Δp ratio | Δp error |
|---|---|---|---|---|
| `model` | +0.27 % | 0.9932 | 0.9979 | -0.21 % |
| `kappa_exact` | +0.00 % | 0.9932 | 0.9948 | -0.52 % |
| `delta_unit` | +0.27 % | 0.9932 | 1.0047 | +0.47 % |
| `gauge` | +0.00 % | 0.9932 | 1.0029 | +0.29 % |

`gauge` replaces *both* operators by their exact analytic counterparts (κ = 1/R,
unit-normalisation delta). It is the instrument error bound: whatever it returns
is how well this harness can measure, so the model error is judged against it.

### 4.4 Independent cross-check

| R/dx | Δp from the pressure field | Δp from `−∫ρ a_r dr` | agreement |
|---|---|---|---|
| 10.0 | 3.6350 Pa | 3.5826 Pa | -1.44 % |
| 13.3 | 3.6218 Pa | 3.5853 Pa | -1.01 % |
| 20.0 | 3.5923 Pa | 3.5785 Pa | -0.38 % |

The flux route never touches the Poisson solve, so agreement between the two
is what makes the number trustworthy rather than a solver artefact.

### 4.5 Negative control (`σ = 0`)

With σ = 0 the same pipeline returns a jump of **0.0000 Pa**
(the gate is 1e-10), i.e. the harness does not manufacture a jump.

## 5. Convergence and sensitivity

Observed order over the finest pair: **2.58**.
The errors at the two finest levels are at the instrument floor (§4.3), so the
fitted order is not a clean truncation order; it only shows the error is not
growing with refinement, which is the defect the earlier rounds exposed.

| knob | value | Δp ratio | κ error |
|---|---|---|---|
| vapour band | r/R ∈ [1.15, 1.35] | 0.9980 | — |
| vapour band | r/R ∈ [1.25, 1.45] | 0.9979 | — |
| vapour band | r/R ∈ [1.35, 1.55] | 0.9976 | — |
| h/dx | 1.2 | 0.9761 | +0.22 % |
| h/dx | 1.3 | 0.9979 | +0.27 % |
| h/dx | 1.4 | 1.0202 | +0.56 % |
| fit stencil | 2.0 h | 0.3043 | -107.59 % |
| fit stencil | 3.0 h | 0.6004 | -53.00 % |
| fit stencil | 4.0 h | 0.7554 | -29.84 % |
| fit stencil | 5.0 h | 0.8403 | -18.40 % |

Two things to read off this table:

1. The measurement is robust to the vapour-band choice (spread ≲ 1 %), so the
   number is not an artefact of where the reference pressure is sampled.
2. The quadratic surface fit is *worse* than `∇·n` on this carrier, and it
   only converges as its stencil is widened. That is the opposite of the
   result recorded for the one-sided carrier in
   `docs/verification_csf_laplace_consistent.md` (+25 % for `∇·n`), which is
   consistent once the cause is named: on a one-sided surface `∇·n` loses the
   outer half of the shell, while on a complete carrier with a twice-smoothed
   normal it is the better operator. The fit's lead there does not transfer.

### 5.1 Spurious currents — honest FAIL

The projection time integration of §2 diverges after **13 steps** (dt = 0.001562 s, CFL on the capillary time):

| step | t/tσ | max|u|/√(σ/ρR) | mean|u|/v_ref | Δp [Pa] | area ratio |
|---|---|---|---|---|---|
| 5 | 0.50 | 0.889 | 0.182 | -114.708 | 1.0166 |
| 10 | 1.00 | 31.446 | 11.454 | -7323.106 | 1.4227 |

The Adami et al. (2012) admissibility limit is `max|u|/v_ref < 0.1`; the
integration exceeds it within half a capillary time. **Cause, measured not
guessed:** the projection removes only the *divergent* part of the CSF
acceleration, and the residual of the discrete momentum balance
(median 0.362 of the mean
CSF acceleration on the interface shell, but only
0.0058 in the bulk) is
largely solenoidal. The bulk number is the reason the *static* result stands:
away from the interface shell the discrete pressure cancels the traction to
better than 1 %, so the pressure field really is in equilibrium; what fails is
the time marching of a force field whose lattice curl is not resolved. Cures
worth trying next are an adjoint-consistent divergence/gradient pair with
ghost-particle free-surface boundaries (a pair-based gradient was tried here
and did not stabilise it), or a semi-implicit capillary treatment.

## 6. Failure modes, limits, artefacts

**Reproduce**

```bash
PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure
PYTHONPATH=src python benchmarks/laplace_sph/make_doc.py
PYTHONPATH=src python -m pytest tests/test_laplace_sph_isph.py -q
```

**Artefacts**

| path | role |
|---|---|
| `src/tensorsph/laplace_isph.py` | projection-SPH Laplace measurement |
| `benchmarks/laplace_sph/run.py` | harness (writes `results.json`, figure) |
| `benchmarks/laplace_sph/results.json` | all numbers of this page |
| `benchmarks/laplace_sph/laplace_sph_isph.png` | pressure field, radial profile, ladder |
| `docs/verification_laplace_sph_isph.md` | this page (rendered) |

**Limits**

* 2-D, static, zero Bond number, uniform density. Capillary waves, droplet
  oscillation and any dynamic Laplace problem are *not* covered.
* The vapour carrier has the liquid density: this is a colour-function
  completion device for a static verification, **not** a two-phase flow model.
  Dynamic two-phase flow (density/viscosity jump, interface advection) remains
  unimplemented.
* The 5 % gate is met by the pressure-field route; the weakly compressible
  (Tait) route still cannot carry the jump (resolvability argument, §2) and
  its verdict stays WITHHELD. Nothing here retro-fits the WCSPH model.
* The spurious-current gate fails; the time integration is not a solver.

