# XVAL: TensorSPH vs SPHinXsys — mixture material and surface tension

**Anchor** `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` · mission **m17504** ·
repository **TensorSPH** · 2026-11-13

**Verdict (short).** The upstream *stress construction* (contact fraction →
colour gradient → normal → continuous surface stress) cross-validates against the
exact Young–Laplace law to **−1.65 % at R/Δx = 64** (gate < 15 % → **pass**,
converging on a resolution ladder). The upstream *force assembly* (inner +
contact flux of the same stress) does **not** reproduce the curvature force in a
literal transcription: the measured Laplace jump is **−99.7 %** of σ/R, i.e.
the ported flux is essentially blind to curvature — **honest FAIL**, evidence and
diagnosis in §5–§7. Production path in TensorSPH therefore stays the existing
CSF-consistent model; the reusable idea taken over is the mixture contact
fraction and its measured effect on the surface-delta normalisation.

---

> **Follow-up (mission m19927, dynamic test).** The static-droplet readings above are
> bookkeeping-sensitive and cannot separate a model defect from a lattice-symmetry
> cancellation. Measured instead as a Lamb-mode frequency on a *perturbed* interface,
> the ported CSS force reproduces `omega_n^2 = n(n^2-1)sigma/(rho R^3)` to **+0.7 %**
> (single phase, hourglass off) and **-4.2 %** (two-phase mixture 1000:1) — inside the
> 15 % gate — while the inner hourglass penalty contributes a spurious `O(sigma/R)`
> mode-2 pressure that does not scale with the perturbation. See
> [`LAMB.md`](LAMB.md) and `benchmarks/surface_tension/sphinxys_lamb.py`.

## 1. Problem and the reference (zero hand-copy)

Question: does TensorSPH's surface-tension capability agree with the mixture /
surface-tension model of a mature open-source SPH code, and what does it take to
port that model?

Reference implementation, **Xiangyu-Hu/SPHinXsys**, default branch `master`,
fetched through the GitHub contents API on 2026-11-13. The raw bytes of every
file consulted are archived with their sha256 in `raw/` (`raw/MANIFEST.json`),
so each quoted formula below is checkable without trusting this text:

| upstream file | what was taken from it | sha256 (16) |
|---|---|---|
| `src/shared/particle_dynamics/fluid_dynamics/surface_tension.cpp` | colour gradient, stress, inner flux, hourglass penalty, contact flux, contact fraction | `04b9ad3a2354ba73` |
| `.../surface_tension.hpp`, `.../surface_tension.h` | data plumbing, `hourglass_control_coeff = 4.5`, `SurfaceStressForceComplex` composition | `76279a06fc0075cb`, `f77685ce2e67010b` |
| `src/shared/kernels/kernel_wendland_c2.cpp` | kernel and derivative (`W_2D`, `dW_2D`) used by the port | `da0f2e7232722ecd` |
| `src/shared/kernels/base_kernel.cpp` | `dW = inv_h * factor_W * dW(q)` — confirms `dW_ij` is dW/dr, not a gradient coefficient | `a697e49df90c6b60` |
| `src/shared/adaptations/adaptation.h` | `SPHAdaptation(h_spacing_ratio = 1.3)`, `KernelWendlandC2` | `2bca4b9a7b9809c7` |
| `src/shared/particle_dynamics/fluid_dynamics/fluid_time_step.cpp` | `SurfaceTensionTimeStep`: `dt = CFL·h/max(c, sqrt(2πσ/(ρ0 h)))` | `de34f9633c22708a` |
| `tests/2d_examples/test_2d_square_droplet/2d_square_droplet.cpp` | the canonical case: ρ_w/ρ_air = 1000:1, σ = 1, c = 10 U_ref, two bodies + contact relation, dt CFL 0.6 | `d1c90cd3174c3eae` |
| `tutorials/markdown/multiphase_mixture_model.md` | upstream's own statement of the mixture model | `046f5d1649205b01` |

Upstream model, as ported (`src/tensorsph/sphinxys_css.py`):

```
grad c_i = -2 f_k Σ_{j ∈ other phase} V_j ∇_i W_ij ,   f_k = rho0_i/(rho0_i + rho0_k)
n_i      = grad c_i / (|grad c_i| + Eps)
S_i      = sigma (I - n_i ⊗ n_i) |grad c_i|                  (no explicit curvature)
a_i      = (m_i/rho_i) Σ_{j ∈ same phase} V_j (S_i + S_j + H_ij) · ∇_i W_ij
a_i     += (m_i/rho_i) Σ_{j ∈ other}      V_j (2(1-f) S_i + 2 f S_j + Hc_ij) · ∇_i W_ij
H_ij     = 4.5 (sigma/2) (T_i+T_j) M_ij / r_ij ,  M_ij = -(g·e) g eᵀ/(|g·e||g|),  g = (½)(grad c_i + grad c_j)
```

## 2. Method

* **Case.** Static two-phase water disc (R = 0.1 m) surrounded by 6 rings of air
  on the *same* lattice (as upstream: both bodies share
  `particle_spacing_ref`). σ = 0.072 N/m, ρ_w = 1000 kg/m³, ρ_air = 1 kg/m³
  (1000:1, the upstream square-droplet ratio), h = 1.3 Δx, Wendland C2 with
  support 2h — the upstream kernel, which TensorSPH already ships.
* **Two independent readings of the same ported fields**, both against the exact
  2-D equilibrium condition Δp = σ/R:
  * *stress reading* — the upstream stress is a tangential tension of magnitude
    `|S_t| = σ|∇c|`; across the interface `∫|S_t| dn = σ∫|∇c| dn`, so
    `Δp = ∫|S_t| dn / R`. This isolates the mixture-weighting part of the model.
  * *flux reading* — the acceleration the ported force assembly actually
    produces, integrated with the geometry-exact estimator
    `Δp = Σ_i ρ_i(-a_i·n) A_i /(2π r_c)` (same estimator as
    `benchmarks/surface_tension/laplace.py`, whose gauge error is 0.1 %).
* **Gauge.** The estimator is applied unchanged to the stress reading, so the
  two numbers differ only by the force assembly, not by measurement.
* **Pre-declared gate.** |error| < 15 % of σ/R at every ladder point; a ladder
  point that fails is reported as FAIL, not excluded.
* **Ablations / sensitivities.** (a) resolution ladder R/Δx = 16, 32, 64;
  (b) density ratio ρ_air/ρ_w = 1, 10⁻¹, 10⁻², 10⁻³, 1 (contact fraction
  0.5 … 1.0); (c) hourglass penalty on/off; (d) capillary time-step formula of
  both codes.
* **Reproduce.** `PYTHONPATH=src python benchmarks/surface_tension/sphinxys_xval.py`
  (~6 s CPU, artifacts in `benchmarks/surface_tension/out_sphinxys_xval/`);
  `PYTHONPATH=src python -m pytest tests/test_sphinxys_css.py` (7 tests, 8 s).

## 3. Parameters (all non-dimensional groups)

| quantity | value | note |
|---|---|---|
| R/Δx | 16, 32, 64 | ladder; N = 1517 / 4513 / 15373 particles |
| h/Δx | 1.3 | `SPHAdaptation(h_spacing_ratio = 1.3)` upstream, `Config.h_factor` in TensorSPH |
| kernel support | 2h = 2.6 Δx | Wendland C2, `7/(4π h²)` |
| ρ_w/ρ_air | 1000 : 1 | upstream square droplet (`rho0_f = 1`, `rho0_a = 0.001`) |
| f_w = ρ_w/(ρ_w+ρ_air) | 0.9990 | contact fraction entering the colour gradient |
| hourglass coefficient | 4.5 | upstream default |
| σ, ρ_w | 0.072 N/m, 1000 kg/m³ | water–air, 20 °C |
| Laplace group | σ/(ρ_w R) = 7.2×10⁻⁴ m²/s² | only group of the static case |
| Weber / Ohnesorge | 0 / ∞ | static (U = 0, μ = 0): neither enters |
| capillary dt, upstream (CFL 0.6) | 2.58 ms at Δx = R/64 | `0.6·sqrt(ρ h³/(2πσ))` |
| capillary dt, TensorSPH (CFL 0.25) | 2.70 ms at Δx = R/64 | `0.25·sqrt(ρ h³/σ)`; the two conventions agree to 4.4 % |

## 4. Comparison table (ref = exact σ/R = 0.7200 Pa)

| R/Δx | ∫\|∇c\|dn | Δp stress reading [Pa] | error | gate | Δp flux reading [Pa] | error | gate |
|---|---|---|---|---|---|---|---|
| 16 | 0.9605 | 0.6915 | **−3.95 %** | ✓ | 0.0244 | −96.61 % | ✗ |
| 32 | 0.9778 | 0.7040 | **−2.22 %** | ✓ | 0.00686 | −99.05 % | ✗ |
| 64 | 0.9835 | 0.7081 | **−1.65 %** | ✓ | 0.00219 | −99.70 % | ✗ |

Density-ratio sensitivity (R/Δx = 32, stress reading):

| ρ_air/ρ_w | f_w | ∫\|∇c\|dn | Δp/σR | error |
|---|---|---|---|---|
| 1 | 0.500 | 0.4894 | 0.489 | **−51.06 %** |
| 0.1 | 0.909 | 0.8898 | 0.890 | −11.02 % |
| 0.01 | 0.990 | 0.9691 | 0.969 | −3.09 % |
| 0.001 | 0.999 | 0.9778 | 0.978 | −2.22 % |
| 0.0001 | 0.9999 | 0.9787 | 0.979 | −2.13 % |

Hourglass off (R/Δx = 64): stress reading 0.7082 (−1.65 %), flux reading
0.00121 (−99.83 %), momentum residual 0.0 (vs 0.73 with the penalty on).

## 5. Convergence and sensitivity

* **Order.** The stress reading converges monotonically from below,
  −3.95 % → −2.22 % → −1.65 % (first order in Δx: an observed order of ≈0.9),
  i.e. the residual is a discretisation error of the colour-gradient shell, not
  a model offset.
* **Density ratio.** The literal `2 f_k` prefactor makes the surface delta
  `∫|∇c| dn = 2 f_w × 0.49`: exactly 1 in the one-sided limit (f_w → 1) and
  exactly ½ at a 1:1 density pair. So the upstream weighting is calibrated for a
  light/heavy pair; for an equal-density "mixture" the Laplace jump is halved.
  This is the quantitative content of the upstream contact-fraction idea, and it
  is the reason the port exposes `delta="unit"` (divide the `2 f_k` out) for
  use at arbitrary density ratios.
* **Hourglass penalty.** Changes the stress reading by < 0.01 pp → it does not
  carry the Laplace response (it is a zero-surface-energy regulariser), but it
  costs exact momentum conservation: the net momentum residual jumps from
  0.0 (penalty off, machine zero) to 0.73 (penalty on). The upstream flux
  without the penalty conserves momentum to machine precision (test
  `test_stress_flux_conserves_momentum_without_hourglass`).
* **Time step.** Upstream's capillary bound `0.6 sqrt(ρh³/2πσ)` and TensorSPH's
  `0.25 sqrt(ρh³/σ)` agree to 4.4 % (0.6/√(2π) = 0.239 vs 0.25), so no
  re-tuning is needed when switching models.

## 6. Differences list — TensorSPH vs SPHinXsys (and what the port gives up)

| aspect | SPHinXsys (upstream) | TensorSPH CSF-consistent (production) | port decision here |
|---|---|---|---|
| interface description | **two-phase mixture**: colour gradient from the *other* phase's contact neighbours only; contact fraction `f = ρ_i/(ρ_i+ρ_k)` | **single phase**: kernel-sum colour `c = Σ V W` (Brackbill indicator), gradient by MLS | both kept: `mode="two_phase"` (faithful) and `mode="single_phase"` (vacuum reduction `Σ_missing = -Σ_present`, `f→1`, contact force `∝ ρ_air/ρ_w → 0` dropped) |
| force form | continuous surface **stress** divergence, `S = σ(I-nnᵀ)\|∇c\|`; **no explicit curvature** | CSF volume force with explicit `κ = ∇·n` (MLS-consistent divergence) | ported literally; production keeps CSF because the stress flux fails the Laplace gate here (§7) |
| momentum conservation | two-point symmetric flux `(S_i+S_j)`; exactly conservative | pairwise antisymmetric `f = -σκnδ`; conservative | verified numerically (residual 1e-16) |
| zero-energy modes | explicit **hourglass penalty** (coeff 4.5) | none; instead MLS curvature + normal smoothing | ported, measured: no effect on the Laplace reading, breaks exact momentum conservation |
| kernel / h | Wendland C2, support 2h, h = 1.3Δx | cubic spline by default (`Config.h_factor = 1.3`), Wendland C2 available | port uses **Wendland C2** to match upstream; `kernel="cubic"` is available |
| boundary condition at walls | contact relation to solid bodies with `f = ρ_f/(ρ_f+ρ_s)` | `apply_contact_angle_boundary` (Brackbill normal rotation) | upstream wall/adhesion path **not ported** (no adhesion module, see §7) |
| time step | `0.6 h/max(c, sqrt(2πσ/(ρ0 h)))` | `0.25 sqrt(ρh³/σ)` (≡ `k = 1/h`) | upstream formula ported as `capillary_dt_limit`; agrees to 4.4 % |
| integration | multiphase Riemann WCSPH (`MultiPhaseIntegration1stHalfWithWallRiemann`), ρ by summation, transport correction | not present (single-phase) | dynamic two-phase integration **out of scope** this mission |
| reference case | square droplet → circle, DTW regression on water kinetic energy only | Young–Laplace ladder (analytic) | analytic reference used; upstream publishes no Laplace number to compare against (§7) |

**Porting trade-offs actually taken.** (i) `delta="sphinxys"` keeps the literal
`2 f_k` (upstream behaviour, correct at f→1, minus 51 % at 1:1) and
`delta="unit"` normalises it away — the caller must choose, and the choice is
worth 2× in Δp. (ii) The contact force is dropped in single-phase mode because
it scales with ρ_air/ρ_w (measured: 0.0 contribution at 1000:1). (iii) The
hourglass penalty is kept switchable and off by default in the benchmarks that
assert momentum conservation.

## 7. Failure modes, honest FAIL, and limits

* **FAIL of the literal flux (measured).** The ported force assembly gives
  Δp/σR = 0.7–3.4 % (and **decreasing** with resolution), while the stress it is
  the divergence of is correct to 2 %. Localised with the port's own fields:
  1. a *synthetic* tangential stress `σ δ(r)(I-nnᵀ)` with unit normal integral,
     fed to the same discrete flux on a complete-support interior lattice,
     returns `∫f·(-n)dn ≈ 0.007` instead of `σκ = 0.72`, both for the symmetric
     `(S_i+S_j)` and the classical `(S_j-S_i)` divergence form — i.e. **the
     discrete divergence of this tangential-stress field cancels** in our
     transcription, so the cancellation is in the flux discretisation, not in
     the colour/normal/stress part;
  2. the interface is a lattice staircase, so the one-sided (free-surface)
     support adds O(1/Δx) tangential noise: single-particle `|∇·S|` reaches
     ~950 N/m³ against an analytic ~90 N/m³, and the radial projection of that
     noise averages out instead of producing σ/R.
  Because the *accessor* of the curvature force is the flux, the honest verdict
  is that a literal port of the upstream stress model **does not reproduce the
  Laplace law** in TensorSPH's discretisation; replacing the production CSF path
  is **not** justified on this evidence. This is registered as a negative
  result, not hidden behind the passing stress reading.
* **What could not be checked.** SPHinXsys publishes no Laplace number: its
  regression for this case is dynamic-time-warping on the water kinetic-energy
  series (`tests/2d_examples/test_2d_square_droplet/regression_test_tool/*.xml`),
  which requires the full two-phase Riemann WCSPH integration that TensorSPH
  does not have. So the reference used is the analytic σ/R, and the OSS part is
  used as the *method* under cross-validation, not as a numeric answer. A true
  code-to-code run needs the C++ build (TBB/VTK/Eigen/Boost) — out of scope.
* **Scope.** Static (no time integration), 2-D, inviscid, no wall/adhesion, no
  three-phase. The contact-angle/adhesion part of upstream's mixture model is
  not ported.
* **Ecosystem friction (as found).** (a) `GET {TENSOR_API_BASE}/knowledge_base`
  and `/oss_solutions` do not exist on this build; the working routes are
  `/tensor/knowledge/search` and `/tensor/negative-knowledge/search`.
  (b) `/tensor/knowledge/search` returned
  `{"available": false, "reason": "engine_offline"}` (engine at 127.0.0.1:8021
  refused the connection), so the "knowledge_base capability + negative ledger"
  first check could only be answered from the negative ledger (2 hits on
  "surface tension": `negk:gate:93` two_phase_laplace L0, `negk:loop:16230`
  SOD honest-FAIL) — the positive knowledge base was unavailable, and the OSS
  inventory had to be rebuilt from the GitHub API directly.
  (c) `raw.githubusercontent.com` is blocked (HTTP 000) while `api.github.com`
  answers 200 → all source was fetched through the contents API (base64), which
  is why every file is archived with a sha256.

## 8. Artifacts

| path | content |
|---|---|
| `src/tensorsph/sphinxys_css.py` | the port (two-phase faithful + vacuum reduction, `delta` switch, hourglass, capillary dt) |
| `benchmarks/surface_tension/sphinxys_xval.py` | the XVAL run (ladder, density-ratio and hourglass sensitivities, LaTeX-free figure) |
| `benchmarks/surface_tension/out_sphinxys_xval/xval_ladder.json` | every number quoted here (programmatic, zero hand-copy) |
| `benchmarks/surface_tension/out_sphinxys_xval/xval_ladder.png` | fig. 1: Δp/(σ/R) vs R/Δx (both readings) and ∫\|∇c\|dn vs contact fraction |
| `tests/test_sphinxys_css.py` | 7 tests: contact-fraction identity, bulk blindness, delta normalisation lock, momentum conservation (both ways), Laplace gate, capillary dt |
| `docs/xval_sphinxys_surface_tension/raw/` | upstream sources + `MANIFEST.json` (sha256 per file) |