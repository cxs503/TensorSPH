# Method selection — mixture material & surface tension (round 2, m18642)

Round 2 of the two-round method (`owner 2026-11-11`): survey mature methods first, then combine.
Round 1 (m17909 / m18603) went straight to a faithful port and honest-FAILed on the force reading.
This note is the survey that was missing, plus the selection actually implemented.

## 1. OSS horizontal survey (`~/oss-cache` / archived sources)

`oss_inventory` / `oss_solutions` are **not reachable from this harness**: the platform API exposes
no `oss_inventory`/`oss_solutions` route (`/openapi.json` scan for `solution`/`inventory` returns
`[]`), the local `*.db` files are 0-byte placeholders, and `~/oss-cache` on the control host only
holds three LBM repos (`Maarten-vd-Sande_lbm`, `hschwane`, `pylbm`) — none matches
`%TENSION%`. So the survey was done against the source that actually matters for this anchor and
is available byte-exactly in this repo:

| upstream file (SPHinXsys, master) | what was learned | sha256 (16) |
|---|---|---|
| `…/fluid_dynamics/surface_tension.cpp` | colour gradient from the *other* phase only, `f_k = ρ0_i/(ρ0_i+ρ0_k)`; stress `S = σ(I−nnᵀ)|∇c|`; inner + contact flux; hourglass penalty 4.5 | `04b9ad3a2354ba73` |
| `…/fluid_dynamics/surface_tension.hpp` | data plumbing: `ColorGradient`, `NormDirection`, `SurfaceTensionStress` per particle | `76279a06fc0075cb` |
| `…/kernels/kernel_wendland_c2.cpp` | Wendland C₂, support 2h | `da0f2e7232722ecd` |
| `…/adaptations/adaptation.h` | h/Δx = 1.3 | `2bca4b9a7b9809c7` |
| `…/fluid_time_step.cpp` | capillary dt `0.6·√(ρh³/2πσ)` | `de34f9633c22708a` |
| `tests/2d_examples/test_2d_square_droplet` | the only shipped case; validated by DTW against its own kinetic-energy curve, **no Laplace verification** | `d1c90cd3174c3eae` |

Raw bytes + `MANIFEST.json`: `docs/xval_sphinxys_surface_tension/raw/`.

**Consequence of the survey.** Upstream publishes no exact-anchor verification of this model, so
"faithful port" is not the same as "verified model": the port must be *anchored* independently.
That is exactly the gap round 1 fell into.

## 2. Method genealogy (what exists, what it is good for)

| family | formulation | strength | weakness | used here |
|---|---|---|---|---|
| **CSF** (Brackbill, Kothe & Zemach 1992, *J. Comput. Phys.* 100, 335) | `f = σκδn`, curvature explicit | cheap, explicit curvature | κ from a second derivative of a noisy colour field | no (but it is what the "stress reading" measures) |
| **SPH colour function** (Morris 2000, *Int. J. Numer. Methods Fluids* 33, 333) | kernel-sum colour `c_i = ΣV_j c_j W_ij`, `∇c` from the same sum | smooth delta, `∫|∇c|dn → 1` | needs a complete support; one-sided at a free surface | partially (the vacuum-reduction identity is the same trick) |
| **CSS / continuous surface stress** (this anchor; SPHinXsys) | `S = σ(I−nnᵀ)δ`, force = `∇·S` | momentum-conserving pair flux, no explicit κ | the discrete divergence of a *second-rank* tensor is fragile (see §4) | **selected** |
| **SPH stress divergence, symmetric form** (Monaghan 1992, *Annu. Rev. Astron. Astrophys.* 30, 543) | `a_i = Σ_j m_j (A_i/ρ_i² + A_j/ρ_j²)·∇W_ij` | provably momentum-conserving, standard | needs the bracket to be the symmetric two-point form of *one* field | **selected** |
| **Reproducing divergence** (Adami, Hu & Adams 2010, *J. Comput. Phys.* 229, 5011) | first-order-consistent (MLS/CSPM) divergence, `M_i = ΣV_j (x_j−x_i)⊗∇W_ij` | removes lattice bias, works at free surfaces | matrix inversion per particle; `M_i` singular on a truncated support (measured here: 1 484 of 1 517 edge particles at R/Δx = 16) | probed, rejected for this case (the operator floor is already 2.6 %) |

Bibliographic identifiers are from standard references; they were not re-verified against the
publisher PDFs in this run, and no gate number depends on them.

## 3. Selection (what round 2 implements)

1. **Keep** the mixture contact fraction `f_k = ρ0_i/(ρ0_i+ρ0_k)` and the one-sided colour gradient
   — it is the anchor's subject, and its quantitative effect is measured (density-ratio sweep).
2. **Keep** the CSS formulation — no explicit curvature is needed, the momentum-conserving pair
   structure survives.
3. **Replace** the pair weight `(m_i/ρ_i)V_j` by `V_j/ρ_i` (dimensional consistency with
   `a = ∇·S/ρ`). This is the defect: the force density was ∝ `m_i = ρ_i Δx²` and decayed with the
   grid, which is exactly the round-1 failure signature.
4. **Replace** the cross-phase bracket `2(1−f)S_i + 2fS_j` by the symmetric `S_i + S_j` for the
   *force*, keeping the mixture weighting in the *colour gradient*. Rationale measured, not
   asserted: the mixture bracket makes the light phase push outward (tension sign reversed on the
   air side) and costs 25 pp of the Laplace jump.
5. **Turn the hourglass penalty off** for the gate: it breaks momentum conservation by construction
   and buys nothing (0.01 pp). Keep it as a documented switch.
6. **Measure over both phases** when checking Laplace: the interface shell is two-sided, and the
   one-sided mixture delta puts ~30 % of its mass on the light side.

## 4. Reusable lessons (negative knowledge, for the ledger)

- **N1.** For a second-rank tensor, `Σ_j V_j (S_j − S_i)·∇W_ij` has an exact zero mode on an
  isotropic particle neighbourhood: measured reading 0.0000 (1e−17) on an analytic unit-integral
  stress bump at R/Δx = 16, 32, 64. Only the symmetric two-point bracket `(S_i + S_j)` carries the
  divergence. Do not "simplify" to the difference form.
- **N2.** A pair weight that carries the particle mass makes the surface-force density scale as
  `Δx²`. The symptom is a XVAL error that *grows* with refinement while the field quantities
  converge. Check for it before blaming the discretisation.
- **N3.** A dynamic self-reference test (DTW of one's own kinetic energy) cannot catch a pure scale
  error in the force; only an exact static anchor can.
- **N4.** Water-only normal integrals over a one-sided surface delta are biased by ~30 % — an
  estimator artefact that looks exactly like a model failure.
- **N5.** The upstream hourglass penalty (∝ 1/r) is resolution-divergent without a second phase
  (−0.89 % → −375 % → +215 % on R/Δx = 16/32/64): never enable it on a free-surface-only particle
  set.

## 5. Migration plan (what a downstream user does)

- TensorSPH: use `tensorsph.sphinxys_css.surface_tension_acceleration(..., assembly="consistent",
  bracket="symmetric", hourglass=False)`.
- Any WCSPH multiphase run: apply the same two substitutions wherever a CSS surface stress is
  discretised; nothing else in the solver changes (the force is still a symmetric pair flux, so
  momentum conservation is preserved).
- Do **not** adopt the vacuum (single-phase) reduction of this model as-is: it is stable but −30 %
  on Laplace with the assembly fixed (documented, needs the both-phase shell or a renormalised
  delta).