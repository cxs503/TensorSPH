# Audit: the SPHinXsys CSS surface-tension *flux* reading

**Anchor** `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` · **Repository** TensorSPH
· **Task** m18526 · **Domain** ship hydrodynamics / numerical methods (SPH surface tension)

**Reviewed artifacts** (all read-only, no upstream edits):

| artifact | identity |
|---|---|
| `src/tensorsph/sphinxys_css.py` (ported CSS model) | branch `main`, merged by `bf741db` (Merge #663, m17909) |
| `benchmarks/surface_tension/sphinxys_xval.py` (the cross-validation under review) | same commit |
| `docs/xval_sphinxys_surface_tension/README.md` + `dev_report_sphinxys_mixture_surface_tension.md` | same commit |
| `docs/xval_sphinxys_surface_tension/raw/src__shared__particle_dynamics__fluid_dynamics__surface_tension.{cpp,hpp}` | upstream SPHinXsys sources, sha256 in `raw/MANIFEST.json` |

**Independence limitation (declared).** I am a different seat and did not write the
reviewed code, but I reuse the same physical scene builder (`make_two_phase`) and the
same estimator geometry as the reviewed benchmark; the new controls (O1, O3, O5) are
independent of it, but the *reproduction* of the reported number is not an independent
implementation of the scene.

---

## 1. Case and parameters

Identical to the reviewed case: 2-D water disc (`R = 0.1 m`, `σ = 0.072 N/m`,
`ρ_w/ρ_air = 1000:1`), square lattice at spacing `Δx`, Wendland C2 kernel with
`h/Δx = 1.3` (upstream `SPHAdaptation` default), hourglass penalty **off** so that
only the stress flux is being judged. Reference: the exact Young–Laplace law for a
2-D circular interface,

```
Δp = σ / R = 0.7200 Pa
```

Resolution ladder `R/Δx = 16, 32, 64`.

## 2. What is being audited

The reviewed report claims the *stress* reading of the ported colour-gradient field
reproduces `σ/R` to < 3 %, while the *flux* reading of the ported force assembly
(inner + contact stress flux) reproduces only ~1 % of it, and concludes:

> "**Flux reading**: essentially zero force, worsening with refinement. […] This is a
> fundamental property of the CSS discretisation on a particle lattice, not a bug in
> the port."

That conclusion is a *mechanism* claim plus a *magnitude* claim. Both are testable.

## 3. Controls (all numbers programmatic, `out_css_flux_audit/audit.json`)

### O1 — operator calibration (no surface tension involved)

Apply the very operator the port uses,
`(div S)_i = Σ_j V_j (S_i + S_j) · ∇_i W_ij`, to a field with a *known* divergence,
`S = x² I` (exact `div S = 2x`), on a plain lattice:

| probe | pair form | exact | error |
|---|---|---|---|
| x = 2.0 | 3.8957 | 4.0000 | −2.61 % |
| x = 5.0 | 9.7392 | 10.0000 | −2.61 % |

**The conservative pair form is a correct divergence operator on the lattice** (same
−2.6 % truncation bias at both probes). "The discrete divergence cancels on a
lattice" is not true as a general statement about the operator.

### O2 — reproduction of the reviewed number

| R/Δx | Δp reported [Pa] | vs σ/R |
|---|---|---|
| 16 | 0.02053 | **−97.15 %** |
| 32 | 0.00491 | **−99.32 %** |
| 64 | 0.00121 | **−99.83 %** |

Reproduced (the reviewed report quotes −96.61 / −99.05 / −99.70 % for the same
ladder; the small difference is the single-phase vs two-phase reading).

### O3 — where the deficit actually comes from

Two facts fix the attribution:

1. **Bookkeeping.** The ported acceleration carries the upstream prefactor
   `mass_[i] / rho_[i] = V_i`
   (`surface_tension.cpp`: `summation += mass_[i] * dW_ij * Vol_[j] * (...) * e_ij`,
   then `/ rho_[i]`). The force density is therefore `∇·S = a_i / V_i`, while the
   benchmark integrates `ρ0 a_i A_i` — an extra factor of the particle mass
   `m_i = ρ0 V_i = ρ0 Δx²`.
2. **Prediction vs measurement.** If the deficit were that factor, then
   `Δp_ported / (m_i σ/R)` must be constant in `Δx`. It is:

| R/Δx | `m_i = ρ0 Δx²` | `Δp_ported / (m_i σ/R)` | normalised Δp / (σ/R) |
|---|---|---|---|
| 16 | 0.03906 | **0.7298** | 0.730 (−27.0 %) |
| 32 | 0.00977 | **0.6977** | 0.698 (−30.2 %) |
| 64 | 0.00244 | **0.6884** | 0.689 (−31.2 %) |

The particle mass spans a factor 16 across the ladder; the ratio stays within 6 %.
A *physical* cancellation (the reviewed explanation) cannot reproduce `m_i` to two
digits. **The headline "−99 %" is a normalisation artefact of order `m_i`.**

### O4 — per-particle comparison

Over the interface band (`|∇c| Δx > 0.05`) the ported inward acceleration is
`0.0363 / 0.0206 / 0.0105` of the exact curvature force `σ|∇c|/(ρ r)` at
`R/Δx = 16 / 32 / 64`, i.e. it tracks `m_i = 0.0391 / 0.0098 / 0.0024` within a
factor 0.9 / 2.1 / 4.3 — the same story at the particle level, with the residual
half of the story living in the *shape* of the band, not in its total.

### O5 — the idealised CSS field

Feed the exact analytic CSS stress ansatz
`S = σ f(r) (I − r̂ r̂)` with the **exact radial normal** and an exactly normalised
one-sided radial delta `∫ f dr = 1`, through the *same* pair divergence and the same
estimator. Swept over interface width (`1, 2, 3 Δx`) and lattice shifts
(`(0,0), (0.5,0), (0.5,0.3), (0.25,0.5) Δx`) to exclude a lattice-symmetry artefact:

| R/Δx | worst `|Δp| / (σ/R)` over 12 combinations |
|---|---|---|
| 16 | 3.7e−05 |
| 32 | 8.9e−06 |
| 64 | 2.2e−06 |

**The idealised tangential stress field produces essentially no force through the
conservative pair form**, at every width, shift and resolution. The curvature term
`−σ |∇c| r̂ / r` is not recovered. Since O1 proves the operator is sound and O3 proves
the reported deficit is bookkeeping, this identifies the real mechanism: on a
mirror-symmetric neighbour set the leading contribution of a purely tangential,
radially-normalised stress field cancels, and the force that the port *does* produce
(0.69–0.73 `σ/R`) is generated by the deviation of the *computed* normals from
radial, i.e. by a lattice-induced component of `n̂` rather than by the CSS curvature
term it stands for.

## 4. Error judgement (explicit)

| reading | measured | gate | verdict |
|---|---|---|---|
| stress reading `σ∫|∇c|dn/R` (ported field) | −1.65 % at R/Δx = 64 (reviewed), δ-integral 0.960/0.978/0.984 (this audit) | < 15 % | **PASS** |
| flux reading, as reported by m17909 | −97.2 / −99.3 / −99.8 % | < 15 % | FAIL — **but the number is not the physical deficit** |
| flux reading, normalised by `m_i` (this audit) | −27.0 / −30.2 / −31.2 % | < 15 % | **FAIL (honest)** |
| operator calibration O1 | −2.61 % | < 5 % | PASS |
| idealised CSS field flux O5 | ≤ 3.7e−05 | — | FAIL (the curvature term is not produced) |

**Answer to "is the key quantity within 3 %?"** For the stress reading, yes
(< 2 %). For the force actually assembled by the port, **no**: after removing the
normalisation artefact it is 27–31 % low and does not improve with refinement, so the
CSS force cannot be certified for quantitative capillary dynamics at these
resolutions.

## 5. Findings, severity-ranked

1. **[blocking, evidence integrity]** The reviewed conclusion "flux reading ≈ 0, a
   fundamental CSS lattice property, not a bug" is **falsified as stated**. Its
   headline number is dominated by the particle-mass factor `m_i = ρ0Δx²`
   (`Δp` ratios 0.730 / 0.698 / 0.688 while `m_i` changes 16×). A normalised reading
   differs by 27–31 %, not 99 %. The report overstates the deficit by ~3× in the
   exponent and attributes it to the wrong mechanism.
2. **[major, still open]** The normalised flux reading is 27–31 % low and drifts the
   wrong way with refinement (0.730 → 0.688). It **does not** meet the owner's 15 %
   gate. The anchor therefore remains only partially verified: the model *field* is
   verified, the model *force* is not.
3. **[major, mechanism]** O5: the idealised CSS tangential field yields ≤ 4e−05 of
   `σ/R` through the ported operator for every width/shift/resolution tested. So the
   ported curvature force is not the CSS curvature term; the ~70 % it does deliver
   comes from the non-radial content of the computed normals. Any use of the port for
   capillary dynamics inherits this coupling to lattice anisotropy.
4. **[minor, bookkeeping]** `surface_tension_acceleration` returns `V_i·∇·S` (upstream
   prefactor), not `∇·S/ρ`. Whether that is correct depends on how
   `ForcePrior::force_prior_` is consumed upstream (`force_prior.h` was not in the
   archived source set), so the port's absolute scale cannot be certified from the
   archived evidence alone — callers must normalise explicitly (`WITHHELD`).
5. **[minor, test quality]** `sphinxys_xval.py` asserts a *magnitude* against a
   reference whose own normalisation was never calibrated against an analytic force
   field; the pass/fail gate therefore tested the estimator's bookkeeping, not the
   model. The control that would have caught this (O1/O5) was absent.

## 6. Checks not run

* One-sided `delta="unit"` mode and the density-ratio sweep: not re-audited here (the
  reviewed numbers are consistent with the `2 f_k` weighting explanation and the
  stress reading agrees with `∫|∇c|dn`).
* Momentum-conservation residual with the hourglass penalty **on**: not re-measured;
  the reviewed 0.73 residual is a documented property of a deliberately
  non-conservative penalty and is out of scope for this audit.
* Three-dimensional version, contact-angle wall model, and any dynamic (time-stepping)
  validation: not run. The oscillating-droplet dispersion test would be the natural
  next step to decide finding 2 (whether the 30 % bias matters dynamically), and is
  left as a follow-up.
* Upstream SPHinXsys was not compiled or executed (`force_prior.h`,
  `surface_tension.hpp` call sites beyond the archived files): the port was judged by
  transcription, not by numerical agreement with the C++ code.

## 7. Disposition

**Request changes** (verification-debt closure), not approve:

* accept the stress reading as REAL evidence (δ-integral 0.960–0.984, `Δp` within
  2 % of `σ/R`);
* re-classify the flux reading: `Δp_flux = 0.69–0.73 σ/R` (−27…−31 %), i.e. the CSS
  force assembly is **WITHHELD**, not "fundamentally zero";
* carry findings 2 and 3 into the backlog: either (a) demonstrate that the 30 % bias
  is immaterial dynamically (oscillating-droplet / capillary-wave dispersion), or
  (b) replace the force assembly by a form whose flux reading converges.

## 8. Reproduction

```bash
PYTHONPATH=src python benchmarks/surface_tension/css_flux_audit.py --levels 16,32,64
PYTHONPATH=src pytest -q tests/test_sphinxys_css_flux_audit.py
```

Artifacts: `benchmarks/surface_tension/out_css_flux_audit/audit.json` (every number
above) and `audit.png` (left: as-reported vs normalised reading against the exact
`σ/R`, with `m_i` overlaid; right: the O1 operator calibration).