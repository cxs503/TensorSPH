# CASE-HOLLOW-SPHERE-3D — the spherical Young-Laplace jump, `Δp = 2σ(1/R_in + 1/R_out)`

**Verdict: every gate passes.** The projection route with imposed curvature
reproduces the closed form to **0.10 %** on the finest hollow
sphere, the self-consistent `div n` curvature route (the *shipped* CSF route) to
**2.87 %**, and both curvature radii are recovered
independently.

Why the case exists: the 2-D anchor `CASE-LAPLACE-SPH`
(`benchmarks/laplace_sph/`) uses a **circle**, where the second principal
curvature is identically zero. A solver that computes `σ/R` instead of
`2σ/R` — a factor-2 error that appears the moment free-surface codes are lifted
from 2-D to 3-D — passes it. The spherical shell is the cheapest geometry that
cannot be fooled.

---

## 1. Problem setting and reference

Liquid between `R_in` and `R_out`, gas (same-density carrier) in the cavity and
outside. The exact static solution is a two-level staircase

```
p = p_∞ + 2σ/R_out                     in the liquid shell
p = p_∞ + 2σ(1/R_in + 1/R_out)         in the cavity
```

so `Δp = 2σ(1/R_in + 1/R_out)`; `R_in → 0` degenerates to a drop (`2σ/R`) and
`R_in → R_out` to a thin film (`4σ/R`, the soap bubble). Both limits are
asserted in `tests/test_hollow_sphere_3d.py`.

| parameter | value |
|---|---|
| σ (water/air, 25 °C) | 0.072 N/m |
| ρ₀ | 1000.0 kg/m³ |
| Δx (lattice; each level scaled by Δx) | 1.0 m |
| R_in/Δx, R_out/Δx (shell ladder) | 4/10, 6/14, 8/18 |
| R/Δx (drop ladder) | 4, 6, 8, 10 |
| Bond / Weber | 0 / 0 (statics) |
| kernel | 3-D cubic spline, h = 1.3 Δx, carrier = cubic lattice |

References: Young (1805) / Laplace (1806); the two-surface film formula
`Δp = 2σ(1/R₁+1/R₂)` in Batchelor (1967), *An Introduction to Fluid Dynamics*,
sec. 3.9; CSF of Brackbill, Kothe & Zemach (1992), *J. Comput. Phys.* **100**:335;
projection SPH of Cummins & Rudman (1999), *J. Comput. Phys.* **152**:584.

## 2. Method

A WCSPH Tait EOS cannot carry this jump: at the 2-D anchor's numbers the density
ratio needed is `Δp/B ≈ 1e-5` against `1e-3` density noise (recorded in
`docs/verification_laplace_sph_isph.md`), and the 3-D case is worse because
`Δp = 2σ/R` is *twice* the 2-D value only in the numerator while the noise of a
3-D kernel sum is larger. So the pressure is the Lagrange multiplier of a
projection:

```
a_CSF = -σ κ δ_s n / ρ ,     κ = ∇·n ,  δ_s = |∇c|
-∇²p  = -ρ ∇·a_CSF ,         p = 0 on the outer carrier layer
```

with D-generic MLS operators (the shipped 2-D `csf_consistent` matrix is
hard-coded `2×2`; the module re-derives the same construction as
`threed.csf3` so the case verifies the *shipped* 3-D operator, not a private
variant). Two curvature routes:

* `analytic`: `κ = 2(n·r̂)/R_face` imposed — isolates colour → δ_s → traction →
  Poisson → jump.
* `mls`: `κ = ∇·n` from the particle field — the self-consistent route, scored
  against the same closed form.

Three independent cross-checks: the per-face traction flux
`-∫ρ a_r dV /(4πR²)` (never touches the Poisson solve), the δ_s normalisation
`∫δ_s dn = 1`, and the momentum-balance residual.

## 3. Results

### 3.1 Gates (fixed before the run)

| gate | measured | limit | result | detail |
|---|---|---|---|---|
| `laplace_analytic_finest` | 0.001036 | 0.05 | **PASS** | shell 8/18 dx |
| `laplace_analytic_all_levels` | 0.03397 | 0.05 | **PASS** | drops R/dx in 4, 6, 8, 10; shells 4/10, 6/14, 8/18 |
| `laplace_mls_finest` | 0.02871 | 0.05 | **PASS** | self-consistent div-n curvature, shipped CSF route |
| `flux_both_faces` | 0.04204 | 0.08 | **PASS** | inner 0.9580 x 2 sigma/R_in, outer 0.9853 x 2 sigma/R_out |
| `kappa_two_radii` | 0.0008942 | 0.05 | **PASS** | analytic route; mls numbers reported per row |
| `delta_normalisation` | 0.018 | 0.05 | **PASS** | int delta_s dn = 0.9820 |
| `negative_control_sigma0` | 0 | 1e-10 | **PASS** | sigma = 0 shell must carry no rejection pressure |

Negative control (`σ = 0`, hollow sphere): `Δp = 0 Pa`.

### 3.2 Hollow sphere, analytic curvature

| $R_{in}/dx$ | $R_{out}/dx$ | $N$ | $\Delta p_{theory}$ [Pa] | $\Delta p_{SPH}$ [Pa] | err [%] | $\kappa_{in}$ err [%] | $\kappa_{out}$ err [%] | $\int\delta_s\,dn$ | $|a_{spur}|/|a_{CSF}|$ | CG it |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 10 | 11513 | 0.0504 | 0.0491857 | -2.41 | +0.31 | -0.07 | 0.9624 | 0.178 | 33 |
| 6 | 14 | 24405 | 0.0342857 | 0.0341811 | -0.31 | +0.13 | -0.09 | 0.9800 | 0.163 | 37 |
| 8 | 18 | 44473 | 0.026 | 0.0259731 | -0.10 | +0.09 | -0.06 | 0.9820 | 0.150 | 45 |

### 3.3 Hollow sphere, shipped self-consistent curvature (`div n`)

| $R_{in}/dx$ | $R_{out}/dx$ | $N$ | $\Delta p_{theory}$ [Pa] | $\Delta p_{SPH}$ [Pa] | err [%] | $\kappa_{in}$ err [%] | $\kappa_{out}$ err [%] | $\int\delta_s\,dn$ | $|a_{spur}|/|a_{CSF}|$ | CG it |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 10 | 11513 | 0.0504 | 0.0527313 | +4.63 | +17.20 | +4.40 | 0.9624 | 0.187 | 34 |
| 6 | 14 | 24405 | 0.0342857 | 0.0356785 | +4.06 | +0.51 | +0.36 | 0.9800 | 0.180 | 37 |
| 8 | 18 | 44473 | 0.026 | 0.0267463 | +2.87 | -0.99 | +0.16 | 0.9820 | 0.168 | 46 |

### 3.4 Drop ladder (`R_in = 0`)

Analytic route:

| $R_{in}/dx$ | $R_{out}/dx$ | $N$ | $\Delta p_{theory}$ [Pa] | $\Delta p_{SPH}$ [Pa] | err [%] | $\kappa_{in}$ err [%] | $\kappa_{out}$ err [%] | $\int\delta_s\,dn$ | $|a_{spur}|/|a_{CSF}|$ | CG it |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 2109 | 0.036 | 0.0347771 | -3.40 | — | -0.01 | 0.8970 | 0.145 | 17 |
| 0 | 6 | 4169 | 0.024 | 0.0237053 | -1.23 | — | -0.12 | 0.9761 | 0.160 | 20 |
| 0 | 8 | 7153 | 0.018 | 0.0178765 | -0.69 | — | -0.05 | 0.9623 | 0.157 | 24 |
| 0 | 10 | 11513 | 0.0144 | 0.0142333 | -1.16 | — | -0.06 | 0.9754 | 0.150 | 28 |

Self-consistent route:

| $R_{in}/dx$ | $R_{out}/dx$ | $N$ | $\Delta p_{theory}$ [Pa] | $\Delta p_{SPH}$ [Pa] | err [%] | $\kappa_{in}$ err [%] | $\kappa_{out}$ err [%] | $\int\delta_s\,dn$ | $|a_{spur}|/|a_{CSF}|$ | CG it |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 2109 | 0.036 | 0.0402056 | +11.68 | — | +0.57 | 0.8970 | 0.137 | 18 |
| 0 | 6 | 4169 | 0.024 | 0.0249689 | +4.04 | — | -0.55 | 0.9761 | 0.160 | 20 |
| 0 | 8 | 7153 | 0.018 | 0.0186092 | +3.38 | — | +0.81 | 0.9623 | 0.158 | 24 |
| 0 | 10 | 11513 | 0.0144 | 0.0145641 | +1.14 | — | +0.35 | 0.9754 | 0.158 | 29 |

### 3.5 Figures

* `benchmarks/hollow_sphere_3d/figures/dp_ratio.png` — `Δp/Δp_theory` vs `R_out/Δx`,
  both geometries and both curvature routes, with the 5 % band.
* `benchmarks/hollow_sphere_3d/figures/p_profile.png` — the whole radial pressure
  field of the finest shell against the exact staircase (this is the figure that
  would break if only one interface were handled).
* `benchmarks/hollow_sphere_3d/figures/curvature.png` — `κ` error on each face,
  i.e. the curvature operator on its own: two different radii recovered
  separately (`κ_in = -2/R_in`, `κ_out = +2/R_out`).

## 4. Error judgement (vs the closed form, item by item)

| quantity | reference | measured | error | < 3 %? |
|---|---|---|---|---|
| `Δp`, shell shell 8/18 dx (analytic κ) | 0.026 Pa | 0.0259731 Pa | -0.10 % | **yes** |
| `Δp`, shell 8/18 (self-consistent κ) | same | 0.0267463 Pa | +2.87 % | yes (≤3 %) |
| `Δp`, drop `R/Δx = 10` (self-consistent κ) | 0.0144 Pa | 0.0145641 Pa | +1.14 % | yes |
| inner-face flux, 8/18 | 0.018 Pa (`2σ/R_in`) | 0.0172433 Pa | -4.20 % | no (−4 %) |
| outer-face flux, 8/18 | 0.008 Pa (`2σ/R_out`) | 0.00788227 Pa | -1.47 % | yes |
| ∫δ_s dn (8/18) | 1 | 0.9820 | -1.80 % | yes |

The inner-face flux is the one number **outside the 3 % bar** (−4 %): the
inner interface has the *smaller* area, so the same absolute particle-count
error of the δ_s shell integral weighs more there. It stays inside the case's
own 8 % flux gate; it is reported as such rather than being re-normalised away.

## 5. Notes, limits, scope

* **Not a multiphase flow model.** The gas is a same-density carrier that
  completes the colour `1 → 0`; only statics is verified. This is the standard
  static-verification practice and is what makes `∫δ_s dn = 1` hold.
* **Spurious acceleration is reported, not gated** (`|a_spur|/|a_CSF|` column,
  order 0.15). The 2-D anchor records the same
  lattice artefact as an honest FAIL (`spurious_currents_adami_0p1` in
  `docs/verification_laplace_sph_isph.md`); the projection removes only the
  divergent part of the CSF traction. No time integration is attempted here —
  the 2-D `relax_projection` loop is already documented as diverging.
* **A useful independent observation:** the 3-D `div n` curvature on a cubic
  lattice is far better than the 2-D result. The 2-D hex-lattice audit needed
  `R/Δx ≥ 10` for `div n` to reach 15 % (`R/Δx = 5` gave +25 %,
  `docs/verification_surface_tension_consistent.md`), while in 3-D at
  `R_out/Δx = 4` it is already within 1 % on the drop. The 3-D stencil carries
  `O(h²)` more information per particle, which is why the self-consistent route
  meets the gate here.
* The `analytic` curvature route is a *gauge*, not a model: it must not be used
  to claim predictive capability, only to bound the instrument error of the
  rest of the chain. The self-consistent column is the honest capability number.
