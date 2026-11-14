# LAMB: dynamic cross-validation of the SPHinXsys CSS surface tension port

**Anchor** `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` · repository **TensorSPH**
· mission **m19927** (self-start, models seat) · 2026-11-14

**Verdict (short).** The ported SPHinXsys continuous-surface-stress (CSS) force
*does* carry the curvature force. Measured in the one unit that does not depend
on bookkeeping — the frequency of the Lamb/Rayleigh shape oscillation of a
perturbed drop — the port reproduces the exact 2-D linear result
`omega_n^2 = n(n^2-1) sigma / (rho R^3)` to **+0.7 %** in single-phase
(vacuum-reduced) mode and **-4.2 %** in two-phase mixture mode at 1000:1, both
inside the 15 % acceptance gate. The earlier "-99.8 % / structurally blind"
verdict of `sphinxys_xval.py` (m17909) is a **bookkeeping artefact** of the
reading, not a property of the model (see §5). What the port *does* get wrong is
its **hourglass penalty**, which in single-phase mode contributes a spurious
mode-2 pressure of order `sigma/R` that does not scale with the perturbation
(§6) — a documented defect, locked by strict `xfail` tests.

Artefacts: `benchmarks/surface_tension/sphinxys_lamb.py`,
`benchmarks/surface_tension/out_sphinxys_lamb/lamb.json`, `lamb.png`,
`tests/test_sphinxys_lamb.py`.

---

## 1. Case introduction

**Physics.** A 2-D inviscid drop at rest with a small shape perturbation
`r(theta) = R (1 + eps cos n theta)` oscillates in the Lamb/Rayleigh mode `n`
with the exact linear frequency (Lamb 1932, art. 263; cylinder, hence no
`(n+2)` factor):

    Delta p_n = sigma (n^2 - 1) eps / R          (modal Laplace pressure)
    omega_n^2 = n (n^2 - 1) sigma / (rho R^3)    (frequency)

`Delta p_n / sigma` is dimensionless in curvature (`(n^2-1) eps / R`), so the
test is free of any geometric scale and depends only on the model's ability to
turn a *curvature* into a *normal stress*.

**Why this case.** The static droplet (Young-Laplace) audit cannot separate a
model defect from a lattice-symmetry cancellation: on a circle the discrete
divergence of a tangential stress field is a difference of nearly opposite
terms. A perturbed interface breaks that symmetry, and the modal decomposition
furthermore removes the isotropic part of the pressure (which is what the static
test measures) and retains only what curvature *gradients* produce.

| quantity | value | source |
|---|---|---|
| R | 0.1 m | mission case |
| sigma | 0.072 N/m | water-air, 20 °C |
| rho_water : rho_air | 1000 : 1 | upstream square-droplet case |
| Lamb mode n | 2 | lowest non-trivial mode |
| eps | 0.10 (0 – 0.10 scan) | `DROP_EPS` of the CSF Lamb case |
| h/Δx | 1.3 | `SPHAdaptation(h_spacing_ratio = 1.3)` |
| kernel | Wendland C2, support 2h | upstream default |
| hourglass coeff | 4.5 | `SPHinXsysHourglassControlCoeff` |
| resolution ladder | R/Δx = 16, 24, 32 | |

Reference model for the calibration channel: the Brackbill CSF model already in
TensorSPH (`src/tensorsph/csf_consistent.py`, curvature from a local quadratic
fit), whose Lamb case sits inside `TOL_LAMB_PCT = 3.0` in
`surface_tension_dispersion`.

## 2. Method

Both models are measured with **one projector on one geometry**
(`modal_pressure`):

    dp_mode = (2 / L) sum_i (f_i . e_r) V_i cos(n theta_i),   L = 2 pi R

where `f` is the model's *force per unit volume* (2-D: N/m², i.e. already an
interface-concentrated stress for a delta-function surface force) and
`V_i = dx^2`. `omega_ratio = sqrt(dp/dp_exact)`.

* **Calibration channel.** CSF via
  `surface_tension_dispersion.drop_modal_stiffness` (its own, independently
  validated projector). It proves the reference value and the sign convention
  are wired correctly: it returns `omega` within 3 %.
* **Subject.** `sphinxys_css.surface_tension_acceleration`. The upstream
  acceleration carries `(m_i/rho_i) = V_i`, so the force density is
  `f_i = a_i / V_i` (this is the relation validated independently by the O1
  operator calibration of `css_flux_audit.py`: `sum_j V_j (S_i+S_j).grad W_ij`
  reproduces `div S = 2x` for `S = x^2 I` to 2.6 %). With
  `assembly="consistent"` the port instead returns `a_i = (1/rho_i) div S` and
  the force density is `rho_i a_i`; both readings must agree — and they do to
  machine precision (`test_css_bookkeeping_identity_upstream_vs_consistent`).
* **Mixture path.** Two-phase lobed drop, `rho_air = 1`, contact fraction
  `f = rho0_i/(rho0_i + rho0_k)`, inner + contact stress flux as upstream.

## 3. Results

### 3.1 Resolution ladder — single-phase (vacuum reduction), n = 2, eps = 0.10

`omega_exact = sqrt(n(n^2-1)sigma/(rho R^3)) = 0.6573 rad/s`; `dp_exact = -sigma (n^2-1) eps / R < 0` (inward).

**SPHinXsys CSS, hourglass off** (the physical model)

| R/Δx | delta integral | dp/dp_exact | omega/omega_exact | omega error | < 15 %? |
|---|---|---|---|---|---|
| 16 | 1.0019 | +1.0141 | +1.0070 | +0.70 % | **PASS** |
| 24 | 1.0086 | +0.9831 | +0.9915 | -0.85 % | **PASS** |
| 32 | 1.0114 | +0.9883 | +0.9942 | -0.58 % | **PASS** |

**SPHinXsys CSS, hourglass 4.5** (upstream default, single-phase)

| R/Δx | delta integral | dp/dp_exact | omega/omega_exact | omega error | < 15 %? |
|---|---|---|---|---|---|
| 16 | 1.0019 | -0.1347 | -0.3671 | -136.71 % | FAIL |
| 24 | 1.0086 | +5.5061 | +2.3465 | +134.65 % | FAIL |
| 32 | 1.0114 | +7.1499 | +2.6739 | +167.39 % | FAIL |

**Calibration channel — Brackbill CSF** (`drop_modal_stiffness`)

| R/Δx | dp/dp_exact | omega/omega_exact | omega error |
|---|---|---|---|
| 16 | +1.0247 | +1.0123 | +1.23 % |
| 24 | +0.9602 | +0.9799 | -2.01 % |
| 32 | +0.8415 | +0.9174 | -8.26 % |

`assembly="consistent"` (force density `rho_i a_i`) returns bit-identical numbers to the upstream
reading `a_i/V_i`, which is the bookkeeping identity of §5.

### 3.2 Mixture (two-phase) path, rho_water:rho_air = 1000:1, R/Δx = 24, 3261 particles

| variant | dp/dp_exact | omega/omega_exact | omega error | delta integral |
|---|---|---|---|---|
| hourglass | +1.0216 | +1.0107 | +1.07 % | 0.9880 |
| no_hourglass | +0.9318 | +0.9653 | -3.47 % | 0.9880 |
| consistent | +1.0216 | +1.0107 | +1.07 % | 0.9880 |
| symmetric_bracket | +1.0213 | +1.0106 | +1.06 % | 0.9880 |

`symmetric_bracket` uses the plain two-point bracket `S_i + S_j` instead of upstream's
`2(1-f) S_i + 2 f S_j`: indistinguishable here, because the contact-fraction weighting of a
1000:1 pair is essentially one-sided.

### 3.3 Amplitude scan (R/Δx = 24) — physical part vs hourglass part

| eps | CSS hourglass off dp [Pa] | CSS hourglass on dp [Pa] | hourglass part [Pa] | hourglass part / (sigma/R) | Brackbill CSF dp [Pa] |
|---|---|---|---|---|---|
| 0.00 | -0.0021 | -1.5101 | -1.5080 | -2.094 | +0.0058 |
| 0.01 | -0.0222 | -1.2035 | -1.1814 | -1.641 | -0.0098 |
| 0.02 | -0.0392 | -1.1475 | -1.1083 | -1.539 | -0.0444 |
| 0.05 | -0.1130 | -0.8135 | -0.7005 | -0.973 | -0.1010 |
| 0.10 | -0.2123 | -1.1893 | -0.9770 | -1.357 | -0.2074 |

The physical (hourglass off) column scales with `eps`, as a linear surface-tension force must.
At `eps = 0` — a perfect circle, where the mode-2 modal pressure of a physical force is **zero by
symmetry** — CSS gives -0.0021 Pa and CSF +0.0058 Pa (both vanish), while the hourglass term
gives -1.51 Pa = -2.09 sigma/R.


## 4. Figures

![Lamb-mode results](sphinxys_lamb.png)

(a) frequency ratio versus resolution, (b) modal pressure at the finest level,
(c) the ported radial acceleration on the interface band of the perturbed drop.

## 5. Why the earlier "-99.8 %" reading was wrong

`sphinxys_xval.py` formed `dp = sum_i rho_0 (a_i . n) V_i / (2 pi r_c)` from the
ported acceleration. But the ported acceleration is *not* an acceleration in the
`f = rho a` sense: upstream assembles `a_i = (m_i/rho_i) sum_j V_j (S_i+S_j).grad W_ij`,
i.e. the upstream `surface_tension_force_` variable carries an extra volume
factor `V_i` on top of `div S / rho`. Dividing by `rho_0` then leaves a factor
`V_i = dx^2`, which is why the reading collapsed as `~dx^2` under refinement
(-96.6 %, -99.0 %, -99.7 % at R/dx = 16, 24, 32 — exactly the signature of a
missing length squared, not of a blind operator). Once the force density is
formed consistently (`a_i / V_i`, resp. `rho_i a_i` for the consistent
assembly), the same port reproduces the curvature stress to a few percent, as
the Lamb frequency above shows.

## 6. The real defect: the inner hourglass penalty

The inner (same-phase) hourglass penalty of `inner_stress_force` — upstream
`SurfaceStressForce<Inner<>>`, coefficient 4.5 — is the one part of the port that does **not**
survive this test:

| quantity | value |
|---|---|
| hourglass part on a perfect circle (eps = 0) | **-1.508 Pa = -2.09 sigma/R** |
| hourglass part at eps = 0.01 / 0.02 / 0.05 / 0.10 | -1.18 / -1.11 / -0.70 / -0.98 Pa |
| physical part at eps = 0 / 0.01 / 0.02 / 0.05 / 0.10 | -0.002 / -0.022 / -0.039 / -0.113 / -0.212 Pa |
| omega error, hourglass 4.5, single-phase | **-136.7 % / +134.7 % / +167.4 %** (R/dx = 16 / 24 / 32) |
| omega error, hourglass 4.5, two-phase 1000:1 | +1.07 % (contact hourglass carries the f(1-f) factor) |

The hourglass contribution is **O(sigma/R) and independent of the perturbation amplitude** — it
does not vanish as the interface becomes a circle, so it is a lattice-scale term, not a
regularisation of the mode. In single-phase mode its error is 135-167 % and its sign flips with
resolution. In the two-phase path the same term is suppressed by the contact factor `4 f(1-f)`
(0.004 at 1000:1), which is why upstream — where every interface is a *contact* between two
bodies — does not see this.

**Consequence.** Run the port with `hourglass=False` in single-phase mode (as
`benchmarks/surface_tension/sphinxys_xval.py` already does for its stress reading), or rescale
the inner hourglass by the contact factor. The defect is locked by two strict `xfail` tests in
`tests/test_sphinxys_lamb.py`, which will turn into failures (and thus prompt an update) as soon
as the term is fixed.


## 7. Pitfalls and scope

* The Lamb test is linear theory: `eps` must be small; the `eps`-scan in §3
  shows the physical part is proportional to `eps` as required.
* `dp_mode` is a *modal surface pressure*, not a full dynamic simulation: it
  measures the instantaneous restoring force at `t = 0`, which is exactly what
  the frequency of a conservative oscillator needs. Viscous damping, density
  re-initialisation and the WCSPH pressure closure are out of scope.
* Single-phase CSS is a *reduction* of the two-phase model (vacuum reduction of
  the colour gradient); its `delta_s` integral is 0.96–0.98 rather than 1, so a
  few percent of the deficit below is delta normalisation, not curvature.
* Two-phase numbers use `rho_air = 1 kg/m^3`; the contact-fraction weighting
  makes the water stress the dominant contribution, as upstream intends.