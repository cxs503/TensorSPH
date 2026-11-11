# Aeration buffer decalibration — Wood / van Wijngaarden low-frequency limit + PBM moment closure → α-resolved impact-pressure law

**Node** `p457w2-n04-aeration-decalibration` · **Repo** TensorSPH ·
**Modules** `tensorsph.aeration.mixture_acoustics`, `tensorsph.aeration.pbm_moments`,
`tensorsph.aeration.impact_law` ·
**Tests** `tests/test_aeration_impact_law_p457w2n04.py` (26 passed) ·
**Evidence** `docs/p457w2-n04/acceptance.json` + `fig1..fig4` (all numbers below are
program output of `scripts/p457w2n04_report.py`, nothing hand-typed).

---

## 1. Case introduction (physics, geometry, parameters)

A bow-flare slam loads the hull through a very short pressure spike whose magnitude is set
by the **acoustic impedance of the water that is stopped** at the shell.  Water is not clear:
breaking waves, spray and the ship's own bow wave entrain air, and a bubbly mixture is
*enormously* more compressible than clear water.  The platform used to model that with a
single calibration constant,

    p_buffer = k_b * rho_w c_w v,     k_b = 0.65        (legacy)

i.e. a blanket −35% on every slam condition.  That is a decalibrated representation: the
physical effect is not a constant, it is a function of the **void fraction**
`α = V_air / V_mixture`.  This node replaces the constant with

    p_peak(α, v) = Z(α) v · C_Mach(α, v) · C_wall(α)

where `Z = ρ_m c_m` is the Wood mixture impedance, and `α(z,t)` comes from a bubble
population balance.

### Phase properties and geometry of the case set

| quantity | value | source |
|---|---|---|
| water ρ_l, c_l | 998.2 kg/m³, 1483 m/s | Del Grosso / Kroebel (20 °C, 1 atm) |
| air ρ_g, c_g, γ_g | 1.204 kg/m³, 343 m/s, 1.4 | ideal gas, 20 °C |
| steel ρ_s, c_s (wall mismatch) | 7850 kg/m³, 5900 m/s | mild steel, longitudinal |
| Wöhler/Hinze critical Weber `We_c` | 4.7 | Hinze (1955) |
| Saffman–Turner prefactor | 1.294 | Saffman & Turner (1956) |
| Coulaoglou–Tavlarides `c1`, `c2` | 0.00481, 0.08 | CT (1981), *literature values, not tuned* |
| cascade slope / lower cutoff | `n(R) ∝ R^{-10/3}`, `R_min = 20 µm` | Garrettson (1973); Deane & Stokes (2002) |

Six bow-flare slam cases span the design envelope (declared kinematics, no calibration):
impact velocity 6–14 m/s, hull inclination 45°–70°, flow depth 1.2–3.0 m, wetted footprint
0.5–2.0 m, peak spectral period 5–9 s (`impact_law.SLAM_CASES`).

## 2. How it was done

**Step 1 — mixture acoustics (closed form, no iteration).**
Wood: `1/(ρ_m c_m²) = α/(ρ_g c_g²) + (1−α)/(ρ_l c_l²)`, `ρ_m = αρ_g + (1−α)ρ_l`.
Writing `A = 1/(ρ_g c_g²)`, `B = 1/(ρ_l c_l²)`, `m = ρ_l − ρ_g`, the stationarity of
`c_m² = 1/[(ρ_l − mα)(B + (A−B)α)]` gives the **minimum-sound-speed void fraction** and the
minimum sound speed in closed form

    α*  = [ρ_l(A−B) − mB] / [2m(A−B)],        c_min = 2√(m(A−B)) / [ρ_l(A−B) + mB]

→ **α\* = 0.50057**, **c_min = 23.810 m/s**.
The van Wijngaarden (1972) low-frequency dilute limit
`1/c² = ρ_l[(1−α)/(ρ_l c_l²) + α/(γ_g p)]` is implemented with an **explicit domain
`α ≤ 0.05`**: beyond it the limit is meaningless (it drops gas inertia and therefore cannot
tend to `c_g`), so the call *fails closed*.

**Step 2 — impedance and impact pressure.**
`Z = ρ_m c_m`; `p = Z(α) v` (1-D Joukowsky/water-hammer, rigid wall); optional first-order
weak-shock Mach term `1 + (Γ_m+1)M/4` with the volume-harmonic mixture exponent `Γ_m`; and
the 1-D elastic-impact mismatch factor `Z_s/(Z_s + Z(α))` (fluid column arresting against a
solid; → 1 for a rigid wall, 0.969 for steel in clear water).

**Step 3 — void fraction from the population balance.**
Moments `M_k = ∫ v^k n dv` of a bubble population driven by *constant-kernel coagulation*
(Saffman–Turner rate averaged over the spectrum, `K_c = 1.294√(ε/ν)⟨(R₁+R₂)³⟩`) and
*size-independent binary equal-split breakup* (Coulaoglou–Tavlarides frequency at the Sauter
size).  That event pair closes **exactly** in `M0..M3`:

    dM0/dt = +K_b M0 − (K_c/2) M0²         dM1/dt = 0
    dM2/dt = +K_c M1² − (K_b/2) M2         dM3/dt = (3/2)K_c M1 M2 − (3/4)K_b M3

with the closed-form solution `M0` = logistic (`→ 2K_b/K_c`), `M1` constant,
`M2 = M2_∞ + (M2_0 − M2_∞)e^{−K_b t/2}` (`M2_∞ = 2K_c M1²/K_b`) and `M3` from the linear ODE
forced by `M2(t)`.  Kernel parameters come from the **aeration source term**
(`entrainment_flux`, Chanson criterion) plus the spectrum integration — `ε = V_e³/δ_a`,
`α` sets the spectrum amplitude, the Sauter size sets the rise velocity that closes
`α = V_e/u_r`; the loop is iterated to a fixed point (3 sweeps, residual reported).

**Step 4 — footprint load and the comparison with `k_b`.**
A load is an *area integral* of the local pressure, so patchwise aeration enters linearly:

    G_load = 1 − [f_A p(α_patch) + (1−f_A) p_clear] / p_clear = f_A · G_peak,
    f_A = min(1, δ_a/L_wet),    δ_a = q_a / v  (entrained air volume per unit width ÷ v)

`G_peak` is the local patch reduction and saturates (a 2% void fraction already removes ~94%
of the water hammer); `G_load` is the like-for-like replacement of the constant.

**Step 5 — fail-closed domain.** `α > α*` or `α < 10⁻⁸` (minimum resolvable void fraction)
returns `OUT_OF_DOMAIN` and, when a bound is wanted, the *boundary* value at `α*` — never an
extrapolation.

Toolchain: pure Python `math` + `mpmath` (60-digit reference) + `pytest`; figures with
matplotlib (Agg). No SPH run is needed: every criterion is algebraic or a small ODE.

## 3. Results

### 3.1 Acceptance criteria (from `docs/p457w2-n04/acceptance.json`)

| # | criterion | measured | threshold | pass |
|---|---|---|---|---|
| C1 | Wood closed form vs 60-digit mpmath (400 points) | **7.1e-15** rel. | ≤1e-6 | ✅ |
| C1 | α\* closed form vs independent ternary search | **8.8e-9** abs. | — | ✅ |
| C1 | `dc_m/dα` at α\* (central difference) | **0.0** | ~0 | ✅ |
| C1 | c_min magnitude (∝"~20 m/s"; value 23.810) | **1.5e-16** rel. vs numerical min | 15–30 m/s | ✅ |
| C2 | α→0 limit vs `ρ_w c_w v` (6 velocities) | **0.0** (exact) | ≤0.1% | ✅ |
| C2 | α→1 limit vs `ρ_a c_a v` | **≤1.8e-16** | ≤0.1% | ✅ |
| C3 | closed form vs RK4 (4000 steps, t=100 s) | **6.8e-11** | ≤1% | ✅ |
| C3 | closure vs direct double-sum collision integral | **1.7e-11** | ≤1e-9 | ✅ |
| C3 | `M1` drift / `M0` decay / `M2` rate | 0 / 2.06e8→1.466e8 / **0.13313 vs K_b/2 = 0.13313** | — | ✅ |
| C4 | PBM α vs specified field (continuous) | **2.2e-16** | ≤5% | ✅ |
| C4 | discrete population vs field | **2.2e-16** | ≤5% | ✅ |
| C5 | design envelope P95 of `G_load` | **10.91%** | [6%, 12%] | ✅ |
| C6 | `p_peak(α)` strictly decreasing, bounded (157 samples × 3 velocities) | monotone | — | ✅ |
| C7 | `α>α*`, `α<α_min`, NaN, negative, vW beyond dilute | all **OUT_OF_DOMAIN** | — | ✅ |

### 3.2 Case set, law vs the constant (`k_b = 0.65`)

| case | v [m/s] | θ [°] | Chanson K | q_a [m²/s] | δ_a [m] | α_patch | local law | f_A | G_peak | **G_load** |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 head-seas-slight | 6 | 45 | 1.47 | 0.0678 | 0.0113 | 0.155 | IN_DOMAIN | 0.0094 | 98.13% | **0.92%** |
| S2 head-seas-moderate | 8 | 55 | 1.95 | 0.3051 | 0.0381 | 0.435 | IN_DOMAIN | 0.0254 | 99.08% | **2.52%** |
| S3 head-seas-rough | 10 | 60 | 2.47 | 0.7365 | 0.0737 | 0.749 | OUT_OF_DOMAIN | 0.0409 | 99.20%\* | **4.06%** |
| S4 bow-flare-buried | 12 | 65 | 3.08 | 1.5003 | 0.1250 | 1.96 | OUT_OF_DOMAIN | 0.0625 | 99.20%\* | **6.20%** |
| S5 design-slam | 14 | 70 | 4.15 | 2.6431 | 0.1888 | 21.5 | OUT_OF_DOMAIN | 0.1259 | 99.20%\* | **12.49%** |
| S6 stern-quartering | 7 | 50 | 1.59 | 0.1491 | 0.0213 | 0.331 | IN_DOMAIN | 0.0213 | 98.86% | **2.11%** |

\* domain boundary bound at α\* (the drift-flux equilibrium exceeds the packing limit for the
strongest three cases — that state is *refused*, not extrapolated; see §5).

Envelope over the six cases: **G_load = 0.92% … 12.49%**, mean 4.72%, **P95 = 10.91%**,
equivalent footprint buffer coefficient `k_b,eq = 0.891 (P05)`.

### 3.3 What the legacy constant was hiding

`k_b,eq(α) = Z(α)/Z(0)` inverts to **α = 8.82e-5** for `k_b = 0.65`.  So the old constant
silently asserted that *every* slam case sits at a void fraction of ~9e-5 — 3 orders of
magnitude below the aerated-patch void fraction the population balance produces (0.15–0.44
in-domain, saturated above).  The law's spread (0.9%–12.5% of the load) is the decalibration:
no single factor can represent that range, and the constant is neither conservative nor
uniform across the envelope.

## 4. Figures

| file | content |
|---|---|
| `docs/p457w2-n04/fig1_sound_speed.png` | `c_m(α)`: Wood vs van Wijngaarden dilute limit, α\*, c_min, resolution floor |
| `docs/p457w2-n04/fig2_pressure_family.png` | `p_peak(α)` family for v = 6…14 m/s; `k_b,eq(α)` with the inverted legacy constant |
| `docs/p457w2-n04/fig3_case_set.png` | per-case `G_load` bars, aerated area fraction `f_A`, `G_peak` markers, m9319 band, legacy 35% line |
| `docs/p457w2-n04/fig4_moments.png` | `M0` logistic decay, `M1` conservation, `M2`/`M3` exponential relaxation |

## 5. Caveats, traps, validity range

1. **Two different "reductions" must not be mixed.** `G_peak` (local patch) saturates at
   ~99%; `G_load` (footprint integral) is 0.9–12.5%. The m9319 band is compared with
   `G_load`, with the definition stated in `acceptance.json → criteria.C5.definition`.
   Reporting `G_peak` as the design reduction would overstate the effect by an order of
   magnitude.
2. **m9319's artefacts are not readable from this worktree** (they live in the
   ship-performance repository). The [6%, 12%] band is used as the *declared* reference band;
   the comparison is therefore a consistency check against a declared band, not a
   data-to-data validation.  Stated, not hidden.
3. **"M2 exponential growth" (campaign sheet) is the complementary sign.** With the declared
   cascade spectrum `M2_0 > M2_∞`, so `M2` relaxes *downwards* at the rate `K_b/2`. The growth
   branch exists and is pinned by `test_c3_m2_grows_exponentially_when_started_below_equilibrium`.
   This is a deviation from the sheet's wording, declared rather than papered over.
4. **Drift-flux saturation.** For the three strongest cases the equilibrium `α = V_e/u_r`
   exceeds unity (21.5 for S5). That is not a void fraction, it says the *dilute* drift-flux
   balance is saturated: the patch is foam at/above the packing limit.  The acoustic law
   refuses the state (C7) and only the α\* boundary bound is used; `aeration_saturated` flags
   it in the JSON.
5. **Low-frequency assumption.** Wood/van Wijngaarden are quasi-static bubble responses
   (`ω r_b/c_m ≪ 1`). At `α = 0.44` and 1 kHz with `r_b = 0.5 mm` the parameter is 0.13 —
   close to the limit; at `α → α*` it reaches 0.26. Exposed as `low_frequency_ratio`; a
   bubbly mixture at resonance needs a dispersive (Commander–Prosperetti) treatment, out of
   scope here.
6. **Entrainment constant `k_a = 0.02`** is the package's declared model constant (the
   entrainment module says explicitly it is not an experimental value). `f_A ∝ k_a`, so
   `G_load` inherits that linear sensitivity: doubling `k_a` doubles `G_load`. This is the
   dominant calibration uncertainty of the chain and it is a *documented knob*, not a fit.
7. **Field/hull coupling.** The wall-mismatch factor is a 1-D plane-wave result (steel → 3.1%
   relief); 3-D hull curvature/added mass and the structural rise-time filter are not in this
   module (the structural response question is a separate, downstream node).

## 6. Error assessment (explicit)

| key quantity | benchmark | error | < 3%? |
|---|---|---|---|
| `c_m(α)` (Wood) | 60-digit mpmath, 400 α points | 7.1e-15 | **yes** |
| `α*`, `c_min` | independent ternary search | 8.8e-9 / 1.5e-16 | **yes** |
| `p(α→0)`, `p(α→1)` | `ρ_w c_w v`, `ρ_a c_a v` | 0 / 1.8e-16 | **yes** |
| moments `M0..M3(t)` | RK4 integration of the same ODEs | 6.8e-11 | **yes** |
| moment closure | direct double-sum collision integral | 1.7e-11 | **yes** |
| `M2` relaxation rate | `K_b/2` | 0.1331278 vs 0.1331278 | **yes** |
| PBM α vs field | specified analytic field | 2.2e-16 | **yes** |
| `G_load` P95 vs m9319 band | declared band [6%, 12%], centre 9% | 10.91% vs 9% → **+21%** | **no** — the band is matched *inside* its own bounds, but the P95 sits 21% above the band centre; the band is a declared envelope, not a measured reference value, so no stricter claim is made |
| `G_peak`, `α_patch`, `f_A` | **no experimental reference available** | — | **cannot be answered**: no bundled slam-aeration measurement of `α` vs `f_A`; the chain is derivation-only (literature constants, declared cutoffs). This is the honest limit of the deliverable. |

Deviation report: all *numerical* criteria are met at machine accuracy; the *physical*
chain carries the calibration uncertainty of `k_a` (§5.6) and no experimental anchor for
`α_patch`/`f_A` — it does not, and cannot be claimed to, reproduce measured slam pressures
to 3%.

## 7. Reproduction

```bash
python -m pytest tests/test_aeration_impact_law_p457w2n04.py -q   # 26 passed
python scripts/p457w2n04_report.py                                # rewrites acceptance.json + figs
```

Deterministic (no RNG, no time dependence): re-running reproduces `acceptance.json`
bit-for-bit on the same phase properties.