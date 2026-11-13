# XVAL round 2 — mixture material & surface tension: where the Laplace error comes from and how it is fixed

**Anchor** `XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION` · **Mission** m18642 · **Repo** TensorSPH
**Round** 2 (survey → combine). Round-1 failure of record: mission m18603 (`negk:loop:18603`, no
`fail_stage`/`failure_reason` left), with the substantive m17909 study
(`dev_report_sphinxys_mixture_surface_tension.md`) reporting `flux_reading` errors of
−96.6 % … −99.8 % against Young–Laplace.

All numbers below are generated: `docs/xval_sphinxys_mixture_laplace/tables.md` is emitted from
`benchmarks/surface_tension/out_mixture_laplace/gate.json` by
`runs/m18642_round2/emit_tables.py`. Figure `benchmarks/surface_tension/out_mixture_laplace/gate.png`.

---

## 1. Problem setting and reference

**Physics.** For a 2-D circular interface in static equilibrium (no gravity, no viscosity, no
velocity), the jump of normal stress across the interface is exactly

  Δp = σ / R        (Young–Laplace, 2-D; the 3-D sphere carries the factor 2)

with σ = 0.072 N/m (water–air, 20 °C) and R = 0.1 m, giving Δp = 0.7200 Pa. This is an *exact*
anchor, not a fitted one, so the 15 % acceptance criterion of the ledger is a pure
discretisation/model error budget.

**Reference implementation (OSS horizontal survey).** SPHinXsys (`Xiangyu-Hu/SPHinXsys`), the
CSS ("continuous surface stress") surface-tension model with a mixture contact fraction.
The upstream sources were fetched through the GitHub contents API and archived byte-exactly in
`docs/xval_sphinxys_surface_tension/raw/` with `MANIFEST.json`
(`surface_tension.cpp` sha256 `04b9ad3a…`, `surface_tension.hpp` `76279a06…`,
`kernel_wendland_c2.cpp` `da0f2e72…`, `adaptation.h` `2bca4b9a…`, the square-droplet case
`d1c90cd3…`). The ported model lives in `src/tensorsph/sphinxys_css.py`; this report adds
two switchable knobs to it (`assembly`, `bracket`) and nothing else.

**Method genealogy / external anchors** — see `method_selection.md` in this directory:
Brackbill–Kothe–Zemach (1992, CSF), Morris (2000, SPH colour function), Monaghan (1992,
symmetric SPH stress divergence), Adami–Hu–Adams (2010, reproducing divergence), plus the
SPHinXsys mixture implementation. Only the two analytic anchors (Laplace law, and the
analytic-stress operator test of §2.1) are used for the gate; the papers supply structure, not
numbers.

**Parameters (complete, dimensionless where they exist).**

| quantity | value | note |
|---|---|---|
| R | 0.1 m | droplet radius |
| σ | 0.072 N/m | water–air, 20 °C |
| ρ_w / ρ_a | 1000 : 1 | upstream square-droplet ratio |
| h/Δx | 1.3 | `SPHAdaptation(h_spacing_ratio = 1.3)` |
| kernel | Wendland C₂, support 2h | upstream default |
| hourglass coeff | 4.5 | upstream default |
| R/Δx | 16, 32, 64 | ladder; 1 517 / 4 513 / 15 373 particles |
| Laplace group | σ/(ρ_w c² R) with c = 10 U_ref | static case: no We, no Oh, no Bo enters |
| Ohnesorge | — | inviscid: viscosity never enters |
| Weber | 0 | U = 0 |

## 2. Method

The question is narrow and answerable: **which link of the pipeline loses the Laplace jump?**
The pipeline is `mixture weighting f_k → colour gradient → normal → stress S → divergence of S →
acceleration`. Three probes, in increasing order of "blame":

### 2.1 Operator floor (blame the discretisation first)

Feed the *identical* discrete pair assembly an **analytic** continuous surface stress
`S = σ f(r)(I − n nᵀ)`, `f(r) = exp(−((r−R)/w)²)/(w√π)` so that `∫f dn = 1`. Its exact
divergence is `−σ f(r) n / r`, so the exact reading is σ/R and no colour field, kernel or mixture
weighting is involved. Result (T1): **−2.6 … −2.7 % at every resolution and every width w/h ∈
{0.5, 1, 2}** — the operator is sound and its floor is ~2.6 %.

A by-product worth recording: replacing the symmetric two-point bracket `(S_i + S_j)` by the
difference `(S_j − S_i)` gives **exactly zero** reading (−100.0 %, to 1e−17) at every resolution.
For a second-rank tensor the first non-vanishing Taylor term of the difference form is odd in the
kernel-gradient direction `e_ij` and cancels on any isotropic particle neighbourhood. The
obvious-looking "first-order" fix is a trap; this is the negative result of the round.

### 2.2 Normalisation of the pair weight (the actual defect)

The transcription accumulates

  `a_i = (m_i/ρ_i) Σ_j V_j (…)·∇_i W_ij`,  i.e. weight `= (m_i/ρ_i) V_j = V_i V_j`.

The SPH momentum equation needs `a_i = (1/ρ_i) ∇·S_i = (1/ρ_i) Σ_j V_j (…)·∇_i W_ij`, i.e. weight
`V_j/ρ_i`. The two differ by **exactly the particle mass `m_i = ρ_i Δx²`** — a factor that vanishes
with the grid, which is why the round-1 error grew *worse* with refinement (−96.6 % → −99.7 %,
`dx²` decay) while every other quantity converged. `ASSEMBLIES` in
`src/tensorsph/sphinxys_css.py` now exposes both; the test
`tests/test_mixture_laplace.py::test_assembly_normalisation` locks the identity
`upstream/consistent = m_i` to 1e−10 on a uniform-density lattice, and bounds it between `Δx²` and
`ρ_w Δx²` in the two-phase mixture (per pair class: water–water pairs carry `ρ_w Δx²`, air–air and
cross pairs `Δx²`).

### 2.3 Which normal integral *is* the Laplace jump

The mixture colour gradient is **one-sided**: a water particle builds `∇c` from air neighbours
only, so the surface delta `|∇c|` sits mostly on the water side of the interface (measured
`∫|∇c| dn = 0.9605` on the water side, 1000:1). Integrating `ρ a·(−n)` over the water disc
therefore samples only part of the interface shell. Measured with the *same* force field:
water-only −29.8 %, both phases −4.7 % (T2, R/dx = 16). The Laplace jump is the integral of the
surface force density **across** the interface, so the both-phase reading is the physical one; the
water-only reading is kept in the tables as a documented estimator bias, not a model error.

### 2.4 Stress bracket

Upstream assembles the cross-phase flux with `2(1−f) S_i + 2 f S_j`. That bracket is the same at
both ends (because `f_j = 1−f_i`), but it is **not** the symmetric two-point form of one stress
field: at f → 1 the light phase's stress `S_j` is multiplied by 2 while the heavy phase's `S_i` is
dropped, and the light particles — which receive the *same* pair force but 1/ρ_a = 1000× the
acceleration — end up pushing outward, i.e. against the tension. Measured: two-phase,
consistent weight, R/dx = 16, both-phase reading, `bracket="mixture"` **−29.7 %** vs
`bracket="symmetric"` **−4.7 %** (T2). `bracket` is therefore a switch, default `"mixture"`
(upstream-faithful), gated configuration `"symmetric"`.

The mixture weighting itself is *not* discarded: it stays in the colour gradient
(`f_k = ρ0_i/(ρ0_i+ρ0_k)`), which is the anchor's "mixture material" element, and its quantitative
effect is measured in §4.

### 2.5 Gated configuration

`assembly="consistent"` (`V_j/ρ_i`), `bracket="symmetric"`, `hourglass=False` (the upstream
hourglass penalty is antisymmetric under i↔j by construction and breaks momentum conservation —
0.73 relative residual; with it off the residual is < 1e−16 and the Laplace error is unchanged).

## 3. Results

Numbers: `tables.md` (T0–T5), generated from `gate.json`.

**T1 operator floor** −2.61 … −2.70 % (R/dx = 16, 32, 64).

**T2 Laplace ladder, gated configuration (two-phase, consistent, symmetric, hourglass off):**

| R/Δx | both-phase reading Δp/(σ/R) | error | momentum residual |
|---|---|---|---|
| 16 | 0.9531 | −4.69 % | 4.3e−17 |
| 32 | 0.9420 | −5.80 % | 2.9e−17 |
| 64 | 0.9379 | −6.21 % | 6.5e−18 |

**Baseline (faithful transcription, unchanged):** −96.74 %, −99.05 %, −99.69 % at the same three
resolutions — the round-1 result reproduced, now with a mechanism attached.

**Stress reading** σ∫|∇c|dn/R: −3.95 %, −2.22 %, −1.65 % (converging; this half of the pipeline was
already correct in round 1 and is unchanged).

## 4. Convergence and parameter sensitivity (≥ 2 resolutions, ≥ 2 parameter points)

- **Resolution**: the gated error is −4.69 → −5.80 → −6.21 % for R/Δx = 16 → 32 → 64. It does not
  vanish, and it does not diverge: the interface thickness is fixed in units of h (≈ 2h), so the
  residual is the fixed relative bias of a delta-function stress sampled on a lattice, of the same
  size as the operator floor (−2.6 %) plus the one-sided-delta departure from unit integral
  (−4 %). Observation order between the last two points: p = log(5.80/4.69)/log 2 ≈ 0.31 — i.e.
  the error is **saturating at ≈ −6 %**, which is what "the model, not the grid" looks like.
- **Mixture density ratio** (T4, R/Δx = 16, `f_w` = contact fraction): ρ_a/ρ_w = 1 → `∫|∇c|dn` =
  0.4807 and Δp error −11.86 %; 0.1 → 0.8740, −6.16 %; 0.01 → 0.9519, −4.85 %; 0.001 → 0.9605,
  −4.70 %. The delta normalisation degrades by a factor 2 at equal density (the `2 f_k` prefactor
  is calibrated for the one-sided limit), yet the *force* reading stays inside the gate over three
  decades of density ratio — the magnitude error of the mixture stress largely cancels in its
  divergence. This is the honest form of the round-1 observation that "the `2 f_k` prefactor is
  wrong at 1:1": it is wrong, but it is not what decides the force.
- **Hourglass**: on/off changes the gated error by 0.01 pp in two-phase mode; it changes the
  momentum residual from 7.3e−1 to 4e−17.

## 5. Failure modes, pitfalls, limits

1. **The upstream factor is invisible in a dynamic test.** A square droplet driven by a force
   density ∝ Δx² still deforms (weakly); SPHinXsys validates that case with a self-referential
   DTW curve of its own kinetic energy, so a pure scale error in the force cannot be caught there.
   Any XVAL of this model must be a *static exact* anchor (Laplace law), as here.
2. **The hourglass penalty is unstable on a free surface.** In the single-phase (vacuum-reduced)
   configuration the upstream penalty (∝ 1/r_ij) is not balanced by a second phase: with it on,
   the reading is −0.89 % at R/Δx = 16 but −375 % at 32 and +215 % at 64 — a growing,
   resolution-divergent artefact. Single-phase results are therefore reported with the penalty
   off: −29.7 … −30.0 % (stable), i.e. **the vacuum reduction of the mixture model is not
   Laplacian-accurate with this force assembly**; it needs the both-phase shell. Documented, not
   patched over.
3. **Water-only integrals are not the Laplace jump** (see §2.3) — a −30 % apparent error that is
   pure estimator bias.
4. Scope: 2-D, static, inviscid, no gravity, no wall contact, no dynamic multiphase coupling
   (the mixture Riemann solver of upstream is out of scope); the interface-fit operators are 2-D.
5. The bibliographic identifiers in `method_selection.md` were reconstructed from standard
   references and are not re-verified against the publisher PDFs in this run; no number in the
   gate depends on them.

## 6. Reproduction and artifacts

```bash
# gate (7 s on CPU, no cluster needed)
TMPDIR=… PYTHONPATH=src python benchmarks/surface_tension/mixture_laplace.py \
    --radii-dx 16,32,64 --rho-ratios 1,0.1,0.01,0.001
python runs/m18642_round2/emit_tables.py
TMPDIR=… python -m pytest tests/test_mixture_laplace.py tests/test_sphinxys_css.py -q   # 12 passed
```

| artifact | content |
|---|---|
| `benchmarks/surface_tension/mixture_laplace.py` | ladder, operator floor, density-ratio sweep, hourglass ablation, gate |
| `benchmarks/surface_tension/out_mixture_laplace/gate.json` | every number (programmatic) |
| `benchmarks/surface_tension/out_mixture_laplace/gate.png` | 3-panel figure (ladder, gate band, density ratio) |
| `runs/m18642_round2/emit_tables.py` | JSON → `tables.md` (zero hand-copy) |
| `runs/m18642_round2/operator_probe.py`, `profile_vs_normal.py`, `field_sweep.py`, `ablate_structure.py`, `explore_assembly.py` | the diagnostic chain that localised the defect |
| `src/tensorsph/sphinxys_css.py` | `pair_weight`, `assembly`, `bracket` (235 lines touched, all additive) |
| `tests/test_mixture_laplace.py` | 5 invariant tests (operator floor, normalisation identity, gate, upstream regression lock, unit delta) |

## 7. Verdict against the ledger criterion

| criterion | measured | limit | result |
|---|---|---|---|
| Δp vs σ/R, gated config, R/Δx = 16/32/64 | −4.69 / −5.80 / −6.21 % | < 15 % | **PASS** |
| Δp vs σ/R, faithful transcription (baseline) | −96.7 … −99.7 % | < 15 % | FAIL (reproduced, mechanism identified) |
| σ∫|∇c|dn/R (stress half of the pipeline) | −1.65 % (R/Δx = 64) | < 15 % | **PASS** |
| momentum conservation, gated config | ≤ 6.5e−18 | < 1e−12 | **PASS** |
| operator floor (analytic stress) | −2.6 % | < 15 % | **PASS** |

Explicit answer to "is it < 3 %?": **no.** The gated configuration is inside the ledger's 15 %
criterion at all three resolutions (worst 6.21 %), and the residual is a saturating ≈ −6 % model
bias (one-sided mixture delta + lattice sampling of a delta-function stress), not a grid effect.
The 3 % band is reached only by the stress half of the pipeline (−1.65 % at R/Δx = 64), not by the
force.