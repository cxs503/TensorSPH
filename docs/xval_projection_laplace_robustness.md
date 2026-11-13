# Round 2 — the error budget of the projection-SPH Young-Laplace claim

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m18632**
· round 1 = `docs/xval_projection_laplace_oss.md` (m17594).

Round 1 measured `dp = sigma/R` to **−0.21 %** at `R/dx = 20` for this repo's
projection-SPH droplet and cross-validated it against the PySPH operators. All of
those numbers came from **one** experimental set-up. A static droplet is a physical
solution; nothing in the Young-Laplace law selects a lattice alignment, a container
size or a sampling band. So round 1 could not separate *method accuracy* from
*lattice luck*.

Round 2 measures that separation. One knob is varied at a time around the round-1
baseline, 51 equilibria in total, and every row is read with **two independent
instruments**:

* `dp_pressure` = `⟨p⟩_(r<0.5R) − ⟨p⟩_(1.25R<r<1.45R)` from the projection solve;
* `dp_flux` = `−∫ ρ a_r dA /(2πR)`, the radial integral of the CSF traction alone —
  it never touches the Poisson solve.

Artifacts (no number in this page is typed by hand; all quoted values are in the JSON):

| artifact | what |
|---|---|
| `scripts/xval_projection_laplace/robustness.py` | the six-axis audit + figure generator |
| `scripts/xval_projection_laplace/results/robustness.json` | 51 equilibria, every number below |
| `scripts/xval_projection_laplace/results/robustness.png` | fig. 1 (4 panels) |
| `tests/test_xval_projection_laplace_robustness.py` | 10 gates (spreads, not point values) |
| `src/tensorsph/laplace_isph.py` | `hex_block`/`laplace_isph_case` gained `center=` and `angle=` |

## ① Case and parameters

2-D circular droplet in a filled (equal-density) vapour carrier — the standard static
verification set-up, unchanged from round 1.

| quantity | value |
|---|---|
| interface | circle, `R = 0.02 m` (principal curvatures `1/R`, `0`) |
| surface tension | `sigma = 0.072 N/m` (water/air, 20 °C) |
| density (both phases) | `rho0 = 1000 kg/m³` |
| reference | `dp = sigma/R = 3.6000 Pa` |
| baseline lattice | triangular, `dx = 1 mm` (`R/dx = 20`), `h = 1.3 dx`, N = 4773 |
| baseline gauge | Dirichlet `p = 0` on the outer layer of a block `L = R + 12 dx` |
| scheme | Shepard colour → MLS gradient → smoothed normal → `div n` curvature (MLS) → Brackbill `|grad c|` → CSF traction → Cummins & Rudman pair Laplacian, matrix-free CG |

The owner's acceptance is `< 15 %` on the jump. Gates were **declared before the
numbers were inspected** (`GATES` dict in the script, copied into the JSON):
every single configuration < 15 %; translation and orientation spreads < 15 %; the
gauge, readout and `h/dx` spreads < 5 %; resolution levels `R/dx ≥ 20` < 5 %.

## ② Method — what each axis asks

| axis | varied | the physics question |
|---|---|---|
| A translation | interface at 12 sub-cell offsets `(f_x dx, f_y dy)`, `f ∈ {0, 1/4, 1/3, 1/2, 2/3}` | a static drop is translation invariant |
| B orientation | lattice basis rotated by `0, 5, 10, 15, 20, 30°` | capillary force must not be lattice-orientation dependent (triangular period is 60°) |
| C gauge | carrier thickness `outer/dx = 2.5, 4, 6, 9, 12` | a Dirichlet value is a gauge, it must not set the jump |
| D readout | core radius `{0.3, 0.5, 0.7} R` × outer band `{[1.15,1.35], [1.25,1.45], [1.40,1.60], [1.55,1.75]} R` | the estimator must not create the number |
| E kernel support | `h/dx = 1.15, 1.30, 1.45, 1.60` | a discretisation knob, not a physical one |
| F resolution | `R/dx = 10, 15, 20, 30, 40, 60`, each on-lattice **and** at offset `(dx/2, dy/2)` | does the error decay, and does the alignment dependence decay with it? |

Implementation note (why the library changed): the shipped `hex_block` always put the
interface centre on a lattice site, so axes A and B were not expressible. Both knobs
are now parameters (`center=`, `angle=`); `angle = 0` keeps the historical lattice
bit-identically and the refactor is pinned by a regression test against the round-1
JSON (`test_round1_baseline_reproduced`, agreement to `1e-9`).

## ③ Results

### A — lattice translation (`R/dx = 20`)

| offset `(f_x dx, f_y dy)` | `dp_pressure` err | `dp_flux` err | `κR` | `∫δ dn` |
|---|---|---|---|---|
| (0, 0) — round-1 set-up | −0.21 % | −0.60 % | 1.004 | 0.993 |
| (1/3, 0) | −0.09 % | −0.50 % | 1.005 | 0.994 |
| (1/2, 0) | +0.11 % | −0.14 % | 1.006 | 0.998 |
| (0, 1/3) | −0.12 % | −0.64 % | 1.005 | 0.993 |
| (0, 1/2) | −0.11 % | −0.58 % | 1.004 | 0.993 |
| (1/4, 1/4) | −0.36 % | −0.77 % | 1.003 | 0.993 |
| (1/3, 1/3) | −0.35 % | −0.60 % | 1.003 | 0.993 |
| (1/2, 1/2) | −0.25 % | −0.65 % | 1.004 | 0.993 |
| (2/3, 0) | −0.21 % | −0.62 % | 1.004 | 0.994 |
| (0, 2/3) | −0.20 % | −0.64 % | 1.004 | 0.994 |
| (1/3, 1/2) | −0.21 % | −0.52 % | 1.003 | 0.994 |
| (1/2, 1/3) | −0.37 % | −0.78 % | 1.003 | 0.993 |

**Spread 0.47 %** (pressure), **0.63 %** (flux). The round-1 result is *not* a
positional accident: moving the interface anywhere inside a lattice cell moves the
jump by less than half a percent.

### B — lattice orientation

`0, 5, 10, 15, 20, 30°` → errors −0.21, −0.19, −0.16, −0.13, −0.16, −0.21 %.
**Spread 0.08 %**, flux spread 0.53 %. The capillary traction is orientation-isotropic
to well below the owner gate, as it must be for a circular interface.

### C — gauge distance (the largest sensitivity found)

| `outer/dx` | 2.5 | 4 | 6 | 9 | 12 (baseline) |
|---|---|---|---|---|---|
| `dp_pressure` err | **−3.16 %** | −0.54 % | −0.32 % | −0.11 % | −0.21 % |
| `dp_flux` err | +1.24 % | −0.74 % | −0.11 % | −0.58 % | −0.60 % |

Spread 3.04 %. A `2.5 dx` pad leaves the `p = 0` layer less than one particle spacing
outside the interface, so the gauge **truncates the `1/r` pressure field** and biases
the jump low by 3.2 %. From `4 dx` on the number is flat to 0.5 %: the operational
rule is **carrier pad ≥ 4 dx** (the pad and `n_particles` are reported per run, so the
rule is checkable). The flux instrument is insensitive to the pad (+1.24 % at
`2.5 dx`) — it never sees the gauge — and that asymmetry is the diagnosis, not a
disagreement.

### D — readout

12 combinations of core radius and outer band: **−0.19 … −0.24 %**, spread **0.05 %**;
the flux column is identical to the last digit (that route has no readout knob). The
estimator contributes nothing to the error budget.

### E — kernel support

| `h/dx` | 1.15 | 1.30 (baseline) | 1.45 | 1.60 |
|---|---|---|---|---|
| `dp_pressure` err | **−2.42 %** | −0.21 % | +1.75 % | +1.13 % |
| `dp_flux` err | −0.42 % | −0.60 % | −1.17 % | −1.08 % |

Spread **4.17 %** — the second largest. `h/dx = 1.15` is too narrow for the quadratic
curvature fit (too few neighbours in the stencil) and biases the jump low; `1.45`
overshoots high. The flux route stays inside 1.2 % over the same sweep, i.e. the `h`
dependence enters through the *curvature* branch of the chain, not the traction
integral.

### F — resolution ladder (two alignments)

| `R/dx` | N | on-lattice err | offset `(dx/2, dy/2)` err | flux err (on) | `κR` |
|---|---|---|---|---|---|
| 10 | 1203 | +0.97 % | +0.75 % | −0.48 % | 1.011 |
| 15 | 2716 | +0.06 % | +0.42 % | −0.49 % | 1.004 |
| 20 | 4773 | −0.21 % | −0.25 % | −0.60 % | 1.004 |
| 30 | 10656 | −0.34 % | −0.68 % | −0.56 % | 1.004 |
| 40 | 19018 | −0.93 % | −0.44 % | −0.85 % | 0.999 |
| 60 | 42735 | −0.60 % | −0.70 % | −0.71 % | 1.001 |

Observed order over `R/dx ∈ [15, 60]` (least squares on `log|err|` vs `log(R/dx)`):
**1.72** on-lattice, **0.48** at the offset alignment, **0.30** for the flux. The
error is already inside 1 % at every level, so these slopes fit a measurement floor
with a sign change (`κR` crosses 1 between `R/dx` 30 and 40); they are reported as
fitted, **not** as a demonstrated convergence order. What the ladder does establish is
that the alignment dependence does not grow with refinement (on/off-lattice
difference < 0.5 % at every level).

### The error budget, summarised

| knob | spread of the jump |
|---|---|
| readout (D) | 0.05 % |
| lattice orientation (B) | 0.08 % |
| lattice translation (A) | 0.47 % |
| gauge pad (C) | 3.04 % (3.16 % worst case at pad = 2.5 dx) |
| kernel support (E) | 4.17 % |
| **envelope over all 51 equilibria** | **max \|err\| = 3.16 %** (pressure), **1.24 %** (flux) |

## ④ Figure

`results/robustness.png` — (a) translation sweep with both instruments and the ±15 %
band; (b) gauge / `h/dx` / orientation on one log axis ("knobs the physics cannot
see"); (c) the `3 × 4` readout map with the error annotated per cell; (d) `|error|`
vs `R/dx` for the two alignments with the 15 % gate and the 3 % line.

## ⑤ Notes, limits, and what this does **not** say

* **The number to quote changes.** Round 1's −0.21 % is the value at the tuned set-up;
  the honest statement is **−0.2 % at `h/dx = 1.3` and pad ≥ 4 dx, ±3.2 % over the
  knob envelope**. Still far inside the 15 % gate, but the point value alone
  overstated the accuracy threefold.
* **Two knobs are not physical**: the carrier pad and `h/dx` are user choices. That
  they move the answer by 3–4 % while the *physical* invariances (A, B) hold to 0.5 %
  is the useful diagnostic: the residual error of this route is a discretisation-choice
  effect, not an anisotropy of the capillary force.
* **Still static, still equal-density.** As in round 1 the vapour carrier is filled
  with particles of the liquid density; this measures `sigma/R` only and is not a
  two-phase flow model. The dynamic question (does the projection keep the drop at
  rest?) remains open — `relax_projection` still diverges and is shipped as a
  *measurement of that defect*, not as a solver.
* **2-D only.** The 3-D counterpart (`CASE-HOLLOW-SPHERE-3D`, `2σ/R` on a sphere) is
  verified separately against the analytic law, but this error-budget audit has not
  been run in 3-D, and the `angle=` knob has no 3-D analogue here.
* **Friction (ecosystem), repeated from round 1**: PySPH is still not in the image, so
  the OSS arm cannot be re-run from a clean checkout. This round does not need it
  (both instruments belong to the same arm), but a future XVAL round does.
* Axes A/B probe invariance by moving/rotating the *interface relative to the lattice*;
  a Cartesian lattice variant is not exercised, and the triangular 60° period is why
  the rotation range stops at 30°.

## ⑥ Error statement (required)

Key quantity `dp` vs the analytic `sigma/R`:

* worst configuration of the whole audit: **3.16 %** (gauge pad `2.5 dx`) — **< 15 %** ✓
* **50 of 51** equilibria are **< 3 %**; the single exception is the deliberately
  truncated gauge pad (`2.5 dx`), an operational limit rather than a method error
  (`4 dx` and beyond: ≤ 0.54 %)
* physical invariances: translation 0.47 %, orientation 0.08 % — both **< 3 %** ✓
* second instrument (pressure-free traction flux): worst **1.24 %** over the same
  envelope — **< 3 %** ✓
* "is it < 3 %?": **yes** for the method at the operating point and for every physical
  invariance; **no** for a `2.5 dx` carrier pad, which the pad rule now excludes.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/xval_projection_laplace/robustness.py --quick   # 18 equilibria, 6 s
PYTHONPATH=src python3 scripts/xval_projection_laplace/robustness.py           # 51 equilibria, 26 s
PYTHONPATH=src python3 -m pytest tests/test_xval_projection_laplace_robustness.py -q
```