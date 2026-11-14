# Round 3 — the projection-SPH Young-Laplace jump with a real density ratio

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m20557**
· round 1 = `docs/xval_projection_laplace_oss.md` (m17594), round 2 =
`docs/xval_projection_laplace_robustness.md` (m18632), 3-D extension = `77c9c4f`.

Rounds 1-2 established that `dp = sigma/R` is measured to −0.21 % and that the
number is a property of the *method*, not of the lattice, the container or the
readout band — **all on a 1:1 density carrier**. The vapour was filled with
liquid-density particles, stated as deliberate because "the law contains `sigma`
and `R` only". That is the correct *reference*, but it left the *numerics*
untested: the projection step solves a pressure Poisson equation whose
coefficient is `1/rho`, so every published number for this module was measured
with the one term the density ratio multiplies switched off by construction.

Round 3 turns it on. `rho_liquid/rho_vapour` is swept from 1 to 1000; the
lattice, the cell volume `V` and therefore the colour function are held fixed
(`m = rho_phase V` ⇒ `vol = m/rho_phase` unchanged), so the ratio is the only
knob that moves.

## Artifacts

| artifact | what |
|---|---|
| `scripts/xval_projection_laplace/density_ratio.py` | the sweep + figure generator (gates declared in `GATES`) |
| `scripts/xval_projection_laplace/results/density_ratio.json` | every number below (40 equilibria) |
| `scripts/xval_projection_laplace/results/density_ratio.png` | fig. 1 (4 panels) |
| `tests/test_xval_projection_laplace_density_ratio.py` | 11 gates + 1 strict xfail (the measured defect) |
| `src/tensorsph/laplace_isph.py` | `rho_vapour=`, `poisson_weighting=`; `rho=` operator in `pressure_poisson` |

## ① Case and parameters

| quantity | value |
|---|---|
| interface | circle, `R = 0.02 m` (principal curvatures `1/R`, `0`) |
| surface tension | `sigma = 0.072 N/m` (water/air, 20 °C) |
| densities | `rho_l = 1000 kg/m³`, `rho_v = 1000/ratio` (ratio 1…1000) |
| reference | `dp = sigma/R = 3.6000 Pa` — **independent of the ratio** |
| baseline lattice | triangular, `dx = 1 mm` (`R/dx = 20`), `h = 1.3 dx`, N = 4773, 1452 liquid |
| gauge | Dirichlet `p = 0` on the outer 1.6 dx layer of a block `L = R + 12 dx` |
| scheme | Shepard colour (volume-weighted) → MLS gradient → smoothed normal → `div n` curvature → Brackbill `|grad c|` → CSF traction → Cummins & Rudman pair Laplacian, matrix-free CG (`rel_tol 1e-10`) |

Two arms and one ablation:

* **`density`** — `-div((1/rho) grad p) = -div(a)`, pair coefficient `1/rho_ij`
  with the harmonic pair mean `rho_ij = 2 rho_i rho_j/(rho_i+rho_j)`. This is the
  projection step at `u = 0`, where `grad p = f` is a **force-density** balance
  and therefore contains no `rho`.
* **`uniform`** — constant-coefficient `-grad²p = -rho_l div(a)`, the naive
  implementation that treats the ratio as a source correction.
* **`diffuse`** — density carried on the smoothed colour field,
  `rho_i = rho_v + (rho_l-rho_v) c_i` (diffuse interface). Tested as a candidate
  fix for the resolution drift; **falsified**.

Gates were fixed before the sweep and are stored in the JSON: 3 % band on the
validated arm (`R/dx = 20`), 3 % ratio spread, 1 % flux spread,
5 % resolution band for `R/dx >= 20`, 5 % translation spread, the naive arm
**must** exceed the 15 % owner gate, `sigma = 0` must return `dp = 0`.

## ② Results — the jump is density-ratio blind, if the operator is

### A — ratio ladder, `R/dx = 20`

| `rho_l/rho_v` | `dp` [Pa] | density-weighted err | naive (uniform-coefficient) err | `∫δ dn` | `κR` |
|---|---|---|---|---|---|
| 1 | 3.5923 | **−0.21 %** | −0.21 % (identical by construction) | 0.9932 | 1.0035 |
| 2 | 3.5554 | −1.24 % | +46.7 % | 0.9932 | 1.0035 |
| 5 | 3.5292 | −1.97 % | +187 % | 0.9932 | 1.0035 |
| 10 | 3.5270 | −2.03 % | +422 % | 0.9932 | 1.0035 |
| 30 | 3.5466 | −1.48 % | +1 360 % | 0.9932 | 1.0035 |
| 100 | 3.5809 | −0.53 % | +4 642 % | 0.9932 | 1.0035 |
| 300 | 3.6101 | +0.28 % | +14 021 % | 0.9932 | 1.0035 |
| 1000 | 3.6298 | **+0.83 %** | **+46 846 %** | 0.9932 | 1.0035 |

* Validated arm: max |err| = **2.03 %**, ratio spread **2.85 %** — both inside
  the 3 % band. The interface geometry is bit-identical in every row
  (`∫δ dn = 0.9932`, `κR = 1.0035`), so this is the density ratio acting alone.
* The naive operator does not merely degrade: at 1000:1 it over-predicts the jump
  by a factor 470. The coefficient belongs *inside* the divergence; a "source
  correction" is not an option.
* Independent instrument: `dp_flux = -∫ rho a_r dA / 2πR` — with `a = f/rho` and
  `m = rho V` the density cancels analytically, and it does numerically too:
  the spread over the whole ladder is **3.3e-14 %** (max |err| 0.60 %). The two
  instruments agree, so the pressure field is not inventing the number.

### B — resolution, and the one gate that failed

| `R/dx` | ratio 1 | ratio 10 | ratio 1000 |
|---|---|---|---|
| 10 | +0.97 % | −0.19 % | +2.67 % |
| 15 | +0.06 % | −1.35 % | +1.67 % |
| 20 | −0.21 % | −2.03 % | +0.83 % |
| 30 | −0.34 % | −5.26 % | **−6.36 %** |
| 40 | −0.93 % | −8.32 % | **−12.73 %** |

The ratio-1 arm is flat to <1 % (as in rounds 1-2). With a real ratio the jump
deficit **grows under refinement** and is systematic (it is not lattice
alignment: at `R/dx = 30` the four sub-cell offsets give −6.36 / −4.17 / −7.69 /
−5.48 %). The traction-flux instrument stays inside 0.9 % at every level, so the
loss is in the Poisson solve, not in the traction. **Declared gate
`|err| < 5 % for R/dx >= 20` fails** and is registered as a strict `xfail`
(`test_resolution_band_at_high_contrast`), not relaxed. The owner gate (15 %)
still holds in every configuration measured.

### C — why the operator choice matters (`R/dx = 30`, ratio 1000)

(shipped row is in the JSON; the rejected variants below were measured in a
one-off probe of the same case and are *not* stored as artifacts — they are
listed so the negative results are on the record)


| source/operator variant | `dp` err |
|---|---|
| `-div((1/rho)grad p) = -div(a)`, harmonic `rho_ij` (shipped) | **−6.36 %** |
| source from a Shepard-smoothed acceleration | +781 % |
| source `-div(f)/rho_i` (force-density divergence) | −300 % |
| diffuse density `rho(c)`, same operator | −30.9 % (and −36.6 % at `R/dx = 40`) |
| CG tolerance 1e-10 → 1e-12 | −6.36 % (unchanged: not a convergence artefact) |

### D — controls

* `sigma = 0` at ratio 1000: `dp = 0.0 Pa` and `dp_flux = 0.0 Pa` (exact).
* Translation invariance at ratio 1000, `R/dx = 20`: spread **0.82 %**,
  max |err| 1.04 %.
* Per-phase cell volume spread: **0.0** (mass ratio 1000.0 exactly) — the colour
  function, and hence the interface, is unchanged by the ratio.

## ③ Notes, limits, and what to fix next

* **The physical claim survives; the discretisation does not converge yet.** The
  Young-Laplace law is ratio-blind and the density-weighted pair-Laplacian
  reproduces it to 2 % at `R/dx = 20` for ratios up to 1000, but at finer meshes
  the jump deficit grows to −12.7 % (`R/dx = 40`). A converged two-phase ISPH
  pressure solve (e.g. a consistent, summation-by-parts divergence for `a` on the
  same graph as the Laplacian, or the Adami/Hu-Adams density-weighted divergence
  with the density *also* advancing with the phase field) is the next step; the
  diffuse-density shortcut was measured and rejected.
* **This is not a multiphase flow model.** It is the static equilibrium of the
  projection step on a filled carrier with a real density ratio. The vapour is a
  particle carrier at `rho_v`, not a simulated gas; dynamic relaxation
  (`relax_projection`) is still the documented unstable path from round 1.
* **Interface thickness vs density step.** The sharp per-particle density step
  sits inside the ~`2h` traction shell. That was the suspected cause of the
  drift; the diffuse-density ablation falsifies it (three times worse), so the
  bias comes from the discretisation of `div(a)` against the stiff `1/rho` more
  than from the placement of the density jump.
* The equal-density path is bit-identical to rounds 1-2 (pinned by
  `test_equal_density_path_bit_identical`, −0.2133 % at `R/dx = 20`), so no
  earlier number was silently re-measured.

## ④ Error verdict

| quantity | reference | best measured | gate |
|---|---|---|---|
| `dp`, ratio 1 (regression) | `sigma/R = 3.6 Pa` | −0.21 % | <3 % ✓ |
| `dp`, ratio 1000, `R/dx = 20` | 3.6 Pa | **+0.83 %** | <3 % ✓ |
| `dp`, all ratios 1…1000, `R/dx = 20` | 3.6 Pa | max **2.03 %** | <3 % ✓ |
| `dp`, ratio 1000, `R/dx = 30/40` | 3.6 Pa | −6.36 % / −12.73 % | <5 % ✗ (**honest FAIL**, xfail) |
| traction flux, all rows | `σ/R` | max 0.60 %, spread 3e-14 % | <1 % ✓ |
| naive operator, ratio 1000 | 3.6 Pa | +46 846 % | must fail >15 % ✓ (it fails) |
| `sigma = 0` control | 0 Pa | 0.0 Pa | ✓ |

**Is the key quantity within 3 %?** Yes for the validated configuration
(`R/dx = 20`, ratios 1…1000: max 2.03 %) and for the equal-density regression
(−0.21 %). **No** for the high-contrast refined meshes: there the error is
6.4-12.7 %, inside the owner's 15 % acceptance but outside the 3 % band, and it
is recorded as a live defect rather than described as convergence.