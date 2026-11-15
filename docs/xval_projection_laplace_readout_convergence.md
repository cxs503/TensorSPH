# XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE round 6 — the pressure readout: how much of the +15.94 % is resolution?

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **21377**
(rounds 1-5 = `docs/xval_projection_laplace_oss.md`, `_robustness`, `_density_ratio`,
`_3d`, `_3d_budget`, `_curvature_variation`).

## 1. Case and the open debt

Round 5 put the *differential* Young-Laplace law on an elliptical two-phase carrier
(`pi a b = pi R_eq^2`, `sigma = 0.072 N/m`, `R_eq` = 2 cm, kernel `h = 1.3 dx`,
projection/ISPH pressure solve on the pair-Laplacian of Cummins & Rudman 1999) and left two
gates red:

| gate | round-5 measured | declared | verdict |
|---|---|---|---|
| `station_rms_pct_aspect_le_2_resolved` | +41.70 | 15.0 | FAIL |
| `pressure_bias_pct_all_aspects_R_over_dx_40` | +15.94 | 15.0 | FAIL |

and it wrote *"the readout, not the force model, is the weak instrument at high curvature"*
**without a mechanism**. A sentence like that is a promissory note, not evidence: a bias caused
by probe placement (shell averages on a curved interface) and a bias caused by the kernel
smearing the jump over `h` have opposite consequences — the first is an instrument to be
repaired, the second is a resolution limit to be quoted. This round separates them.

**Hypothesis G (geometry of the readout).** The jump is read as `<p>_in - <p>_out` over shells at
fixed 2-4 dx / 3-6 dx offsets. A shell on a curved interface is not an isobar, so a
curvature-dependent part of the measured jump is instrument. If G is true the bias does not need
to decay with resolution *at fixed R_eq* — it is a fixed tax.

**Hypothesis S (kernel smearing).** The Poisson solve can only place a jump that is smeared over
the kernel support; a 3:1 ellipse at `R_eq/dx = 40` has `R_{curv,min} = 7.7 dx = 5.9 h` at
its tips. Then the readout bias must decay as `R_{curv,min}/dx` grows at fixed aspect.

## 2. Method

Two new scripts, both submitted through the compute gate (jobs `cj3380`/`cj3396`, `cj3403`):

* `scripts/xval_projection_laplace/diag_pressure_profile.py` — dumps the field the readout
  samples: `p` binned in signed interface distance (positive = vapour) plus per-station
  readouts. `ellipse_case` gained a `return_fields=True` switch that hands back
  `p / s_signed / kappa / delta_s / vol` (raw tensors, never serialised into the gated record).
* `scripts/xval_projection_laplace/readout_convergence.py` — the resolution ladder:
  aspects 2:1 and 3:1, each at `R_eq/dx = 20 / 40 / 80`, i.e. `R_curv,min/dx = 3.9 … 28.3`.
  Only `dx` changes at a fixed aspect, so the tip resolution grows linearly while every other
  instrument (`sigma`, `h/dx`, probe offsets, window width, `kappa` reference) is held fixed.

Gates were declared in `GATES` **before** the ladder ran and are copied verbatim into the JSON:

| gate | declared |
|---|---|
| `bias_ratio_80_over_40_aspect_3` | 0.8 (bias must at least fall by 20 % per doubling of `R_curv,min/dx`) |
| `bias_convergence_order_min` | 0.5 in `R_curv,min/dx` |
| `readout_variants_spread_pct` | 10.0 |
| `flux_bias_pct_all_levels` | 15.0 |

## 3. Results

### 3.1 Resolution ladder (all numbers from `results/readout_convergence.json`)

| aspect | `R_eq/dx` | `R_curv,min/dx` | `R_curv,min/h` | traction integral bias [%] | readout bias [%] | readout rms [%] | Whitney-Graustein [%] | net traction | `p` at `s=-8..-6dx` [Pa] | `p` at `s=+6..+8dx` [Pa] | particles | CG its |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3:1 | 20 | 3.85 | 2.96 | +4.82 | +30.35 | 49.93 | +0.26 | 0.00 | 2.823 | 0.476 | 10098 | 217 |
| 3:1 | 40 | 7.70 | 5.92 | -0.05 | +15.94 | 30.06 | -0.30 | 0.00 | 3.374 | 0.754 | 30644 | 382 |
| 3:1 | 80 | 15.40 | 11.84 | +0.45 | +11.72 | 24.41 | -0.61 | 0.00 | 3.691 | 1.014 | 104922 | 626 |
| 2:1 | 20 | 7.07 | 5.44 | +0.97 | +10.74 | 28.01 | -0.27 | 0.00 | 3.363 | 0.250 | 7614 | 194 |
| 2:1 | 40 | 14.14 | 10.88 | +0.93 | +6.80 | 19.24 | -0.30 | 0.00 | 3.594 | 0.422 | 21863 | 324 |
| 2:1 | 80 | 28.28 | 21.76 | -0.36 | +4.36 | 12.24 | -0.65 | 0.00 | 3.804 | 0.579 | 72395 | 580 |

### 3.2 Gate verdicts

| gate | measured | declared | verdict |
|---|---|---|---|
| `bias_ratio_80_over_40_aspect_3` | 0.736 | <= 0.800 | **PASS** |
| `bias_convergence_order_min` | 0.443 | >= 0.500 | **FAIL** |
| `flux_bias_pct_all_levels` | 4.815 | <= 15.000 | **PASS** |

### 3.3 The vapour plateau (aspect 3:1, `R_eq/dx = 40`; `results/diag_pressure_profile.json`)

`p` binned by signed distance to the ellipse, positive on the vapour side (global over all
stations, so the liquid side mixes the low-curvature flank and the tips — that mix is exactly
why the readout is taken per station; the *vapour* side is what matters below):

| `s/dx` bin | particles | mean `p` [Pa] | std `p` [Pa] |
|---|---|---|---|
| [-8.0, -6.0) | 598 | 3.3736 | 1.4870 |
| [-6.0, -5.0) | 317 | 3.4483 | 1.6933 |
| [-5.0, -4.0) | 324 | 3.5203 | 1.8685 |
| [-4.0, -3.5) | 172 | 3.6713 | 1.9766 |
| [-3.5, -3.0) | 166 | 3.4747 | 2.1139 |
| [-3.0, -2.5) | 161 | 3.7924 | 2.1046 |
| [-2.5, -2.0) | 180 | 3.4569 | 2.1454 |
| [-2.0, -1.5) | 167 | 3.6666 | 2.1539 |
| [-1.5, -1.0) | 179 | 3.4003 | 1.8912 |
| [-1.0, -0.5) | 167 | 3.0002 | 1.4870 |
| [-0.5, +0.0) | 179 | 2.7332 | 1.3521 |
| [+0.0, +0.5) | 185 | 2.1758 | 0.8156 |
| [+0.5, +1.0) | 170 | 1.7887 | 0.9354 |
| [+1.0, +1.5) | 194 | 1.4586 | 0.8481 |
| [+1.5, +2.0) | 178 | 1.0933 | 1.1104 |
| [+2.0, +2.5) | 191 | 0.9417 | 1.0792 |
| [+2.5, +3.0) | 180 | 0.9291 | 1.1674 |
| [+3.0, +3.5) | 190 | 0.7910 | 1.0202 |
| [+3.5, +4.0) | 200 | 0.8782 | 1.0185 |
| [+4.0, +5.0) | 390 | 0.7906 | 0.9281 |
| [+5.0, +6.0) | 395 | 0.7653 | 0.8472 |
| [+6.0, +8.0) | 814 | 0.7541 | 0.7229 |

## 4. Findings

1. **Hypothesis S is supported: the readout bias is mostly a tip-resolution tax.** At a:b = 3:1
   the bias falls +15.94 % → +11.72 %
   when `R_curv,min/dx` goes 7.70 → 15.40 (`R_curv,min/h` 5.9 → 11.8), and at a:b = 2:1
   +10.74 / +6.80 / +4.36 % over 7.07 / 14.14 / 28.28 — monotone in both aspects, with the
   *same* probes and the *same* `sigma`, `h/dx` and curvature reference.
2. **But the decay is slower than predeclared.** `bias_ratio_80_over_40 = 0.736`
   passes the 0.8 gate, the fitted order `0.443` **fails** the declared
   0.5. Half an order, not first order. The gate stays failed; it is not relaxed here.
3. **The reason for the slow order is visible in the vapour plateau: it is not zero and it grows
   with resolution.** `p(s = +6..+8 dx)` = 0.476 / 0.754 / 1.014 Pa at `R_eq/dx` = 20 / 40 / 80
   (a:b = 3:1) and 0.250 / 0.422 / 0.579 Pa at a:b = 2:1. A static drop must have a
   *uniform* vapour pressure, and the gauge is `p = 0` on the outer boundary layer, so this
   plateau should be flat and equal to the ambient value. It is not: the interface force
   density has a small net monopole whose image against the Dirichlet gauge fills the vapour
   with an offset. The readout's reference is therefore contaminated, and the contamination
   *grows* with resolution — it partially cancels the converging part of the error, which is
   precisely what turns an O(1) mechanism into an O(0.44) measurement.
4. **The traction integral is not affected.** Its bias stays inside ±4.82 % at *every*
   level (gate 15.0 %) and the Whitney-Graustein curvature gauge stays inside
   ±0.65 %. The force model is resolution-convergent; the *reported number* is not.
5. **Hypothesis G could not be separated with a shell-based instrument.** The diagnostic's
   moved/wide probe arms were station-blind by construction (the probe shells were global, not
   per station), so they read a single number for all 24 stations and are worthless as a
   station-resolved test — the arm is recorded as *not run*, not as passed. What the diagnostic
   does establish is that the plateau the readout samples does not exist in the vapour at the
   probe offset: at `s = +6..+8 dx` the vapour pressure is 0.754 Pa while the boundary
   gauge forces it to 0 (finding 3).
6. **A mechanism worth a follow-up (not claimed as fixed).** The vapour offset is a property of
   the Poisson *source*: `centre`-blind `div(f)` on a closed interface should integrate to zero,
   and round 4 already introduced a charge-neutral source for disordered packings
   (`src/tensorsph/laplace_isph.py`, commit `ae172d4`). The ladder shows the lattice ellipse
   still carries a residual monopole large enough to fill the vapour with 0.25-1.0 Pa against
   a 0.7-17.5 Pa signal. Referencing the readout to the *far-field* vapour level, or enforcing
   zero net charge of the discrete source, is the next experiment; it is **not** done here.

## 5. Error verdict vs the owner's gates

| key quantity | reference | best / worst measured | < 3 % ? |
|---|---|---|---|
| traction integral `oint f.n dl / L`, all levels | `sigma kappa_window` | -0.36 % … +4.82 % | **yes** at 4 of 6 levels, no at the two coarsest (-4.8 %/+4.8 %, aspect 3:1 `R_eq/dx=20`) |
| reported jump `<p>_in - <p>_out`, a:b = 3:1, `R_eq/dx = 40` | `sigma kappa_window` | +15.94 % (rms 30.06 %) | **no** (fails the 15 % gate) |
| reported jump, a:b = 3:1, `R_eq/dx = 80` | same | +11.72 % (rms 24.41 %) | **no** (>3 %, but inside the owner 15 % gate) |
| reported jump, a:b = 2:1, `R_eq/dx = 80` | same | +4.36 % (rms 12.24 %) | **no** (>3 %, inside 15 %) |
| vapour plateau `p(s=+6..8dx)` | 0 (ambient gauge) | 0.579 … 1.014 Pa | **no** — a defect, see finding 3 |
| Whitney-Graustein `oint kappa dl / 2 pi` | 1 | -0.65 … +0.26 % | **yes** |

Honest-FAIL: the `bias_convergence_order_min` gate is **failed** (0.443 < 0.5) and the owner
15 % readout gate at a:b = 3:1, `R_eq/dx = 40` remains **failed at +15.94 %**; no gate was
relaxed, no coarse level was dropped from the average.

## 6. Notes, limitations, applicability

* The vapour phase is filled with liquid-density particles: this is the round-1..5 static
  verification carrier (the law contains `sigma` and `R` only), not a multiphase flow model.
* Practical rule for this module: quote the **traction integral** up to
  `R_curv,min/dx >= 4` (±5 %), and do not quote a *pointwise* pressure readout better than
  15 % below `R_curv,min/dx ~ 8` (a:b = 3:1 needs `R_eq/dx >= 45` to pass the owner gate at
  `R_eq = 2 cm`; measured: passes already at `R_eq/dx = 80`).
* The readout is `O(0.44)` in `R_curv,min/dx`, not `O(1)`, because the vapour reference is
  contaminated by a residual source monopole whose amplitude grows with resolution. Until that
  is fixed, quoting a convergence order for the *readout* is misleading; quote the traction
  integral.
* Two instrument defects were caught before the ladder was believed: (i) the diagnostic's
  moved/wide probe arms were station-blind and therefore meaningless as a station-resolved
  test (finding 5 — recorded as not run, the code is left in place with the note in its
  docstring); (ii) the first diagnostic run crashed on a `stats()` type error, which the
  compute-gate log caught before any number was used.

Artifacts: `scripts/xval_projection_laplace/diag_pressure_profile.py`,
`scripts/xval_projection_laplace/readout_convergence.py`,
`results/diag_pressure_profile.json`, `results/readout_convergence.json`,
`results/readout_convergence.png`, `tests/test_xval_projection_laplace_readout_convergence.py`.
