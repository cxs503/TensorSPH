# Young–Laplace verification of the CSF surface-tension model

**Verdict up front:** the CSF model merged in m277 (`src/tensorsph/surface_tension.py`)
**does not reproduce the Young–Laplace law** and does not converge with
resolution. It produces an inward-pointing force of roughly the right order,
which is what the benchmark it shipped with checked, but it cannot sustain the
pressure jump `Δp = σ/R` that defines a capillary model. Evidence grade for
capillary physics: **MOCK** — geometrically plausible output, no quantitative
agreement with the analytic law. The model is off by default
(`Config.surface_tension_coefficient = 0.0`), so no other published number in
this repository is affected.

Harness: `benchmarks/laplace/run.py` (probes P1–P4, writes
`benchmarks/laplace/report.json` and `laplace_verification.png`).
Gates: `tests/test_laplace_verification.py`. Root-cause experiment:
`benchmarks/laplace/halo_experiment.py`.

---

## 1. Case

A 2-D disc of weakly-compressible liquid of radius `R` in vacuum, no gravity,
one fluid phase, surface-tension coefficient `σ`. In two dimensions the exact
capillary pressure jump across a circular interface of curvature `κ = 1/R` is

    Δp = σ κ = σ / R.

| parameter | value |
|---|---|
| `R` | 0.05 m |
| `σ` | 0.072 N/m (water/air) |
| `ρ₀` | 1000 kg/m³ |
| `c₀` | 20 m/s (sweep includes 5 m/s) |
| `γ` (Tait) | 7 |
| kernel | 2-D cubic spline, `h = 1.3 Δx` |
| discretisation | square lattice, offset 0.37 Δx (no particle at `r = 0`) |
| resolution ladder | `R/Δx = 10, 20, 40` (318 / 1262 / 5032 particles) |
| theory | `Δp = σ/R = 1.4400 Pa`, `κR = 1` |

The disc is a genuine test of the *force model*, because the Laplace pressure is
the only steady state a capillary force field can have.

## 2. Method

Everything is measured on the production code path, at `float64`, on CPU, and
is deterministic (no stochastic step anywhere in the solver).

* **P1 colour-jump integral.** The CSF surface delta function is
  `δ_s = |∇c|`. For a phase indicator that goes from 1 (liquid) to 0 (vacuum),
  `∫ |∇c| dr` across the interface must equal **1**: that integral is the
  discrete statement that the surface force carries the right amount of
  interfacial area. Measured by angular averaging `|∇c|` into radial shells and
  summing `|∇c|·Δr`.
* **P2 curvature.** `κ` from `compute_curvature` (divergence of the normal
  field), averaged over the interface band (`r > 0.8 R`, surface-flagged
  particles), reported as `κR` against the exact value 1.
* **P3 discrete Young–Laplace.** The radial acceleration returned by
  `compute_surface_tension_force` is integrated inward,
  `Δp = −∫ ρ₀ a_r dr`, i.e. the pressure rise the force field *asks* the
  equation of state to supply. This is the Young–Laplace law as the force field
  itself sees it, independent of the time integrator.
* **P4 relaxation.** The production RHS (`droplet_impact.rhs`, which adds the
  CSF acceleration) is advanced 0.05 s with gravity off and a step limited by
  both acoustic Courant and the capillary limit; interior pressure (`r < 0.5 R`)
  and peak spurious current are reported. Two sound speeds are run because the
  interior pressure must not depend on `c₀` — if it does, the reading is
  discretisation noise, not capillary pressure.

Acceptance criteria, fixed before the run: `|∫|∇c|dr − 1| ≤ 5%`,
`|κR − 1| ≤ 15%`, `|Δp/(σ/R) − 1| ≤ 15%` for both P3 and P4. 15% is this
repository's "unacceptable beyond" line; 3% is the strict line asked for in the
report.

## 3. Results

### 3.1 Static probes

| `R/Δx` | N | `∫\|∇c\|dr` | err | `κR` | err | `Δp` (P3) [Pa] | err |
|---|---|---|---|---|---|---|---|
| 10 | 318  | 0.407 | −59.3% | −0.135 | −113.5% | +1.8263 | **+26.8%** |
| 20 | 1262 | 0.306 | −69.4% | −0.140 | −114.0% | +1.3838 | −3.9% |
| 40 | 5032 | 0.382 | −61.8% | +0.205 | −79.5% | +5.7039 | **+296.1%** |
| exact | — | 1 | — | 1 | — | 1.4400 | — |

### 3.2 Relaxation (P4, `R/Δx = 20`)

| `c₀` [m/s] | bulk `B=ρ₀c₀²/γ` [Pa] | density excess needed `Δp/(γB)` | `ρ/ρ₀` reached | `p` interior [Pa] | err | `\|v\|max` [m/s] |
|---|---|---|---|---|---|---|
| 20 | 57 143 | 3.6e−6 | 1.000010 | +3.8035 | **+164%** | 7.40e−3 |
| 5  | 3 571  | 5.8e−5 | 1.000148 | +3.6925 | **+156%** | 7.35e−3 |

Aspect ratio and circularity of the droplet are preserved (the shape is not the
problem); the *pressure* is.

## 4. Figure

`benchmarks/laplace/laplace_verification.png` — left: `κR` against `r/R` for the
three resolutions with the exact value 1; middle: integrated `Δp` against
`σ/R` on a log resolution axis; right: relative error bar chart of every probe
against the ±15% band.

![Young–Laplace verification](laplace_verification.png)

## 5. Why it fails (root cause, measured)

Three distinct defects, in order of how much they matter:

1. **No exterior phase — the colour field never jumps.** The colour function is
   built from a neighbour count (`c = n_neighbours/mean` on "surface" particles,
   `c = 1` elsewhere). There are no particles outside the liquid, so `c` has no
   zero to run to: measured floor is ≈ 0.63, its gradient integrates to
   **0.28–0.41** instead of 1, and it moves by only ±0.05 when the resolution is
   doubled. A surface delta function that is 3× too small in this sense makes
   the model's force magnitude arbitrary.
   `artifacts`-free reproduction: `benchmarks/laplace/halo_experiment.py` adds a
   three-layer vacuum halo with `c = 0` and recomputes the same gradient; the
   integral then reaches **1.04 at `R/Δx = 20`** (from 0.306) — which confirms
   this defect and nothing else.
2. **Curvature from `∇·n` on a discontinuous normal field.** With the halo in
   place `∫|∇c|dr` is fixed but `κR` becomes −10.8 at `R/Δx = 20` (values from
   −15.5 to +8.6 across the band): the normal is clipped to zero inside the
   liquid and in the far halo, so `∇·n` integrates the *edges of the band*, not
   the curvature. `κ` needs a normal field that is defined (and smooth)
   throughout the band, plus an MLS/Shepard renormalisation; the merged code has
   neither. This is why `κR` is −0.14 with the wrong sign at low resolution.
3. **The equation of state cannot resolve the pressure.** Tait with `c₀ = 20 m/s`
   has `B = 57 kPa`, so `Δp = 1.44 Pa` requires a density excess of
   `Δp/(γB) = 3.6e−6`. Lattice density summation has a bias of order `1e−3` —
   two to three orders larger. The interior therefore equilibrates wherever the
   summation bias leaves it (measured `ρ/ρ₀ = 1.000010` at `c₀ = 20`), and the
   "measured" `Δp` is bias-dominated: it changes only from +164% to +156% when
   `c₀` is cut fourfold, though the required compression rises 16×. Any
   verification of a capillary model at this scale must either lower `c₀` to
   ≈ 3 m/s *and* remove the summation bias (Shepard/MLS renormalisation) or
   scale `σ` up so `Δp/B ≳ 1e−3`.

Contributing defect, worth stating because it misled the merged benchmark:
`benchmarks/static_droplet/run.py` writes `"status": "PASS"` on the criteria
"force points inward" and "force magnitude is non-zero", and its `criteria`
list claims `dot product > 0.5` while the number it actually reported was an
average dot product of 0.17. That benchmark has been corrected to state the
checks it really performs and to record `laplace_law_verified: false`.

## 6. Error judgement against the >= 3% / >= 15% standard

| quantity | reference | measured | error | < 3%? | < 15%? |
|---|---|---|---|---|---|
| `∫\|∇c\|dr` (P1, R/Δx=20) | 1 | 0.306 | −69.4% | **no** | **no** |
| `κR` (P2, R/Δx=20) | 1 | −0.140 | −114.0% | **no** | **no** |
| `Δp = −∫ρ a_r dr` (P3, R/Δx=20) | 1.4400 Pa | 1.3838 Pa | −3.9% | **no** | yes |
| `Δp` (P3, R/Δx=10) | 1.4400 Pa | 1.8263 Pa | +26.8% | **no** | **no** |
| `Δp` (P3, R/Δx=40) | 1.4400 Pa | 5.7039 Pa | +296.1% | **no** | **no** |
| `p` interior, relaxation `c₀=20` | 1.4400 Pa | 3.8035 Pa | +164.1% | **no** | **no** |
| `p` interior, relaxation `c₀=5` | 1.4400 Pa | 3.6925 Pa | +156.4% | **no** | **no** |
| inward fraction of force | 1 (by construction) | 0.734 | — | n/a | n/a |
| momentum residual `\|Σm a\|/Σm\|a\|` | 0 | 8.6e−3 | — | n/a | n/a |

**Is the key quantity below 3%? No.** Below 15%? **No**, not defensibly: the one
entry that falls inside 15% (P3 at `R/Δx = 20`, −3.9%) is a coincidence of two
large cancelling errors, proved by its neighbours at `R/Δx = 10` (+26.8%) and
`R/Δx = 40` (+296.1%). A model whose error moves 300% under refinement has no
usable accuracy to state; the honest number is "not applicable / not converged".

## 7. Scope and consequences

* Applies to `Config.surface_tension_coefficient > 0`. That is **not** the
  default, so the droplet-impact and dam-break results on the same commit are
  untouched.
* The capillary time step `capillary_dt_limit` is analytically correct
  (`0.25√(ρh³/σ)`) and is unaffected; only the force field and its curvature
  are wrong.
* The two properties the model *does* have are pinned as ordinary tests:
  predominantly radial, inward acceleration, and a near-zero net internal
  momentum (8.6e−3 of the force scale). Fixing the model must keep both.
* The three failing statements are pinned as `xfail(strict=True)` gates, so a
  fix that quietly changes behaviour will turn them into XPASS failures and
  force this document to be updated with new numbers.

## 8. How to fix it (acceptance criteria for the follow-up)

1. Colour field with an exterior phase (vacuum halo particles or a ghost
   extension) so that `∫|∇c|dr = 1 ± 5%` at `R/Δx ∈ {10, 20, 40}`.
2. Normal field defined and smooth over the interface band (MLS/Shepard
   renormalised); `κR = 1 ± 15%`, and the error must *decrease* along the
   resolution ladder.
3. `|Δp/(σ/R) − 1| ≤ 15%` for P3 at two resolutions and for P4 after relaxation,
   with the relaxation value independent of `c₀` within 5%.
4. Report spurious currents: `|v|max / √(σ/(ρR))` (measured baseline 7.4e−3 m/s
   against 1.2 m/s capillary scale at these parameters), and the equilibrium
   Laplace pressure must not drift over 0.1 s.
5. Suite stays green: the two positive tests here plus the existing
   `tests/test_surface_tension.py`; the three strict xfails flip to XPASS and
   must be rewritten with the new numbers.

Reproduce everything with:

```bash
python benchmarks/laplace/run.py                # ~2 min on CPU, writes report.json + png
python benchmarks/laplace/halo_experiment.py    # root-cause experiment
pytest tests/test_laplace_verification.py -q    # 2 passed, 3 xfailed (gap open)
```