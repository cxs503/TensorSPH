# Referee report: which CSF surface-tension fix actually verifies Young-Laplace?

**Repository** TensorSPH · **Mission** m2495 (self-start, models seat) · **Branch** agent/m2495
**Scope** independent scoring only: no `src/` file is modified by this mission.

## 1. Why this report exists

The merged tree cannot run surface tension at all, and seven unmerged branches each
claim to repair it. Two of them (m2447, m2454) claim the Young-Laplace gate is met.
Their claims are scored here by one instrument, written for this mission, that touches
only the public API, so that "what the user gets" is what is measured.

The claim that needed checking: m2454 reports the Laplace defect repaired to
12.0 % worst case. Its estimator is
`Δp = Σ ρ(-a·n̂)V / (2πr_c)`, an integral of the *surface-tension force*. Analytically
that integral equals `σκ`, so it re-measures the force normalisation rather than the
pressure the flow solver sustains. The physical statement of the Young-Laplace law is
about the EOS pressure field, and that is estimator B2 below. Both are computed here on
the very same relaxed state.

## 2. Instrument

`benchmarks/csf_referee/score.py` (2-D hexagonal drop, `R = 0.02 m`, `σ = 7.2 N/m`,
`ρ0 = 1000`, `c0 = 20 m/s`, Tait `γ = 7`, `ν = 0.01 m²/s`, `h = 1.3 dx`), three
quantities:

* **A. cut-force identity** `F_cut = Σ_{x>0} m a_x`, exact value `-2σκR`; the ratio to
  `-2σ` therefore *is* `κ/(1/R)` provided `∫δ_s dn = 1`.
* **B1. force-integral jump** (`Δp_force`) — the candidates' estimator, quoted for
  comparison only.
* **B2. pressure-field jump** `p(r<0.5R) − p(0.8R<r<0.95R)` from the Tait EOS on the
  summation density, plus the same average over the last 25 % of the run
  (`Δp_timeavg`) to separate bias from noise.
* **C. spurious current** `max|v|/√(σ/ρR)` (Adami et al. 2012).

Each candidate is scored by materialising a shadow package: current `src/tensorsph`
overlaid with exactly the `src/tensorsph/*.py` files that branch changed
(`benchmarks/csf_referee/materialize.sh`). `master` itself cannot be imported (see §4),
so its physics is scored as `master_p0` = master + the two one-line shims that make the
merged segment runnable; that baseline is at the same footing as the candidates.

Ladder: `dx = 3.0, 2.0, 1.5 mm` (N = 151…585) for A, `dx = 2 mm`, 900 steps
(`t = 1.05 t_σ`) for B/C.

## 3. Results

### 3.1 Headline table (all numbers from `benchmarks/csf_referee/summary.json`)

| candidate | κ err (best dx) | F_cut/2σ (dx=2 mm) | cos(n,r̂) med | Δp **pressure** | Δp time-avg | Δp **force integr.** | spurious |
|---|---|---|---|---|---|---|---|
| master (+P0 import shim) | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2389 | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2399 | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2420 | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2433 | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2451 | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2459 | -47.8% | +0.522 | +0.992 | -95.5% | -87.3% | -84.3% | 0.190 |
| m2447 | +7.4% | +1.077 | +1.000 | -87.8% | -96.6% | -37.0% | 0.037 |
| m2454 | +8.0% | +1.111 | +1.000 | -46.6% | -38.0% | +15.0% | 0.022 |

Read: **κ err** is `κ_eff/(1/R) − 1` with `κ_eff = |F_cut|/(2σR)`, best of the three
`dx`; **Δp pressure** and **Δp force integr.** are the two estimators of the same
`σ/R = 360 Pa`; the 15 % platform gate is `|err| < 15 %`.

### 3.2 The two estimators disagree by up to 100 percentage points

For m2454 the force integral reads **+15.0 %** (i.e. "gate met") while the
pressure field of the same state reads **-46.6 %**. Averaging the pressure over the last quarter of the run
does not rescue it (**-38.0 %**): the residual is a *bias*, not a noise floor, so no amount of
sampling fixes it. Reporting the force integral as "the Laplace law verified" is
therefore not supported by this instrument.

### 3.3 What *is* repaired

* Direction: every candidate puts the force inward (`cos(n, r̂) ≈ +1`), so the m2174
  finding of 51–57 % inward particles is gone.
* Magnitude: m2447 holds `F_cut/2σ = +1.077` at `dx = 2 mm`, m2454
  `+1.111`; the merged baseline is `+0.522` (≈ 2× too weak).
  On the cut-force criterion alone m2447 is inside the 15 % gate at every `dx`, and m2454
  is inside it at `dx = 2–3 mm` but drifts to +18.7 % at `dx = 1.5 mm`.
* Spurious current: m2454 `0.022`, m2447 `0.037`,
  baseline `0.190` — both candidates are far quieter.
* Five branches (m2389, m2399, m2420, m2433, m2451, m2459) reproduce the baseline column
  bit for bit: their `src` change is the contact-angle boundary / packaging only and does
  not touch the physics measured here. They are not physics fixes.

### 3.4 Why the pressure field cannot follow

| candidate | ρ_center/ρ0 | ρ_band/ρ0 | Δρ/ρ needed by law | pressure noise (band σ) [Pa] |
|---|---|---|---|---|
| master (+P0 import shim) | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2389 | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2399 | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2420 | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2433 | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2451 | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2459 | 0.999998 | 0.888078 | 0.000898 | 21.3 |
| m2447 | 1.000751 | 0.930968 | 0.000898 | 249.4 |
| m2454 | 1.001087 | 0.937160 | 0.000898 | 247.4 |

The law needs `Δρ/ρ = 0.000898` between centre and interface; the summation density
differs from `ρ0` by `1 − ρ_band/ρ0 = 0.1119` at the interface *before* any surface
tension acts, i.e. a 25× larger free-surface deficit than the signal. The clamped Tait
pressure then saturates the outer shell at `p = 0` and the interior is left with a
pressure of order `c0²·(ρ_c−ρ0)/ρ0 ≈ 21 Pa` rather than 360 Pa. Figure (c) shows exactly
this for m2454: `p(r<0.3R)` oscillates around a small positive value instead of settling
on `σ/R`.

## 4. Two defects in the merged tree (blocking, and neither is physics)

1. `src/tensorsph/__init__.py:41` imports `apply_contact_angle_boundary`, which does not
   exist in `src/tensorsph/surface_tension.py`: **`import tensorsph` raises ImportError**,
   so every test module that imports the package fails at collection.
2. `src/tensorsph/droplet_impact.py:609` calls
   `compute_surface_tension_force(..., wall=...)`, which the merged force function does
   not accept: **any solver run with `σ > 0` raises TypeError**.

Both come from commit `bec5848` (m2090), whose commit message records
"1 个文件被门禁滤除": the `surface_tension.py` part of the change (+208 line diff) was
dropped by the merge gate while `__init__.py` and `droplet_impact.py` landed. That is why
the same bug has been re-attacked by six consecutive missions and why none of their fixes
is visible on `main`. This is a delivery-pipeline defect, not a modelling one; it is
reported here and not acted on (platform is frozen for agents).

## 5. Verdict and recommendation

* **Merge candidate: m2447.** Its `src` change is the only one that puts the cut-force
  identity inside 15 % at every `dx` (+1.077 at 2 mm) with inward normals and a
  25× quieter spurious current than the merged baseline, and it also carries the
  contact-angle BC that unbreaks the import. m2454 is a close second on the force
  criterion but drifts at the finest `dx` and does not carry the contact-angle BC.
* **Do not report "Young-Laplace verified"** on the strength of a force integral. The
  pressure-field criterion is `-46.6 %` at best (m2454) and `-87.8 %` (m2447); under the
  platform's 15 % rule the physical criterion remains **FAIL**.
* The honest evidence grade for "surface tension verified against Young-Laplace" stays
  **WITHHELD**: force normalisation is repaired, the pressure jump is not demonstrated.

## 6. Limits of this referee study

* Single estimator geometry (circular drop, hex lattice), one resolution for B/C
  (`dx = 2 mm`), one capillary time; a resolution ladder for Δp_pressure is the obvious
  next measurement and is *not* claimed here.
* Only the default public path is scored (`σ > 0` through `rhs`). Any candidate whose
  repair is opt-in behind a non-default flag would be scored as "unchanged", which is
  the intended semantics (a user who sets `σ` must get the repaired model).
* Shadow-package scoring isolates `src/tensorsph`; a candidate whose fix additionally
  needs a change outside `src/tensorsph` would be under-scored.

## 7. Reproduction

```bash
sh benchmarks/csf_referee/run_all.sh master_p0 m2389 m2399 m2420 m2433 m2447 m2451 m2454 m2459
python benchmarks/csf_referee/summarize.py
```
