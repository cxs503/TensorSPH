# CASE-POISEUILLE-SPH · R3 combined Couette-Poiseuille — C1 normaliser ruling

Anchor: `CASE-POISEUILLE-SPH` · 算例池·泊肃叶流 · ref 解析解 · 容差 `< 5 %`
Artifact: `benchmarks/case_poiseuille/out/combined.json`
Tools: `benchmarks/case_poiseuille/rebase_gauge.py`,
`benchmarks/case_poiseuille/pool_conformance.py`,
`tests/test_case_poiseuille_gauge_rebase.py`

## 1. 案例介绍

R3 is the combined Couette-Poiseuille channel used by the anchor case pool: a
plane channel `0 <= y <= H`, `H = 0.1 m`, `nu = 0.01 m²/s`, `rho0 = 1000.0 kg/m³`,
`c0 = 20.0 m/s`, bottom wall at rest, top wall moving at
`U_w = 1.47 m/s`, driven by an adverse body force `g = -4.905 m/s²`. The steady field is

    u(y) = g y (H - y) / (2 nu) + U_w y / H ,

which runs from `u_min = -0.0984 m/s` (an interior reverse-flow layer) to `U_w`,
`Re = U_w H / nu = 14.7`. The predeclared C1 gate is *profile RMS error
over all rows, normalised by `U_w`, `<= 5 %`*; C3 (superposition) uses the same
normaliser in the same artifact.

## 2. 怎么做的

1. **Reconstructed the grader from the committed rows** rather than from its
   summary (`rebase_gauge.py`): RMS of `u - u_a` over the eight/nine rows,
   divided by a gauge; the derivation is proven to be the same function the
   solver-side grader ran because it reproduces the stored values exactly
   (rel diff `0.0e+00`) whenever the stored value came from the rule under test.
2. **Diagnosed the normaliser.** `combined.py` normalised by `max|u_a|` over the
   graded rows. That maximum is a property of the row set, so it moves with
   resolution: 8: 1.2344, 12: 1.3108, 16: 1.3498 m/s, against the declared `U_w = 1.47`.
   The committed artifact spans two code revisions: the gauge actually used is
   recorded per rung in `summary_rebase.stored_gauge_rule_by_rung`.
3. **Ruled the gauge = `max|u_a|` over the continuum channel** (`gauge_scale` in
   `combined.py`, `pool_conformance.py`, `rebase_gauge.py`): endpoints
   `u(0)=0`, `u(H)=U_w` plus the parabola vertex `y* = H/2 + nu U_w/(gH)`. It is
   closed form (no rows), it equals `U_w` here — so the predeclared text is
   honoured verbatim — and it degrades to `g H²/(8 nu)` for a static-wall
   Poiseuille leg (`U_w = 0`), where normalising by `U_w` is undefined. The same
   rule therefore serves every configuration; the code was patched to use it, and
   the resolver `run_channel` was also patched to stamp each leg's rows with its
   *own* closed form (defect D2 below).
4. **Rebased the committed summary from its own rows** — no solver re-run and no
   measured row touched (`measured_rows_touched: false`); superseded values and
   the pre-rewrite SHA-256 are kept inside the artifact, and the tool refuses to
   write if any gate boolean would flip or if C1 would fail afterwards. The
   repair is idempotent (`rebase_gauge.py --check`).
5. **Wired the ruling into the pool record**: `pool_conformance.py` now
   re-derives `CP_profile_rms@Hdx*` with the same closed-form gauge, the pinned
   `KNOWN_BINDING_EXCEPTIONS` is empty, and the leg-row defect is carried as an
   explicit `not_recomputable` record instead of being invisible.

## 3. 计算结果

Table 1 — C1 per rung: committed (mixed gauge) vs one declared gauge

| rung | H/dx | superseded gauge [m/s] | superseded C1 [%] | stored rule | declared gauge [m/s] | C1 [%] | worst row [%] |
|---|---|---|---|---|---|---|---|
| combined_Hdx8 | 8 | 1.234424 | 0.883993 | declared_U_w | 1.470000 | 0.883993 | 0.983747 |
| combined_Hdx12 | 12 | 1.310820 | 0.466242 | sampled_max_abs_u_a | 1.470000 | 0.415755 | 0.550138 |
| combined_Hdx16 | 16 | 1.349817 | 0.175224 | sampled_max_abs_u_a | 1.470000 | 0.160898 | 0.258270 |

Convergence trend, single gauge: **0.884 → 0.416 → 0.161 %**
(committed trend mixed gauges: 0.884 → 0.466 → 0.175 %).
Every rung stays far inside the 5 % gate, so **no verdict changes** — what changes
is that the trend is now readable on one scale and the artifact matches its own
declared gate text.

Pool record after the repair: 55 functionals
(33 recomputed from raw rows, 20 stored,
2 quarantined as not-recomputable), binding max relative
diff **5.98e-12** (was `1.9e-01` with a pinned exception),
binding violations **0**, verdict `FAIL`
carried by the pre-existing honest FAIL on the coarse-rung backflow peak.

## 4. 结果图

`docs/figures/case_poiseuille_gauge_ruling.png`

* left panel — the analytic profile, the declared gauge `U_w` (dashed) and the
  per-rung sampled maxima (dotted): the sampled maximum is a property of the row
  set, `U_w` is a property of the problem;
* right panel — C1 vs `H/dx`, committed (red dashes, mixed gauge) against the
  single declared gauge (blue squares), with the 5 % gate.

## 5. 注意点 / 局限

* **D1 (repaired).** Mixed gauge in the gate-bearing records; the rung-8 value
  had been produced by a revision that normalised by `U_w`, rungs 12/16 by the
  sampled maximum. Repaired by the closed-form gauge plus a rebase from the
  committed rows. The rebase is a *post-processing* repair: the artifact's rows
  are the original solver output, the summary is recomputed, and both are pinned
  by SHA-256 so an un-regenerated edit is detectable. A full `combined.py` rerun
  would regenerate the rows too — it was **not run** here: the local machine is
  over its compute budget (load15 ≈ 130–175 on 128 cores) and the local-compute
  guard blocked it. *Status: not_run, not passed.*
* **D3 (repaired, found while verifying).** With `g = 0` the Couette leg divided
  by a zero momentum-budget normaliser (`ZeroDivisionError`), so the suite could
  not finish end-to-end on the current code — additional evidence that the
  committed leg block predates this revision. The closure is now reported as
  `defined: false` with a NaN percentage instead of raising.
* **D2 (quarantined, not repaired).** `legs[*].rows[].u_analytic` carries the
  *combined* reference for both legs, so leg-level diagnostics are not
  reproducible from the file:

| leg record | g | U_w | max abs(column − own) [m/s] | stored C1 [%] | own ref (sampled gauge) [%] | own ref (declared gauge) [%] | rms/worst reproducible |
|---|---|---|---|---|---|---|---|
| `legs.poiseuille` | -4.905 | 0.000 | 1.3781 | 2.092932 | 2.092932 | 2.060230 | True / False |
| `legs.couette` | 0.000 | 1.470 | 0.6035 | 0.049271 | 0.049271 | 0.046192 | True / False |

  The `profile_rms_pct` of each leg **is** reproducible from its rows against its
  own closed form (rel diff `0.0e+00`); `profile_worst_row_pct` and `tau_*_pct`
  are not. No gate reads the leg records (C1/C2/C4/C5/C6 read `runs`, C3 reads
  measured-only superposition residuals, C7 the FD reference, C8 the static
  control), so no verdict moves. Repairing the columns needs a re-graded artifact
  or a solver re-run; rather than hand-editing evidence the defect is published
  in `out/legs_row_audit.json` and asserted by tests.
* Scope: the ruling concerns the *normaliser* of a percentage, not the physics.
  It does not re-open the honest FAIL of `CP_backflow_peak` at `H/dx = 8/12`
  (`-12.370 %` / `-8.232 %`), which is a genuine coarse-grid discretisation
  error and stays a FAIL.

## 6. 误差判定

| quantity | reference | committed | after ruling | < 3 %? | < 5 % (anchor)? |
|---|---|---|---|---|---|
| C1 profile RMS, H/dx = 8 | analytic `U_w`-normalised | 0.8840 % | 0.8840 % | yes | yes |
| C1 profile RMS, H/dx = 12 | analytic | 0.4662 % | 0.4158 % | yes | yes |
| C1 profile RMS, H/dx = 16 | analytic | 0.1752 % | 0.1609 % | yes | yes |
| C7 FD reference vs closed form | analytic | 0.001572 % | unchanged (already `U_w`) | yes | yes |
| stored-summary vs raw-row binding | committed rows | 1.9e-01 rel (pinned) | 5.98e-12 rel | n/a | n/a |
| `CP_backflow_peak@Hdx8` (pre-existing) | analytic at own row | -12.370 % | -12.370 % (untouched) | **no** | **no** (honest FAIL) |

**是否 <3 %：C1 三个档位全部是（最差 0.8840 %），C7 是；唯一不是的是合并算例 C6 粗档诚实 FAIL（-12.370 % / -8.232 %），本单未改判。**
**是否 <5 %：除 C6 粗档外全部满足。**
