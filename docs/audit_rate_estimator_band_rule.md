# The band rule, made mechanical: audit of hand-chosen amplitude windows

**Anchor**: `CASE-SPH-LAKE-AT-REST` ｜ **Date**: 2026-11-15 ｜ **Repo**: TensorSPH
｜ **Kind**: 验证欠账结清（把锚点的估计器纪律推广到兄弟实现，不重跑物理）

## 0. Why this exists

`docs/audit_lake_at_rest_anchor_m21507.md` closed the anchor's evidence channel
and left one remediation item explicitly open, verbatim:

> *The rest of the repository's cases quote exponential/linear rates from
> hand-chosen amplitude windows; the band rule here should be checked against
> them before the same failure is discovered twice.*

The failure in question is not subtle: an exponential rate fitted over an
amplitude window that spends its time in a start-up transient returns the
transient's mean slope (`0.887 1/s`, `R² = 0.29`) where the rate is
`16.47 1/s` (`R² = 1.0000`), and the *same* error at the other end returns
`8.19 1/s` when the window reaches saturation. Two consecutive missions built
their conclusions on that estimator before anyone checked the window.

That item is what this mission does. It is a **verification** mission: nothing
here re-runs physics, every number below is a pure function of already-frozen
series, re-derived by `benchmarks/rate_band_audit/run.py` into
`docs/rate_band_audit.json`.

## 1. What was done

### 1.1 The rule stopped being a constant pair

`benchmarks/lake_at_rest/band_fit.py` states the rule as a rationale plus two
constants (`BAND_LOW = 1e-6`, `BAND_HIGH = 1e-3`). Constants do not travel: the
next case has a different roundoff floor and a different saturation amplitude.
`src/tensorsph/rate_fit.py` (new, 460 lines) makes the rule mechanical:

| function | what it decides |
|---|---|
| `fit_band(t, a, low, high)` | slope of `ln a`, `R²`, point count, the `t`/`a` span actually used, validity |
| `ladder(t, a)` | every band of ≥ 1 decade the observed range contains (edges at 1/4 decade) — the rate must survive all of them |
| `plateau(t, a)` | maximal run of adjacent, admissible (`R² ≥ 0.98`) bands whose slopes agree to 5 %. Returns the median rate, the plateau's two amplitude edges, and `reason_kind` when there is no plateau (`sparse` / `not_exponential` / `narrow_plateau` / `curved`) |
| `sub_band_spread(t, a, low, high)` | nested sub-band sweep **inside a declared window** — the check the lake study did by hand, automated; a window straddling a transient or a saturation shows a slope *trend* |
| `relative_seed_defect(t, a)` | what a `[3a₀, 25a₀]` window returns (the form `threed.liquid_column` uses) and how far it sits from the plateau rate |

The reusable decision is **not** "quote the plateau median". A case declares its
band a priori — that is what keeps it free of hindsight — and the rule is that
the declared band must (i) itself be exponential with enough points, (ii) agree
with the plateau rate to within the plateau tolerance, and (iii) be stable under
its own sub-band sweep. `rate_fit.verdict()` returns exactly that, with the
failing reason spelled out.

### 1.2 The two sibling implementations now carry the verdict

* `threed/liquid_column.fit_growth_rate(times, amplitudes, low=None, high=None)`
  — `None` selects the data band; the returned record carries `band_source`,
  `plateau_omega`, `declared_legal`. `ColumnResult.growth_banded()` added.
  Defaults unchanged (`3.0, 25.0` × seed) so no shipped number moves, and its
  docstring now says the default band is seed-relative.
* `rayleigh_taylor.fit_growth_rate(times, amplitudes, lo=None, hi=None)` — same
  `None` path; when a window *is* declared the record carries
  `band_rule.declared_band_legal` and the plateau it was compared against, so a
  case report cannot quote the fit without the check.

`threed/solver.fit_decay_rate` (no window at all: it fits through saturation) is
**not** touched — see §5.

## 2. Results

### 2.1 The module reproduces the frozen lake numbers bit-for-bit

`fit_band` on the shipped raw series with the declared band, against
`tests/data/lake_at_rest/stability_band_fit.json` (manifest hashed):

| arm | shipped λ [1/s] | this module | rel. diff |
|---|---|---|---|
| base | 16.46973709593291 | 16.46973709593291 | 0.0 |
| courant_half | 16.557888787866155 | 16.557888787866176 | 1.3e-15 |
| resolution_dx025 | 14.915788696328516 | 14.915788696328532 | 1.1e-15 |
| pairs_rebuilt | 16.421239906612882 | 16.421239906612875 | -4.4e-16 |
| seed_1e-6 | 16.475400768022272 | 16.475400768022270 | -2.2e-16 |
| seed_1e-8 | 16.599840254325564 | 16.599840254325564 | 0.0 |
| viscosity_1e-6 | 16.63421782356855 | 16.634217823568537 | -8.9e-16 |
| saturation_8s | 16.562554494944777 | **refused** (5 points) | — |

Seven of eight agree to machine precision. The eighth is a **finding**, not a
mismatch: the producer's `MIN_POINTS` is 4, so it ships a rate fitted to 5
samples, and at 103 samples over 12 decades (6.9 points/decade) the ladder
cannot resolve a one-decade band at all (`reason_kind = not_exponential`,
`points_per_decade = 6.9`). The arm that was supposed to test *saturation* is
the one arm whose rate is not measurable at this sampling — it must not be cited
as an independent check.

### 2.2 The declared band is legal on every unseeded arm, and one seeded arm is marginal

| arm | declared λ [1/s] | plateau λ [1/s] | sub-band spread | declared band legal |
|---|---|---|---|---|
| base | 16.4697 | 16.3441 | 0.7 % | **yes** |
| courant_half | 16.5579 | 16.4420 | 1.0 % | **yes** |
| resolution_dx025 | 14.9158 | 14.9236 | 1.1 % | **yes** |
| pairs_rebuilt | 16.4212 | 16.3702 | 1.1 % | **yes** |
| seed_1e-8 | 16.5998 | 16.5594 | 1.0 % | **yes** |
| viscosity_1e-6 | 16.6342 | 16.5350 | 1.1 % | **yes** |
| seed_1e-6 | 16.4754 | 16.6038 | **6.1 %** | **no** (marginal) |
| saturation_8s | — | — | — | no (unmeasurable, §2.1) |

`seed_1e-6` is reported, not dropped: its rate is within 0.8 % of the plateau, so
the seed-independence conclusion is unaffected, but its *band's lower edge*
(`1e-6`) sits below its own plateau start (`1.9e-6`). That arm's band should be
quoted as `2e-6 … 1e-3`, not `1e-6 … 1e-3`.

### 2.3 The legacy window is reproduced exactly where it failed

| arm | legacy `1e-10 … 1e-5` λ [1/s] | R² | legal |
|---|---|---|---|
| base | 16.299 | 0.9999 | yes |
| courant_half | 15.519 | 0.9999 | no (-5.6 % vs plateau) |
| resolution_dx025 | 14.889 | 1.0000 | yes |
| pairs_rebuilt | 16.391 | 1.0000 | yes |
| **seed_1e-6** | **0.8874** | **0.288** | no |
| **seed_1e-8** | **3.7236** | **0.575** | no |
| viscosity_1e-6 | 16.536 | 1.0000 | yes |

`0.8874` and `3.7236` are the documented failures (`0.887`, `3.724`). Note that
the same window is *fine* on five other arms — which is exactly why a single-arm
check never found the defect, and why the gate has to be evaluated per arm.

### 2.4 A seed-relative window tracks the seed (the `liquid_column` form)

The window `[3a₀, 25a₀]`, with `a₀` the first sample, on the lake's own arms:

| arm | band | λ [1/s] | R² | error vs plateau |
|---|---|---|---|---|
| base | 1.1e-14 … 9.0e-14 | 13.09 | 0.982 | **-19.9 %** |
| courant_half | 1.4e-14 … 1.1e-13 | 10.55 | 0.965 | **-35.8 %** |
| resolution_dx025 | 8.7e-15 … 7.2e-14 | 7.49 | 0.878 | **-49.8 %** |
| pairs_rebuilt | 1.1e-14 … 8.9e-14 | 9.50 | 0.908 | **-42.0 %** |
| seed_1e-6 | 7.4e-7 … 6.2e-6 | 15.43 | 0.995 | -7.1 % |
| seed_1e-8 | 7.4e-9 … 6.2e-8 | 15.58 | 0.988 | -5.9 % |
| viscosity_1e-6 | 1.06e-14 … 8.87e-14 | 11.36 | 0.901 | -31.3 % |

The window *width* is identical on every arm (0.92 decades) and its *position*
moves by seven orders of magnitude, because it is anchored to the seed and not
to the exponential. Every one of these fits has `R² ≥ 0.878`, and three have
`R² ≥ 0.98` (base 0.982, seed_1e-6 0.995, seed_1e-8 0.988): **`R²` alone would
have passed them.** This is the defect class,
measured on frozen data, and it is the form still used by
`threed.liquid_column.ColumnResult.growth()` (defaults `3.0, 25.0`).

### 2.5 Synthetic controls: three regimes, two refusals

| series | truth | what a window gives | what the rule gives |
|---|---|---|---|
| exact exponential | σ = 3.7 | 3.7000000000000 | 3.7000, plateau 6.4 decades, rel. err **5e-15** |
| start-up transient | σ = 16 | window in the join: **28.78** (+76.6 %), `R² = 0.953` → **illegal** (sweep 103 %) | 16.29 (+1.8 %) |
| saturating | σ = 6 | window into saturation: **4.61** (-23.1 %), `R² = 0.937` → **illegal** | window on the plateau: 6.000000 (−3e-16) |
| power law `(1+t)³` | none | window `5…200`: `R² = **0.977**`, σ = 0.852 → looks like a rate | **refused**: sub-band sweep 84.6 %, `reason_kind = curved` |
| sparse (13 samples / 8 decades) | 2.5 | — | **refused**: `reason_kind = sparse`, 1.5 points/decade |

The power-law row is the one that justifies `sub_band_spread`: `R²` inside a
window is not evidence of an exponential, because a curve is locally straight in
`log a`. A slope *trend* is.

### 2.6 Siblings through their public entry points

| call | result |
|---|---|
| `liquid_column.fit_growth_rate(t, a, None, None)` on an exact exponential | ω = 15.000000000000, `band_source = auto`, rel. err **-1.1e-16** |
| `liquid_column.fit_growth_rate(t, a, 3.0, 25.0)` on a run that saturates at 8a₀ | ω = **0.209** vs 15 (−98.6 %), `R² = 0.151`, `declared_legal = false` |
| `rayleigh_taylor.fit_growth_rate(t, a, 1e-7, 1e-5)` | σ = 15.000000, `band_rule.declared_band_legal = true`, bias 4.4e-16 |
| `rayleigh_taylor.fit_growth_rate(t, a, None, None)` | σ = 15.000000000000005, `band_source = auto` |

## 3. Figures

* `docs/figures/rate_band_lake_ladder.png` — λ vs band centre for all eight arms,
  with the declared band (green) and the legacy window (red) shaded. The plateau
  is where the curves are flat; the legacy window sits under the seeded arms'
  start-up.
* `docs/figures/rate_band_regimes.png` — the three regimes a window can fall
  into (transient / saturation / power law), with the biased window marked and
  the fit it returns.
* `docs/figures/rate_band_seed_relative.png` — rate error of the seed-relative
  window vs the declared band, per lake arm.

## 4. Error judgement (explicit)

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| module vs frozen producer band λ (7 arms) | `stability_band_fit.json` | ≤ 1.3e-15 rel. | **yes** |
| plateau λ vs true σ, synthetic exponential | 3.7 | 5e-15 | **yes** |
| plateau λ vs true σ, synthetic transient | 16.0 | +1.8 % | **yes** |
| plateau λ vs true σ, synthetic saturation | 6.0 | -3e-16 | **yes** |
| `liquid_column` auto band vs true σ | 15.0 | -1.1e-16 | **yes** |
| `rayleigh_taylor` declared window vs true σ | 15.0 | 4.4e-16 | **yes** |
| *refused* series (power law, sparse, saturation_8s) | — | no number quoted | n/a — refusal is the result |

The anchor's own verdict is unchanged by this mission and is **not** re-graded
here: `FAIL` as an unconditional machine-precision rest claim, `PASS` over the
certified horizon `t* = 1.1805 s > 0.50 s`. Nothing in this document asserts
otherwise.

## 5. Limitations and what was **not** run (unknown, not passed)

1. **No physics re-run for any sibling case.** `rayleigh_taylor` and
   `threed.liquid_column` were exercised only on synthetic series and on the
   lake's frozen series. Whether their *shipped* rates are biased is therefore
   **not_run**: what is shown is that the *estimator form* returns numbers 6–50 %
   off on real data, and that the form is in the code. Getting the actual numbers
   needs those cases' runs (RT 2D SPH, liquid column 3D SPH) and was out of this
   mission's compute envelope (load15 ≈ 73/128 at start).
2. `threed/solver.fit_decay_rate(times, amplitudes)` still fits the **whole**
   series with no band, i.e. it includes the saturated tail by construction. It
   is left untouched because its callers consume the number and a silent change
   would move published values; flagged here as the next case to convert.
3. `saturation_8s` is unmeasurable at one-decade resolution (§2.1). Its shipped
   5-point rate is retained but is not an independent check.
4. `freeze.py --verify` (full reproduction of the frozen channel) was not run —
   same limitation as m21507, and for the same reason (it re-runs physics).
5. The `plateau` search is greedy (longest agreeing run, then walk inwards). On a
   series with two separated exponential regimes it would report the longer one;
   none of the audited series has that shape, and the ladder is reported in full
   in the JSON so a reader can see it rather than trust the choice.

## 6. Checks run

```
PYTHONPATH=src pytest tests/test_rate_fit_band_rule.py -q        # 14 passed
PYTHONPATH=src pytest tests/test_rayleigh_plateau.py tests/test_rayleigh_taylor.py -q
                                                                  # 20 passed, 1 xfailed
PYTHONPATH=src python benchmarks/rate_band_audit/run.py           # writes docs/rate_band_audit.json + 3 figures
```

## 7. Disposition

**Approve with notes.** The remediation item from m21507 is closed for the two
implementations whose window form is a defect or an unchecked declaration
(`liquid_column`, `rayleigh_taylor`), and closed *mechanically*: the check now
runs where the number is produced, not in a document. The item is **not** closed
for `threed/solver.fit_decay_rate` (§5.2) and cannot be closed for the sibling
cases' shipped rates without re-running them (§5.1) — both are named follow-ups,
not silent gaps.

Next case to convert, in priority order:
1. `threed/solver.fit_decay_rate` — add the band, then re-run its case once and
   report both numbers side by side.
2. `rayleigh_taylor` run — the case has an analytic reference
   (`theory_growth_rate`, finite-depth and viscous-corrected), so a
   `declared_band_legal` gate there converts this audit from "the form is
   unsound" to "this case's published rate is/isn't biased", which is the
   measurement that matters.
3. `freesurface.profiles.peak_decay_rate` — the peak envelope of a *damped
   oscillation*: same class (the envelope is not exponential once the oscillation
   drops into the numerical floor), no band, no `R²`. Gate it the same way.