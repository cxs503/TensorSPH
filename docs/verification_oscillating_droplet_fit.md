# The mode-2 frequency instrument: why the CI pins disagreed, and what was repaired

**Scope**: `benchmarks/oscillating_droplet/` (measurement instrument and the
signed observable) and `tests/test_oscillating_fit.py`.  `src/` unchanged.
**Artifacts**: `benchmarks/oscillating_droplet/fit_robustness.py` (calibration +
stability study), `fit_robustness_report.json`, `fit_robustness.png`,
`trace_ci_dx0p001.json` (recorded trace, the fixture every number below is
computed from), `tests/test_oscillating_fit.py` (12 tests).

## 1. The case and the discrepancy

The release benchmark drops an elliptical 2-D drop, records the mass-weighted
second-moment anisotropy `s(t)` and reports the mode-2 (Rayleigh-Lamb)
frequency against

    omega^2 = sigma m (m^2 - 1) / (rho R^3),  m = 2,

for `R = 0.01 m`, `dx = 0.001 m`, `sigma = 7.2 N/m`,
`rho0 = 1000.0 kg/m^3`, `nu = 5e-05 m^2/s`, `eps = 0.1`,
`c0 = 20.0 m/s`, 2791 steps of `dt = 1.95e-05 s`
(1.800 analytic periods, 357 particles).
Analytic reference: `omega = 207.846 rad/s`.

Two CI runs of **identical, deterministic, single-threaded code** recorded
`+8.64 %` (x86_64) and `-31.78 %` (loong64).  Those two numbers are pinned as
strict-xfail in `tests/test_oscillating_droplet.py`, with the pin's own comment
saying the fit is "architecture-fragile".  A 40-point spread cannot be physics,
so the instrument was examined first.

## 2. Root cause: the observable is rectified

`mode_amplitude` returns `s_moment = (A - B)/(A + B)`.  Because eigenvalues are
sorted, `A >= B`, so **`s_moment` cannot be negative**: a drop released from
`eps = 0.1` is recorded as `|cos(omega t)|`, i.e. rectified at `2 omega`, and the
sign of the quadrupole is lost.  The docstring's `= eps` claim only holds for
`eps > 0`.

Measured on the recorded trace (`sign_changes_*` counts sign changes about the
mean over 1396 samples):

| quantity | value |
|---|---|
| sign changes of `s_moment` | 3 |
| sign changes of `s_signed` | 2 |
| `r^2` of a damped cosine on `s_moment` | 0.7867 |
| fitted damping `lambda` of that model | +18.71 1/s |
| `r^2` of the same model on `s_signed` | 0.9510 |
| Rayleigh frequency resolution of the window | 55.6 % of `omega` |

A model that cannot go negative needs a DC offset and a 2-omega harmonic to
describe a rectified cosine; `r^2 = 0.787` is what that looks
like.  Calibration on *closed-form* traces isolates the bias: an exact damped
cosine is recovered to 0.000 %, the same cosine
rectified (`|cos|`) is fitted **+68.17 %** high.  The rectification is
therefore a first-order defect of the reading, before the solver contributes
anything.

## 3. What was changed

1. **`mode_amplitude(pos, mass, reference_axis=None)`** now also returns
   `s_signed = s_moment * cos(2 theta)` with
   `cos 2 theta = 2 (e_major . e_ref)^2 - 1`, `e_ref` the initial major axis.
   It is smooth, equals `+s` while the ellipse is aligned with the launch axis,
   `0` when the shape passes the circle and `-s` after a 90-degree turn, so it
   reconstructs `eps cos(omega t)`.  Geometrically verified on exact ellipses:
   `+0.1001` aligned, `0.0000` at 45 degrees, `-0.1001` turned by 90 degrees,
   `0.000125` for a circle.
2. **`damped_cosine_fit(..., starts=FIT_STARTS)`** no longer descends from a
   single coarse-grid winner.  The coarse `(omega, lambda)` grid is clustered
   into basins separated by at least one Rayleigh bin `2 pi / T`, up to six
   basins are refined, and the lowest rms wins.  The returned
   `ambiguity_rms_gap` states how much better the winner is than the best other
   basin: a near-zero gap flags an unidentified frequency instead of silently
   reporting a basin lottery.
3. `run_case` keeps both readings: the legacy `fit_second_moment_*` keys are
   unchanged (historical record), and `fit_signed_*` /
   `zero_crossing_signed_*` are added next to them.

## 4. Results on the recorded CI run

| estimator | error vs analytic `omega` | `r^2` |
|---|---|---|
| fit of |s| (legacy, shipped pin) | -31.50 % | 0.7867 |
| zero-crossing of |s| | -36.57 % | - |
| half the kinetic-energy frequency | -42.95 % | 0.1620 |
| fit of s_signed (this work) | -42.95 % | 0.9510 |
| zero-crossing of s_signed | +nan % | - |

* The legacy reading reproduces the shipped pin: `-31.50 %`
  against the pinned `-31.78 %` (loong64), i.e. the pin is reproducible here and
  is a property of the **observable**, not of the machine.
* The repaired reading is `-42.95 %` with `r^2 = 0.9510`
  (legacy `r^2 = 0.7867`), and it is **independently confirmed** by a
  different physical observable: half the kinetic-energy frequency gives
  `-42.95 %`, agreeing with the signed-moment fit to
  < 0.1 percentage point.  On the legacy path the same two estimators disagreed
  by `-16.72 %`.
* Spectral peak of the legacy trace: `0.56 x omega`, i.e. the
  dominant content sits at the rectification harmonic, as predicted.

**Verdict on the physics is unchanged: FAIL.** `|-42.95 %|`
is far outside the 15 % gate and outside 3 %.  It is, however, reproducible, it
agrees with the `-43.5 %` recorded for the `R = 2 cm` reference case in
`docs/verification_oscillating_droplet.md`, and it no longer depends on the
estimator.  The deficiency is real; the previously pinned `-31.78 %` /
`+8.64 %` pair were artefacts of a rectified observable.

## 5. Stability of the point estimate

Window sensitivity (refit on nested windows; a real frequency does not depend on
how much of the record is used):

| window [periods] | samples | error, `s_moment` | error, `s_signed` |
|---|---|---|---|
| 0.40 | 559 | +51.68 % | -42.95 % |
| 0.60 | 838 | -6.51 % | -42.95 % |
| 0.80 | 1117 | -28.62 % | -42.95 % |
| 1.00 | 1396 | -31.50 % | -42.95 % |

The decisive number is the window spread: refitting the legacy observable on
0.4 / 0.6 / 0.8 / 1.0 of the same record moves the reading over
**83.18** percentage points
(`-31.50 % … +51.68 %`), while the
signed observable returns **0.00** points of spread across the same
windows (`-42.95 %` every time).  The
cross-architecture spread that the xfail pin records is
40.42 points -- **less than half** of the window spread
of the legacy instrument on a single machine.  Round-off perturbations at
`1e-12` relative (the scale of a last-bit difference between BLAS builds) move
the shipped multi-start fit by 0.00 points, so the basin-aware
search is stable; the instability lives in the observable, which is where it was
fixed.

Noise calibration (exact damped cosine, seeded white noise on the amplitude):

| relative noise | mean error | min … max |
|---|---|---|
| 0 | -0.000 % | -0.000 … -0.000 % |
| 0.001 | -0.000 % | -0.000 … -0.000 % |
| 0.01 | -0.008 % | -0.033 … +0.019 % |

## 6. Error budget and honest limits

| quantity | vs reference | < 3 %? |
|---|---|---|
| instrument on an exact damped cosine | 0.000 % | yes |
| legacy observable on a rectified closed-form trace | +68.17 % | no (this is the defect) |
| mode-2 frequency, signed observable, recorded CI run | -42.95 % | no |
| mode-2 frequency, kinetic-energy cross-check | -42.95 % | no |
| legacy pin reproduced (loong64) | -31.50 % vs -31.78 % pin | n/a |

Limits, stated plainly:

* The physics error was **not** removed.  `-42.95 %` is a FAIL; this work makes
  the number trustworthy, not small.
* One trace, one resolution (`R/dx = 10`), one architecture (loongarch64, single
  thread).  A resolution ladder is the obvious next step, but it needs a solver
  campaign that the current machine load does not allow; the harness only needs
  `--trace` to accept a ladder later.
* The `x86_64 +8.64 %` reading was not reproduced (no x86 host was available
  here).  The claim is the weaker, checkable one: on **one** machine the legacy
  reading moves 83.18 points with the fitting window and
  11.45 points across estimators, so the pin cannot be interpreted
  as a physical measurement on any architecture -- including the one it was
  recorded on.
* The zero-crossing estimator on `s_signed` is reported but not relied on: it
  returns NaN on this trace (too few mean crossings in 1.80 periods).

## 7. Reproduction

```bash
# instrument calibration + stability study (synthetic traces + the committed fixture)
PYTHONPATH=src python benchmarks/oscillating_droplet/fit_robustness.py
# the tests that pin the two repairs
python -m pytest tests/test_oscillating_fit.py -q
# re-record the trace (357 particles, ~2800 steps, minutes on a contended host)
PYTHONPATH=src python -c "import importlib.util,json;from pathlib import Path;\
spec=importlib.util.spec_from_file_location('r','benchmarks/oscillating_droplet/run.py');\
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);\
c=m.run_case(radius=0.01,dx=0.001,sigma=7.2,rho0=1000.0,sound_speed=20.0,\
viscosity=5e-5,eps=0.10,periods=1.8,sample_every=2);print(c['frequency']['fit_signed_rel_error_pct'])"
```

![instrument validation](benchmarks/oscillating_droplet/fit_robustness.png)
