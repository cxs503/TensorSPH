# Adversarial review — CASE-POISEUILLE-SPH phase reporting (unit consistency)

**Anchor**: `feature_ledger` `CASE-POISEUILLE-SPH` (算例池·泊肃叶流, acceptance: analytic
reference, tolerance < 5 %). **Reviewed tree**: branch `agent/m21208`, base commit
`a978a66` (merge #779). **Reviewer**: m21208, independent of the authors of merge
#716 / #739 that delivered the case. **Fix commit**: see `git log` on this branch.

## 1. What was reviewed

The merged case-pool entry `benchmarks/case_poiseuille/` (driver `run.py`, FD
reference `ref.py`, note generator `make_doc.py`), its committed artifact
`out/result.json`, the generated note `docs/verification_case_poiseuille.md`,
the figures under `docs/figures/case_poiseuille_*.png`, and
`tests/test_case_poiseuille.py` / `tests/test_case_poiseuille_wo_envelope.py`.
The question was not "does the case pass" (it does, 9/9 gates) but "does the
passing evidence say what the note claims it says".

## 2. Findings (severity-ranked)

### F1 — Major, report correctness: summary phase fields were radians in  keys

`evaluate_harmonic` stored the fitted phase (radians, the `harmonic_fit`
convention) directly into `phase_centre_deg` / `phase_centre_analytic_deg`.
The Womersley centre-line lag at Wo = 1.80 is **-54.81 deg**; the artifact
carried **-0.96**, i.e. a factor **57.3 = 180/pi** low, and the generated note
printed it as

> "the centre-line phase is -0.95 deg against -0.96 deg"

Evidence (pre-fix `out/result.json`, resolution H/dx = 16):
`phase_centre_deg = -0.9547`, `phase_centre_analytic_deg = -0.9567`, while
`rows[k]["phase"] = -0.9547 rad` and `tensorsph.womersley.phase(y_centre, ...)
= -0.9547 rad = -54.81 deg`. The same defect propagates to
`frequency_discrimination.phase_difference_*_deg` (reported -0.39 "deg", true
-22.33 deg).

**Gates were not affected** (this is what keeps the severity at "major" and not
"blocking"): G3/G4 are computed from `phase_error_deg`, which is
`math.degrees(wrap(fit - analytic))` and therefore in true degrees, and the
percentage-of-quarter-turn conversion uses `math.radians` on that degree value.
All nine gate verdicts are unchanged by the fix. `ref.py` (FD reference) was
already in degrees throughout.

What the bug *did* damage is the meaning of the evidence: the note made a
54.8 deg phase lag look like a 1 deg lag, i.e. it hid the fact that G3/G4 are
strongly discriminating. A quasi-steady solver (phase 0) misses the centre-line
phase by 54.8 deg, **12x** the 4.5 deg gate, and the measured error is 0.12 deg,
**38x** inside it.

### F2 — Major, figures: the phase panel mixed radians and degrees

`make_doc.fig_harmonic_profiles` plotted SPH and closed-form rows with
`r["phase"]` (radians) and the FD reference with `r["phase_fd_deg"]` (degrees)
on one axis labelled "phase vs forcing [deg]". Reproduced by rebuilding exactly
that panel from the committed artifact: the autoscaled y-range came out
**-57.52 .. +2.01**, so the three SPH ladder curves and the closed form —
the whole point of the panel — collapsed into a flat line at the top of the
axes. After the fix the same panel spans **-56.28 .. -38.86 deg** (degrees, all three
series mutually consistent).
`fig_centreline_series` applied `math.radians()` to `centre["phase_analytic"]`,
which was already radians: a second conversion, off by 57.3.

### F3 — Moderate, vacuous test: three assertions in the envelope suite never asserted

`tests/test_case_poiseuille_wo_envelope.py::test_centre_amplitude_and_phase_recompute`
contained

```python
assert abs(centre["phase"] - p["phase_centre_deg"]) < 1e-12 \
    if "phase_centre_deg" in p else True
assert centre["phase_analytic"] == pytest.approx(p["rows"][0]["phase_analytic"],
                                                 0.0) or True
```

`phase_centre_deg` **never exists** in envelope points (the study stores
`phase_centre_error_deg`), so the first assertion evaluated to `True`
unconditionally; the second is short-circuited by `or True` and, even without
it, compared the centre row's analytic phase with row 0's. The test executed
and asserted nothing — and the missing assertion is exactly the one that would
have caught F1. Replaced with real invariants (radian range of the stored row
phases, and the stored analytic values recomputed from
`tensorsph.womersley` at the stored ordinates).

### F4 — Minor, latent trap

Because F1's fields were radians under `*_deg` names, any future consumer that
writes a tolerance against them (`abs(phase_centre_deg) <= 5`) would build a
gate **57x** too permissive while looking correct in review. The unit-explicit
schema (`*_rad` / `*_deg`) removes the ambiguity; a schema test now forbids the
regression.

## 3. What was changed

| file | change |
|---|---|
| `benchmarks/case_poiseuille/run.py` | emits `phase_centre_rad`/`phase_centre_deg`, `phase_centre_analytic_rad/_deg`, per-row `phase_deg`/`phase_analytic_deg`, and `phase_difference_{measured,analytic}_{rad,deg}` — every quantity carries its unit |
| `benchmarks/case_poiseuille/make_doc.py` | phase panel plots degrees for all three series; centre-line series uses `phase_analytic_deg` (single conversion) |
| `benchmarks/case_poiseuille/migrate_units.py` | new, idempotent: rewrites the derived phase fields of an existing `result.json` from the stored radian fits (raw samples, profiles and gate verdicts untouched) |
| `benchmarks/case_poiseuille/out/result.json` | migrated to the unit-explicit schema (`units_schema` key records how) |
| `docs/verification_case_poiseuille.md` + figures | regenerated from the migrated artifact (numbers programmatically injected) |
| `tests/test_case_poiseuille_phase_units.py` | new: unit consistency of summary, per-row and frequency fields; `*_deg` within ±180; analytic centre phase equals the closed form; plus a solver-free test that feeds `evaluate_harmonic` a case that *is* the closed form |
| `tests/test_case_poiseuille_wo_envelope.py` | dead assertions replaced by real ones |

**No gate threshold was touched**, no result was re-tuned, and no verdict was
overwritten: the case is still 9/9 PASS, now with the phase evidence reported at
its true magnitude.

## 4. Reproduction

```bash
# pre-fix artifact: the new schema tests fail 5/5 (F1/F4 reproduced)
PYTHONPATH=.:src python3 -m pytest tests/test_case_poiseuille_phase_units.py -q
# post-fix (migrated artifact + fixed generator)
PYTHONPATH=.:src python3 -m pytest tests/test_case_poiseuille_phase_units.py \
    tests/test_case_poiseuille_wo_envelope.py -q      # 81 passed
PYTHONPATH=.:src python3 -m pytest tests/test_case_poiseuille.py -q   # 11 passed (49 s, includes one SPH smoke run)
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/migrate_units.py
PYTHONPATH=.:src python3 benchmarks/case_poiseuille/make_doc.py
```

## 5. Independence limitations (honest)

* The reviewer wrote the fix, so this note is not an arms-length review of the
  fix; the code changes are small and each is covered by a test that fails on
  the pre-fix artifact, which is the strongest available substitute.
* **No re-simulation was completed.** The full ladder was submitted to the
  compute gate (`cj3270`, 2 cores); it was still queued when this note was
  written, and local execution is capped at 120 s, so the committed artifact's
  numeric SPH results are the **original** run's (merge #739), not a fresh
  reproduction. Only the *derived* phase fields were migrated from that run's
  stored radian fits — a relabelling, not a re-derivation of the physics.
* The pre-fix defect's impact was bounded by reading every consumer of the
  affected fields in the repository; a downstream consumer outside this repo is
  not observable from here.

## 6. Checks not run

* full ladder re-simulation (queued, `cj3270`; verdict on the physics unchanged
  by a relabelling, but the reproduction claim remains open);
* no independent second SPH implementation, so the "WCSPH reproduces Womersley
  to 0.1 % in phase" claim rests on the repository's own solver;
* the Wo-envelope study (`study_wo_envelope.py`) was checked for unit
  consistency only, its physics conclusions were not re-derived.

## 7. Overstated claims found in the merged artifact

1. `docs/verification_case_poiseuille.md` §3: "centre-line phase is -0.95 deg
   against -0.96 deg" — true values -54.70 deg against -54.81 deg.
2. `docs/verification_case_poiseuille.md` §3.1: "phase difference between the
   two frequencies [deg] | -0.39 | -0.39" — true -22.33 | -22.14.
3. `docs/figures/case_poiseuille_profiles.png` (right panel) — unreadable
   before the fix, and its caption claims a three-way comparison it did not
   actually display.
4. `tests/test_case_poiseuille_wo_envelope.py` docstring: "recomputes every
   number it can without the solver" — the phase branch did not.

## 8. Disposition

**Approve with changes, changes applied.** F1/F2/F4 fixed and regression-guarded;
F3 fixed. The case-pool entry's acceptance (< 5 % vs analytic) is met on the
evidence that survives the review: amplitude 0.07 % centre / 0.39 % profile RMS,
phase 0.12 deg / 0.21 % of a quarter turn, at H/dx = 16, with the independent
Crank–Nicolson reference agreeing with the closed form to 0.004 % (G9).
