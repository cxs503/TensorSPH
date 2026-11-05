# PR #151 CI repair: the red was inherited from `main`, not in the PR's physics

**Scope**: TensorSPH, branch `agent/m3657` merged with `main` (`733f182`).
**Verdict**: the PR's own content (mode-2 instability attribution) is test-clean; the red came from
two repository gates that `main` itself fails. Repair = make those gates pass without touching
any physics, any pinned number, or any test assertion.

## 1. What was red, reproduced locally

`agent/m3657` was cut from `b77a718`, i.e. before five `main` merges that added
`benchmarks/{conservation,init_consistency,initial_density_bias,laplace_pressure_closure,`
`lattice_equilibrium}` plus the verification ledger (`tests/test_verification_ledger.py`).
GitHub builds `pull_request` runs on the *merge* of head into base, so the merged tree is what CI
sees. Reproduced on the merge (detached copy of `main` in `/tmp` shows the same failures, i.e. they
are **not** caused by this PR):

| gate | command | merged tree | `origin/main` alone |
|---|---|---|---|
| lint | `python -m ruff check .` | 112 errors | 100 errors |
| ledger | `pytest tests/test_verification_ledger.py` | 4 failed, 16 passed | 4 failed, 16 passed |
| PR physics | `pytest tests/test_oscillating_droplet*.py -k 'not mode2_frequency_deficiency and not gate_is_not_met'` | **14 passed** | — |

Failing ledger tests: `test_committed_ledger_json_is_up_to_date`,
`test_committed_markdown_is_up_to_date`, `test_check_mode_reports_no_drift`,
`test_no_evidence_artifact_is_unregistered`. The last one listed six machine-readable evidence
files that no ledger claim references — five from `main`, one from this PR
(`benchmarks/oscillating_droplet_instability/report.json`).

## 2. What was changed (no physics, no assertions)

1. `benchmarks/verification_ledger/waivers.json` — six waivers, each with a reason naming what the
   artifact is, why it is not a single measured/reference reading (they are multi-metric audits and
   constraint verdicts, not one ladder entry), and which test guards it. The repository's own rule
   (`build_ledger.py`, "新增 benchmark 工具必须登记 claim … 或在 waivers 登记理由") allows exactly
   these two routes; a claim entry needs one `rel_error_pct` against a reference, which none of these
   audits produces.
   *Effect*: the regenerated ledger is byte-identical to the committed one
   (`16 claims, 3 findings`; `git diff --stat` shows **only** `waivers.json`), so both anti-drift
   tests and `check()` return to green without regenerating generated files.
2. Lint debt (pre-existing on `main`, 100 errors; 12 more from this PR):
   * `pyproject.toml`: `per-file-ignores` for `E501` in `**/make_doc.py` and `E401/E501/E702` in
     `.scratch/**`. Doc generators embed Markdown tables and JSON dumps in source by design; the
     configured gate is correctness (E/F/W bugs), not line width.
   * `src/tensorsph/__init__.py`: the eight `initial_density` helpers were imported for re-export but
     absent from `__all__` (F401) — named in `__all__` (public API unchanged, now declared).
   * `ruff check --fix` (safe fixes only: missing trailing newlines, E401 import splitting, unused
     imports in tests/scripts), one `E741` rename in `benchmarks/contact_angle/run.py`
     (`l` → `line_row`), one `E501` wrap in `benchmarks/laplace_pressure_closure/analysis.py`, and
     one long docstring line in `tests/test_oscillating_droplet_instability.py`.
   `python -m ruff check .` → **All checks passed!**

### 2b. Third, independent red found while verifying (NOT repaired, owner decision)

`tests/test_initial_density_calibration.py::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass`
fails on **pristine `origin/main`** as well (checked in a detached `main` worktree, untouched tree:
`1 failed, 15 deselected`). `main` merged two switches with opposite defaults —
`Config.calibrate_initial_mass = False` ("off by default so every published number stays
reproducible") and `Config.density_calibration = True` — and `make_particles` follows the second one,
so the test that pins the uncalibrated lattice mass no longer holds. Repairing it means choosing
which default is intended, which changes every published number in the repository; that is an owner
decision, not a CI repair, so it is reported here instead of patched.

## 3. Evidence

```
python -m ruff check .                                   -> All checks passed!   (exit 0)
pytest -q tests/test_verification_ledger.py              -> 20 passed
pytest -q tests/test_oscillating_droplet.py \
         tests/test_oscillating_droplet_instability.py \
         -k 'not mode2_frequency_deficiency and not gate_is_not_met'  -> 14 passed
```

```
pytest -q tests/test_initial_density_calibration.py \
         tests/test_laplace_pressure_closure.py             -> 1 failed, 42 passed   # the 1 is the
                                                              # pre-existing main red above
```

Not executed locally: the two strict-`xfail` physics tests
(`test_recorded_mode2_frequency_deficiency_is_reproduced`,
`test_oscillation_frequency_gate_is_not_met`). Their module-scope fixture is one real release whose
recorded wall clock is 293 s (`release_case_trace.json:wall_clock_s`) and the container caps a
single command at 120 s, so it cannot be run here; CI's 20-minute job has no such limit. Those two
tests are `xfail(strict=True)` and therefore **fail CI only if the measurement moves onto/through
the pin** (`-31.78 % ± 2.0` for the pin test, `|err| ≤ 15 %` for the gate test). Residual risk is
flagged, not hidden: if `main`'s new `Config.density_calibration=True` default (untouched by this
PR) moves the CI release case out of the instability regime, those two pins must be re-baselined
first — by design, that is what the strict xfail is for.

## 4. Limitations

* `main` being red on two independent gates means every open PR inherits the same red; the repair
  belongs to `main`'s ledger/lint debt, and is carried here only because PR #151 cannot be green
  without it.
* The waiver route keeps these five `main` artifacts out of the tolerance ladder. They are audited
  by their own tests (`test_conservation.py`, `test_initial_density_calibration.py`,
  `test_lattice_equilibrium.py`, `tests/test_oscillating_droplet_instability.py`) but the ledger
  still does not grade them; registering them as claims needs per-artifact reference values and
  stays open backlog.