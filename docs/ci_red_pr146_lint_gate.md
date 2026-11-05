# PR #146 CI red: root cause and fix (mission m4110)

## 1. Symptom
All three jobs `pytest (3.10) / (3.11) / (3.12)` of PR #146 (branch `agent/m3597`,
head `3f3ecd8`) were reported as failure on GitHub Actions. The job name is
misleading: the failing **step** is `Lint` (`python -m ruff check .`), the 5th
step of the job, i.e. pytest never even started on the affected runs.

Evidence (GitHub REST, job steps of run 42001248752 / check-runs of `3f3ecd8`):

| job | conclusion | failing step |
|---|---|---|
| pytest (3.10) | failure | step 5 `Lint` |
| pytest (3.11) | failure | step 5 `Lint` |
| pytest (3.12) | failure | step 5 `Lint` |

The same step is red on `main` itself (`24cd6ae`, `f61770f`), so the red is
inherited, not caused by the PR's own test-only change.

## 2. Reproduction (local, loongarch64, ruff 0.15.0)
`git checkout -B fix origin/main` (733f182) then

```
python -m ruff check .            # 100 errors
python -m ruff check . --statistics
  76  E501 line-too-long
  12  F401 unused-import
   8  W292 missing-newline-at-end-of-file
   2  E702 multiple-statements-on-one-line-semicolon
   1  E401 multiple-imports-on-one-line
   1  E741 ambiguous-variable-name
```

The `agent/m3597` branch tip is based on an older `main` (`b77a718`) whose tree is
lint-clean, which is why the red was not visible in the author's local run.

## 3. Fix (this branch)
Nothing is suppressed wholesale; each class is addressed where it is a real finding:

| finding | files | treatment |
|---|---|---|
| 50 E501 | `benchmarks/*/make_doc.py` | `per-file-ignores E501` — these scripts emit Markdown tables/prose; a table row cannot be wrapped |
| 25 E501 + E401 + 2 E702 | `.scratch/mk_report.py` | `.scratch/` excluded: committed throw-away diagnostics, not package code |
| 1 E501 | `benchmarks/laplace_pressure_closure/analysis.py` | line wrapped |
| 8 F401 | `src/tensorsph/__init__.py` | names were re-exported but missing from `__all__` -> added (public API unchanged) |
| 2 F401 | `benchmarks/contact_angle/run.py`, `tests/test_initial_density_calibration.py` | genuinely unused imports removed |
| 1 E741 | `benchmarks/contact_angle/run.py` | `l` -> `row` |
| 8 W292 | 8 files | trailing newline added (behaviour-neutral) |

Additionally `ruff` is **pinned to 0.15.0** in the `test` extra. The gate was
`ruff>=0.6`, so the rule set (and E501's pragma handling, changed in 0.17.0) could
change under the repository without a code change; pinning makes the lint verdict
reproducible and makes the "verified locally" claim transferable to CI.

Verification: `python -m ruff check .` -> `All checks passed!` (ruff 0.15.0,
Python 3.11.6).

## 4. Remaining red (NOT fixed here, owner-visible)
Once lint passes, the very next step fails for a reason that predates this PR:

```
python -m pytest -q tests/test_initial_density_calibration.py::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass
1 failed
E  assert torch.allclose(mass_off, torch.full_like(mass_off, RHO0*dx*dx))
```

Measured on `origin/main` **without** this branch's edits:

| quantity | value |
|---|---|
| max abs difference (kg) | 2.128e-07 |
| max relative difference | 5.320e-05 |
| `torch.allclose` default rtol | 1e-05 |
| verdict | fail (5.3x the default rtol) |

5.3e-5 is ~440 float32 ULP at 0.004, i.e. not ordinary rounding of a single
`rho0*dx*dx` product: the particle mass of the default (non-calibrated) path was
changed by a recent merge to `main`. Deciding whether the formula or the test bit
is wrong is a src-level (physics) question, so this file deliberately does **not**
relax the tolerance: doing that here would hide a mass-initialisation change behind
a CI-green, which is exactly the failure mode this repository warns about.

Consequence: **`main` pytest is red independently of PR #146**; the CI of PR #146
cannot go green by a test-only change on this branch. The blocker is reported with
a one-line reproduction and the measured numbers above.

## 5. Notes / limits
- Verified on loongarch64 only; the arch-conditional mode-2 marker from `m3597` is
  kept as authored (it is the PR's content, and it is not the red).
- Concurrent agents re-point the shared editable install of `tensorsph` at their own
  worktree (`pip list` showed `tensorsph 0.8.0 /root/tp-agent/work/m4111`), so every
  local run here was done with `PYTHONPATH=<this worktree>/src` to guarantee the
  tree under test; a stale editable install silently tests another tree.
