# PR #187 (agent/m3933) CI triage — why it was red, what this branch fixes, what is left

Mission m4350, first round. Evidence files: `artifacts/ci_red_pr187/ci_lint_excerpt.log`,
`artifacts/ci_red_pr187/ci_lint_summary.json` (extracted programmatically from the
GitHub Actions job log, nothing hand-copied).

## 1. Symptom

`pytest (3.10) / (3.11) / (3.12)` all failed, each after ~90 s, i.e. far too early for the
test phase (the suite alone needs many minutes). The failing step is **Lint**
(`python -m ruff check .`), the first step of the `pytest` job, so one lint error turns
all three matrix entries red identically. The checked-out tree is the PR merge ref

```
7ea41c0 Merge a1e67eb (agent/m3933) into 24cd6aeb   # main as of PR creation (#138)
```

Lint output: `Found 97 errors` over 12 files (rule breakdown in the summary JSON above;
`E501` dominates, plus `W292`, `F401`, `E401`, `E702`). CI resolves `ruff>=0.6` to
**0.17.3** (visible in the pip install list of the job log), which is stricter than the
0.15.0 available locally in this sandbox.

## 2. Root causes — two independent ones

**R1 — the branch base is stale, so its tree has no ruff `.scratch` exclusion.**
`agent/m3933` was cut from `ba07a10` (Merge #104); main has since moved to `7d91b89`
(#168–#172). The merge ref still resolves against `24cd6aeb`, whose `pyproject.toml`
predates `exclude = [".scratch"]` (added in `87324f6`, merged with #168). `.scratch/mk_report.py`
is therefore linted and contributes **28** of the 97 errors. That file is inherited from
the base branch; m3933 never touched it.

**R2 — the base branch already carried lint debt.**
The other **69** errors live in files m3933 never touched:
`benchmarks/laplace_pressure_closure/make_doc.py` (38), `initial_density_bias/make_doc.py` (9),
`src/tensorsph/__init__.py` (8), `effective_sigma/make_doc.py` (5) and six more files with
1–3 errors each. Check-run history shows why this debt reached main: the CI of the last
five merges (#168 `bc9e8b1`, #169 `5cdafed`, #170 `dea5923`, #171 `2d89645`, #172 `7d91b89`)
is `cancelled`, not `success` — the merge governor did not wait for the gate. Local ruff 0.15.0
on current main `7d91b89` still reports **113** errors in 18 files (16 more than CI's 97,
because main has gained `contact_angle_geometry` and friends since the PR base).

## 3. What this branch does

The m3933 deliverables are rebased onto current `main` (`7d91b89`), keeping their content
byte-for-byte:

| file | vs main |
|---|---|
| `benchmarks/oscillating_droplet/run.py` | +101 / −10 (the 10 "removed" lines are the same statements rewritten in the guarded form, verified line by line) |
| `benchmarks/oscillating_droplet/make_doc.py` | +73 |
| `tests/test_oscillating_droplet.py` | +81 (section 5: m2919-lineage properties) |
| `docs/verification_oscillating_droplet.md` | +27 |
| `benchmarks/oscillating_droplet/validation_report_m2919.json` | new, byte-identical (md5 `72283277…`) to main's `validation_report_lamb.json` |
| `docs/merge_resolution_pr87.md`, `docs/verification_oscillating_droplet_m2919_lineage.md` | new |

So the delta against main is additive only — nothing of main is reverted. Rebasing onto
main is what removes **R1**: the branch tree now inherits main's `exclude = [".scratch"]`,
and the PR merge ref will be computed against `7d91b89` instead of the frozen `24cd6aeb`,
which also removes the three-file conflict (`run.py`, `test_oscillating_droplet.py`,
`docs/verification_oscillating_droplet.md`) that made the PR unmergeable against current main
(verified: `git merge origin/main` on the old head conflicts in exactly those files).

## 4. Verification performed here

* `python -m ruff check benchmarks/oscillating_droplet/{run,make_doc}.py tests/test_oscillating_droplet.py`
  → `All checks passed!` (ruff 0.15.0, repo config).
* CI's tiny end-to-end step reproduced: `python examples/droplet_impact_2d/run.py --steps 3 --output …`
  → `particles=608 max_speed=0.012 m/s mach=0.001 courant=0.246`, writes
  `diagnostics.csv` + `run_manifest.json` (the two files the workflow asserts on).
* `pytest tests/test_oscillating_droplet.py` — the model-side tests (Laplace dispersion,
  Rayleigh–Lamb coefficients, mode estimators, damped-cosine fit, zero-crossing, peak decay,
  lobed geometry, 2-D/3-D gate agreement, the m2919 property set) pass; the two
  simulation-heavy tests (`…_frequency_deficiency_is_reproduced` via the `release_case`
  fixture and the `sigma = 0` negative control) did not finish inside this sandbox's
  wall-clock budget under the current machine load (single CPU node shared by ~40 agent
  jobs); they had already been observed running for several minutes without failing. They
  are unverified here, **not** passed.

## 5. What is *not* fixed (honest FAIL on the "CI green" criterion)

* **R2 is untouched.** Making the Lint step green requires a repository-wide sweep of the
  ~113 pre-existing errors in 18 files that m3933 never touched. That is a separate
  "lint-debt clearance" mission; it cannot be smuggled into this PR without editing
  unrelated files, which this mission's rules forbid.
* Consequence: PR #187 will **still fail the Lint step** after this branch lands, because
  the gate lints the whole tree, and the whole tree is currently dirty on main itself.
* Local ruff is 0.15.0 while CI installs 0.17.3; the newer version flags 16 more errors on
  the same tree (measured by diffing CI's per-file counts against local ones). PyPI is not
  reachable from this sandbox (`pip install ruff==0.17.3` fails to build), so a local
  "green" cannot be certified to equal CI "green" for the debt sweep.
* Local torch is 2.7.1, CI installs 2.14.1 — numerical tests were not re-run on the CI stack.

## 6. Recommended follow-up

1. One dedicated mission: pin `ruff==0.17.3` in `[project.optional-dependencies].test`,
   run `ruff check --fix` (20 auto-fixable), then wrap the `make_doc.py` string literals
   and add the missing final newlines (11 files) → 0 errors, verified with the pinned version.
2. Re-run PR #187 on top of that; its remaining diff is purely additive, so it should pass.
3. Governor: stop merging on `cancelled` checks (this is what let 113 errors into main).