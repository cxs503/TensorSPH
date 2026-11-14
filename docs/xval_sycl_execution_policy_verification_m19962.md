# Verification of `XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE`

**Independent re-measurement and adversarial review of the shipped deliverable.**
Anchor: `XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE`
(`Xiangyu-Hu/SPHinXsys`, the *sycl-execution-policy-not-transferable* practice).
Reviewed tree: TensorSPH `main` at `e915ab1` (deliverable merged as `e803ff1`,
"Merge #644 (m17598)"); review branch `agent/m19962`.

Verdict: **approve with notes** — the delivered measurements reproduce, but the
central claim was stated without its precondition. The precondition is now
measured, pinned by tests, and corrected in the deliverable text; no shipped
algorithm was changed.

---

## 1. What was reviewed

| artefact | role |
|---|---|
| `src/tensorsph/execution_policy.py` | port of the policy layer (tags, frozen main policy, non-copyable `ExecutionInstance`, `getUniformNdRange` replica, padded block iterator) |
| `src/tensorsph/wcsph/swept.py` | reference arm (vectorised `index_add`) and ported arm (padded masked sweep) |
| `tests/test_execution_policy.py` | 117 assertions |
| `benchmarks/xval_sycl_execution_policy/run.py` | measurement harness (`results.json`, 4 figures) |
| `scripts/xval_execution_policy/render_report.py` | report renderer |
| `docs/xval_sycl_execution_policy_not_transferable.md` | the deliverable's report |

Independence limits (stated up front):

* one host, one architecture (loongarch64, torch 2.7.1, float64) — no
  cross-platform float comparison was possible;
* no SYCL device and no `icpx` toolchain, so the OSS side is still **read, not
  run**; this review re-checked *behaviour of the port*, not the OSS binary;
* the OSS checkout the deliverable pinned (`/tmp/sxs2/...`) no longer exists on
  this host, so the 11 source facts could not be re-hashed (see F2);
* I did not author the reviewed code, and the review harness
  (`benchmarks/xval_sycl_execution_policy/verify_convention.py`) was written
  independently of `run.py` — but both are mine-vs-shipped inside one repo, so a
  shared blind spot in the *problem framing* cannot be excluded by construction.

## 2. Checks performed

| id | check | how | result |
|---|---|---|---|
| C0 | test suite green | `python -m pytest tests/test_execution_policy.py -q` | 117 passed before the review (127 after, incl. 10 new) |
| C1 | deliverable reproduces | `python benchmarks/xval_sycl_execution_policy/run.py --quick` | `worst_density_rel_err = 0.0`, `worst_control_rel_err = 43.186`, `all_gates_pass = true` — matches the report's 0.00e+00 / 43.2 at `local_size=64` |
| C2 | uniform-`h` bit-exactness, re-derived | own harness, 10 work-group sizes | **ULP distance 0** at every size; `torch.equal` true (S1) |
| C3 | third opinion | scalar per-particle loop (no `index_add`) | 5.69e-16 vs both arms |
| C4 | invariance under a **spatially varying** `h` | own row-`h` reference vs sweep, 3 resolutions × 7 amplitudes × 4 work-group sizes | bit-exact everywhere (S2b) |
| C5 | convention gap (sweep vs shipped symmetric-`h` reference) under varying `h` | same grid | 0 % → 44.16 %; breaches the owner 15 % gate at coarse resolution (S2a) |
| C6 | negative control | mask removed at `local_size` 3/64/256 | 0.644 / 36.037 / 36.037 — control is load bearing |
| C7 | negative-control precondition | isolated last swept row | control collapses to exactly 0 (vacuous) |
| C8 | edge cases | `bound=0`, `bound>n`, empty pair list, `n < local_size` | self-terms only / `ValueError` / self-terms / exact |

Reproduction (all commands on the review branch):

```bash
python -m pytest tests/test_execution_policy.py -q
python benchmarks/xval_sycl_execution_policy/run.py --quick
python benchmarks/xval_sycl_execution_policy/verify_convention.py
# -> artifacts/xval_execution_policy/verify_m19962/{results.json,summary.md,fig_*.png}
```

Case for C2–C8: lattice box `[0,0.4] × [0,0.2] m`, `dx = 0.02 m` (200 particles),
quintic kernel, `h = 1.3 dx (1 + amp sin 2πx/L)`, `rho0 = 1000`, static snapshot
(no time integration, so Mach/CFL are undefined for this measurement).

## 3. Findings, severity ranked

### F1 — **High, claim correctness** — the equivalence claim has a precondition that the deliverable does not state, and it can exceed the 15 % gate

The report claims the padded-masked sweep "reproduces the shipped vectorised
summation density for *any* work-group size", and the module docstring said the
two arms differ "only in the launch geometry". Both are true only for a
**uniform** smoothing length. `summation_density_vectorised` evaluates each
unordered pair once with `h[i]` and adds the same `w` to both rows (symmetric);
the sweep evaluates every row with its own `h` (one-sided) — which is what the
OSS device kernel necessarily does, because a work item knows only its own row.
With `h` varying, the two arms are different discrete operators:

| dx [m] | n | amp 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 50 | 0.00 | 2.02 | 4.12 | 6.20 | 11.34 | **22.34** | **44.16** |
| 0.02 | 200 | 0.00 | 1.27 | 2.52 | 3.75 | 5.88 | 8.14 | 12.07 |
| 0.01 | 800 | 0.00 | 0.67 | 1.33 | 1.98 | 3.26 | 4.47 | 5.77 |

(max relative density difference, %; own harness, `local_size = 64`; bold values
exceed the owner gate). At the realistic adaptive-`h` modulation of ±10 % the
gap is 0.67–2.02 %, i.e. it passes the 15 % gate but is **four orders of
magnitude above** the 1e-12 target the same report declares for a
numerics-neutral rewrite. At `dx = 0.04 m`, `amp = 0.9` it reaches 44.16 % and
would have passed a 15 % gate unnoticed.

Reproduction: `python benchmarks/xval_sycl_execution_policy/verify_convention.py`,
stage S2; figures `fig_convention_gap_grid.png`,
`fig_convention_field_map.png`.

Remediation (in this commit): (i) `swept.summation_density_vectorised_row_h`
added — the vectorised form of the per-row operator, so the invariance claim can
be tested under varying `h` (it is bit-exact, see F1b); (ii) the module docstring
and the deliverable report now state the precondition and quantify it;
(iii) `tests/test_execution_policy.py` section 5 pins uniform-`h` equality, the
varying-`h` invariance, the growth of the gap, and the gate breach.

**F1b (sub-finding, good news).** The *launch-geometry* invariance itself is not
an artefact of uniform `h`: against the convention-matched reference the sweep
is bit-identical at `local_size ∈ {1, 8, 64, 100}` for every amplitude up to
`h/h0 ∈ [0.10, 1.89]`. So the padded-masked launch is sound; what was overstated
is the comparison target.

### F2 — **Medium, reproducibility** — the OSS provenance is unpinned in this environment and the artefacts are not in the repo

`run.py --quick` reports `"checkout_available": false, "facts_found": 0,
"facts_total": 11`: none of the 11 source facts the "not transferable" argument
rests on could be re-located here, because the pinned checkout
(`/tmp/sxs2/sphinxsim/sphinxsys`) is gone. `artifacts/` is in `.gitignore`
(line 9), so the `provenance.json`, `results.json` and the four figures the
report cites are **not retrievable from the repository** — only from a rerun.
The degradation is honest (it reports the failure instead of substituting prose),
and the report already says the OSS side was read, not run; but a reader of the
merged tree cannot re-verify the pins.

Remediation (recommended, not done — it needs the checkout or a hash list in the
repo): commit a `docs/xval_sycl_execution_policy_provenance.md` holding the
OSS commit id + per-file sha256, or commit the figures under `docs/figures/`.

### F3 — **Medium, test design** — the negative control has an unstated precondition

The control ("drop the tail mask and the numbers break") depends on the row the
padded work items are clamped onto having neighbours. With an isolated last row
the unmasked contribution is exactly zero and a *missing* mask would pass
unnoticed; measured 0.0e+00 in C7. The benchmark avoids it (dense column), the
report mentions the mechanism, but no test recorded it — so a future refactor
that thins the tail would silently invalidate the control.
Remediation: `test_negative_control_has_a_precondition` added.

### F4 — **Low, documentation** — docstring overstated the arms' difference

`swept.py`'s module docstring said the arms differ "only in the launch geometry".
Corrected; the sentence now names the `h` convention and points here.

### F5 — **Low, coverage** — `bound` edge cases were untested

`bound=0` silently returns self-terms for *every* row (particles outside the
launch window included) and `bound>n` raises. Both behaviours are documented but
were unasserted. Remediation: `test_bound_edges` added.

### F6 — **Info, isolation risk** — the suite mutates the process-wide singleton

`test_device_clamp_sets_oss_default_work_group` and
`test_default_work_group_size_is_used` write to `execution_instance`, one of them
restoring `64` only through `try/finally`. Today the restore value equals the
clamp, so no test depends on order; adding a test that assumes a different
default would break under `pytest-randomly`. Worth a fixture, not a blocker.

## 4. Checks **not** run

| check | status |
|---|---|
| OSS SYCL build / run (`icpx`, device) | **not_run** — no SYCL device or toolchain on this host |
| Re-hash of the 11 OSS source facts | **not_run** — checkout absent (F2) |
| Full (non-`--quick`) benchmark, `dx` axis × 11 work-group sizes × 3 timings | **not_run** — deliverable's own numbers at the base resolution were reproduced via `--quick`; the `dx` axis was replaced by my own S2 grid |
| Cross-architecture float comparison (x86/GPU) | **not_run** — single host |
| Wall-clock comparison on a real accelerator | **not_run** — the CPU timings are a host artefact, as the report itself says |
| Concurrency/thread-safety of `ExecutionInstance.get_instance()` | **not_run** — GIL makes the race effectively unreachable in CPython; not exercised |

## 5. Overstated claims (as merged)

1. "reproduces the shipped vectorised summation density for **any** work-group
   size" → true for uniform `h` only (F1). *Corrected in the report and docstring.*
2. "the agreement is not a shared-implementation artefact" → correct, but the
   third opinion shares the *symmetric* convention, so it cannot detect F1; it
   validates the launch, not the operator identity. *Now separated: F1b.*
3. "the tests pin the invariant" → they pinned it for one `h`. *Ten assertions
   added.*

## 6. Disposition

**Approve with notes.** No shipped numerical path was found wrong: the
padding/mask discipline, the round-up replica and the non-transferability
constraints all hold under independent re-measurement (C2–C4, C6). The blocking
issue is a *scope* defect in the claim and in the tests, not a wrong result; it
is remediated here (code docstring, deliverable text, 10 new assertions,
`fig_convention_gap_grid.png`). F2 remains open as a reproducibility debt that
needs the OSS checkout or a committed hash list.

## 7. Error judgement against the acceptance criterion (`<15 %`)

| quantity | baseline | error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| sweep vs shipped reference, uniform `h`, `local_size` 1…128 | vectorised `index_add` | 0.00e+00 (ULP 0) | yes | yes |
| sweep vs scalar per-particle loop | independent scalar loop | 5.69e-16 | yes | yes |
| sweep vs convention-matched reference, `h` modulated ±50 %, dx 0.02 | row-`h` vectorised | 0.00e+00 (ULP 0) | yes | yes |
| convention gap, `h` modulated ±50 %, dx 0.02 | shipped symmetric reference | 5.88 % | **no** | yes |
| convention gap, `h` modulated ±10 %, dx 0.04 | shipped symmetric reference | 2.02 % | yes | yes |
| convention gap, `h` modulated ±90 %, dx 0.04 | shipped symmetric reference | **44.16 %** | **no** | **no** |
| negative control (mask removed, dx 0.02) | shipped symmetric reference | 36.04 (3604 %) | no (must fail) | no (must fail) |

The last-but-one row is the answer to "is the shipped claim within the standard":
for the physically operating range it is, and for the corner the deliverable's
wording covers it is not, which is exactly why the wording was narrowed.