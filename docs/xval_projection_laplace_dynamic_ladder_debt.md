> **Status update (m22962).** `dx16_nu01` and `dx12_horizon` are now in the tree (fetched
> from `cj4390`/`cj4391`) and both **diverge**: at step 3 (NaN) and step 181 respectively.
> H1/H2 are therefore decided *negatively*, not merely unbacked; `dx24_nu01` and the `nu`
> sweep still have no record. Adjudication:
> `docs/xval_projection_laplace_horizon_instability.md`.

# Evidence debt: the dynamic projection-SPH Young-Laplace ladder is not measured

*Anchor:* `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` (TensorSPH) · acceptance `< 15 %` on the
Laplace jump. This page is a **register of an open verification debt, not a result page**.
It exists because the debt was previously invisible: the gate module that should enforce the
dynamic claim is skipped, and a skipped module reports green.

## ① What the anchor claims, and which arms are actually measured

| arm | instrument | record | state |
|---|---|---|---|
| static carrier (`laplace_isph`) | pressure jump of a lattice drop, no time marching | `benchmarks/laplace_sph/results.json` | measured, reviewed (`docs/audit_m17633/`) |
| well-balanced capillary force | Helmholtz split, `relax_projection_balanced` | `benchmarks/laplace_sph/results_balanced/` | measured (`docs/xval_projection_laplace_wellbalanced.md`) |
| solenoidal diagnostic | one-shot `G^T G q = G^T f` split of the CSF traction | `results_dynamic_ladder/diagnose_solenoid.json` | measured, **H1 falsified** at `R/dx` = 8, 12 (`docs/xval_projection_laplace_solenoid_ladder.md`) |
| **dynamic resolution ladder** | marched projection, spurious-current decay order | `results_dynamic_ladder/summary.json` | **NOT MEASURED — this debt** |

The dynamic arm is the one that decides whether the residual spurious current of the
projection scheme is a lattice artefact (it must fall as `R/dx` grows) or a formulation defect
(it does not). At `R/dx` = 8 the record `results_dynamic_ladder/dx8_nu01.json` exists
(120 steps, `dp` = +1.17 %, `max|u|/v_ref` = 0.263 against Adami's 0.1 line). The remaining
levels are **not** in the tree:

| level | job | job file | state |
|---|---|---|---|
| `R/dx` = 8 | `dx8_nu01` | present | measured (single point only) |
| `R/dx` = 12 | `dx12_nu01` | absent | **not measured** |
| `R/dx` = 16 | `dx16_nu01` | absent | **not measured** |
| `R/dx` = 24 | `dx24_nu01` | absent | **not measured** |
| `R/dx` = 16, nu sweep | `dx16_nu003`, `dx16_nu03` | absent | **not measured** |
| `R/dx` = 12, 600-step horizon | `dx12_horizon` | absent | **not measured** |

Consequence, stated plainly: the H1 decay-order gate
(`spurious_decay_order_min`), the H2 horizon gate (`horizon_drift_hold`) and the H3 viscosity
independence gate (`viscosity_independence_pct`) are **unbacked**. The one-shot solenoidal
diagnostic that *was* measured points the wrong way (fitted order −1.311 against a declared
minimum of 0.50), so the honest current status of the dynamic arm is
**WITHHELD, with the only available evidence negative** — not "in progress and looking fine".

## ② Why it is still open

The ladder is 7 jobs of marching projection solves:

| job | steps | particles | measured / estimated cost on 16 threads |
|---|---|---|---|
| `dx8_nu01` | 120 | 780 | 284 s (measured, present in the tree) |
| `dx12_nu01` | 120 | 1 733 | ~0.5 h |
| `dx16_nu01` | 120 | 3 090 | ~1 h |
| `dx24_nu01` | 120 | ~7 000 | ~3–4 h |
| `dx16_nu003`, `dx16_nu03` | 120 each | 3 090 | ~2 h |
| `dx12_horizon` | 600 | 1 733 | ~2.5 h |

That exceeds the local short-test budget (< 8 cores, < 10 min), so it is a queued compute job,
not an interactive run. The job was submitted as `cj4074` (3 round-robin shards, 16 threads
each, `stage_dir` = this worktree) and was still `queued` when this page was written; the
per-job record design means a later unit can run **any subset** and keep every number already
produced.

## ②b A defect found while registering the debt: `--summarize` did not run at all

The merge step of the deliverable was broken in the shipped tree, and the absence of any record
is why nobody had noticed: `results_dynamic_ladder/` also contains
`diagnose_solenoid.json` (the one-shot Helmholtz-split diagnostic, a different record shape), and
`summarize()` globbed **every** `*.json` in the directory, so it died with

```
File "benchmarks/laplace_sph/run_dynamic_ladder.py", line 185, in summarize
    ladder = sorted([r for r in runs.values() if r["role"] == "ladder"],
KeyError: 'role'
```

Reproduced in this worktree (`PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py
--summarize`). Fixed minimally: the job list from `job_specs()` is now the authority on what a
ladder row is, and a file that is not a declared job (or lacks `role`) is skipped instead of
half-read. Regression test:
`tests/test_xval_projection_laplace_dynamic_ladder_debt.py::
test_summarize_survives_a_side_diagnostic_in_the_same_directory` builds a temporary record
directory with the side diagnostic plus one real job and asserts the merge survives and contains
only the job. This is a *harness* fix only — it produces no new physics number and does not
close the debt.

## ③ The gates that will be applied (declared before the numbers were read)

`run_dynamic_ladder.GATES_DECLARED` is copied verbatim into the record, and
`tests/test_xval_projection_laplace_dynamic_ladder.py` re-derives every verdict from the
recorded measurements, so a gate cannot be relaxed by editing the JSON afterwards:

| gate | threshold | scope |
|---|---|---|
| `laplace_all_levels_15pct` | 15 % | every (level, nu) |
| `area_drift_all_levels_2pct` | 0.02 | every run |
| `projection_exact` | 1e-05 | leftover divergence ratio, every sampled step |
| `spurious_gate_finest` | 0.1 | `max\|u\|/v_ref` at the finest level (Adami et al. 2012) |
| `spurious_decay_order_min` | 0.5 | fitted order in `R/dx` (H1) |
| `horizon_drift_hold` | 15 % | 600-step march at `R/dx` = 12 (H2) |
| `viscosity_independence_pct` | 3 % | spread of `dp` over `nu_factor` (H3) |
| `no_divergence_viscous` | 1 (true) | every damped run completes |

## ④ What this unit did instead of the ladder

1. **Made the debt machine-visible.** `tests/test_xval_projection_laplace_dynamic_ladder_debt.py`
   carries `test_dynamic_ladder_record_present` as a **strict xfail**: a missing record is now a
   counted `xfailed` outcome instead of a silent green skip, and because it is strict it turns
   the suite red as soon as the record appears, which forces the doc to be regenerated from it.
   Three further tests keep *this page* honest: it must name the reproduce command, every
   declared gate with its threshold, and every level that has no job file.
2. **Fixed a mislabelled record field.** `summary.json` shipped a single `t_sigma_s` computed
   with `h` of `R/dx` = 8 while `t_sigma` is level dependent; a reader of the record could take
   it as the capillary time of the run at hand. It is now a per-level map
   `t_sigma_s_by_level`, and the single level-8 value is gone.

## ⑤ Error statement (the `< 3 %` question)

* **Key quantity `dp` vs analytic `sigma/R`** = 3.6000 Pa. Measured on this page: one level only,
  `R/dx` = 8, `dp` = +1.17 % (from the existing job file).
* **Is it `< 3 %`?** At the single measured level, **yes** (+1.17 %). For the anchor's ladder as
  a whole the answer is **not answerable from the tree**: 6 of 7 jobs were never run, so no
  convergence statement exists. This page does not convert one coarse point into a ladder claim.
* **Spurious current:** 0.263 at `R/dx` = 8 against the 0.1 Adami line — **FAIL** at the only
  measured level, and the adjacent one-shot diagnostic measured a *negative* decay order. The
  dynamic arm is therefore recorded as WITHHELD with negative prior evidence.

## ⑥ Reproduce

```bash
# one job (any subset is useful, records are per job)
PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --job dx12_nu01
# or a shard of the 7-job list
PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --shard 1/3
# merge whatever is present, then render the deliverable from the JSON
PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --summarize
PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic_ladder.py
```

| artifact | what |
|---|---|
| `benchmarks/laplace_sph/run_dynamic_ladder.py` | 7-job harness, 8 pre-registered gates |
| `benchmarks/laplace_sph/make_doc_dynamic_ladder.py` | renders `docs/xval_projection_laplace_dynamic_ladder.md` from the record |
| `tests/test_xval_projection_laplace_dynamic_ladder.py` | the gate suite (skipped while the record is absent) |
| `tests/test_xval_projection_laplace_dynamic_ladder_debt.py` | the strict-xfail debt marker + this page's own checks |
## ⑦ Status update (m22611): the render path was broken for partial records

The per-job contract above ("a shard that dies leaves the others usable") failed at the *last*
step: with fewer than two levels carrying a positive spurious-current reading,
`make_doc_dynamic_ladder.py` died in matplotlib's log locator
(`ValueError: Data has no positive values, and therefore cannot be log-scaled`), so a partial
record produced **no page and no figure at all**. Fixed and pinned by
`tests/test_xval_projection_laplace_dynamic_ladder_partial.py`; details in
`docs/xval_projection_laplace_dynamic_ladder_partial_record.md`.

The six missing jobs were submitted as cluster job **`cj4254`** (6 shards × `--threads 16`,
`stage_dir` = the m22611 worktree) and were still `queued` when this page was updated (queue:
27 jobs, 0 running). **The gates, the thresholds and the debt itself are unchanged**: no job
file was added, no record was fabricated, and the strict-xfail marker in
`tests/test_xval_projection_laplace_dynamic_ladder_debt.py` stays in force until
`results_dynamic_ladder/summary.json` exists. The `< 3 %` answer therefore remains *unanswerable
for the ladder as a whole* — one coarse level at +1.17 % with a spurious-current **FAIL**
(0.263 vs 0.1) is not a convergence statement.
