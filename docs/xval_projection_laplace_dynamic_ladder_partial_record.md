# A dead shard used to kill the deliverable — partial dynamic-ladder records

**Anchor** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m22611**
**Type** defect fix + handoff · **Verdict** the render path is fixed; the *physics* debt
(dynamic ladder unmeasured) is **unchanged and still WITHHELD** — this page does not claim it closed.

## ① Why this page exists

The dynamic arm of the anchor is run as **7 independent jobs**
(`benchmarks/laplace_sph/run_dynamic_ladder.py`): the finest level is a 120-step march on ~7 k
particles with a PCG solve per step, so the harness was deliberately built *per job*, and its
module docstring states the contract:

> Jobs may be run in any order and on any host: the record is per-job, so a shard that dies
> leaves the others usable.

That contract held for the summariser (`--summarize` merges whatever is present, and a side
diagnostic in the same directory no longer kills it — see §②b of the debt page) but **not** for
the doc generator. Rendering the deliverable from a record that has fewer than two levels with a
strictly positive spurious-current reading raised, from matplotlib's log locator:

```
ValueError: Data has no positive values, and therefore cannot be log-scaled.
```

So the one situation the per-job design exists for — *a partial record* — was exactly the
situation in which **no page and no figure could be produced at all**. A unit that ran 1 of 7
jobs (e.g. because the queue stalls, as it did in m22611: see §④) got nothing for its compute.

## ② What was changed

`benchmarks/laplace_sph/make_doc_dynamic_ladder.py`:

| site | before | after |
|---|---|---|
| panel (b), `max|u|/v_ref` vs `R/dx` | always `loglog` | `loglog` only when ≥ 2 finite positive points (`_positive_curve`), else linear with the title marked *partial record* |
| panel (b), fitted-order guide line | drawn whenever the order was finite | drawn only when the log axis is valid **and** the order is finite |
| panel (d), `nu` sweep | always `xscale("log")` | log only when the sweep is non-empty (an empty log axis is the second half of the same crash) |

No number, gate, threshold or table cell is affected: the generator still writes only what the
JSON contains, and an unmeasurable cell is still `—`.

## ③ Verification

`tests/test_xval_projection_laplace_dynamic_ladder_partial.py` (new, 3 tests, all green):

| test | asserts |
|---|---|
| `test_partial_record_renders_a_figure_and_a_page` | a 1-of-7 record renders a non-empty PNG **and** a page (used to raise `ValueError`) |
| `test_partial_record_invents_no_number` | every declared gate still appears on the partial page, a dash marks unmeasurable cells, and every gate the record cannot pass is marked **FAIL** — a 1/7 record is not allowed to read green |
| `test_full_record_still_uses_log_axes` | `_positive_curve` is `True` for a real ladder and `False` for a single level, an exact-rest zero or a NaN — the fix cannot silently degrade the complete deliverable |

Run:

```bash
PYTHONPATH=src python -m pytest tests/test_xval_projection_laplace_dynamic_ladder_partial.py -q   # 3 passed
```

The gate suite of the *complete* record (`tests/test_xval_projection_laplace_dynamic_ladder.py`,
`…_debt.py`) is untouched and still skip/strict-xfail: no record was fabricated into the tree.

## ④ The ladder itself: submitted, not measured

The 6 missing jobs (`dx12_nu01`, `dx16_nu01`, `dx24_nu01`, `dx16_nu003`, `dx16_nu03`,
`dx12_horizon`) were submitted to the cluster as **`cj4254`** (6 round-robin shards,
`--threads 16`, 21 000 s cap, `stage_dir` = this worktree). At the time of writing it is still
`queued`: the scheduler daemon is alive but the queue holds 27 jobs with **0 running**, so no
shard has started. The per-job record design means any later unit can run any subset:

```bash
PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --job dx16_nu01 --threads 16
PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic_ladder.py --summarize
PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic_ladder.py
```

## ⑤ Error statement (the `< 3 %` question)

* Key quantity: `dp` vs analytic `sigma/R` = 3.6000 Pa. **This page measures nothing new**, so it
  makes no accuracy claim. The only number in the tree remains the `R/dx` = 8 level,
  `dp` = **+1.17 %** (admissible) with `max|u|/v_ref` = **0.263** against Adami's 0.1 line (**FAIL**).
* **Is it `< 3 %`?** At `R/dx` = 8, yes. For the anchor's dynamic arm as a whole: **not
  answerable** — 6 of 7 jobs are unmeasured, so no convergence order exists. Recorded as
  **WITHHELD**, not as pass.

## ⑥ Things to watch

* The crash was in *rendering*, not in *physics*: a green test suite that skips on a missing
  record hides both. The strict-xfail marker in `…_debt.py` is what makes "record absent" a
  counted outcome; keep it until `results_dynamic_ladder/summary.json` actually lands.
* Do not "fix" a stalled queue by lowering `STEPS_PRIMARY` or `LEVELS`: the gates
  (`spurious_decay_order_min` in particular) are functions of the declared ladder, and both are
  copied verbatim into `summary.json` for provenance checking.
