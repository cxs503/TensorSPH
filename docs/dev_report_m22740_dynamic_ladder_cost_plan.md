# m22740 - dynamic projection-SPH Young-Laplace ladder: cost plan and queue record

*Anchor:* `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` (TensorSPH, acceptance `< 15 %` on `dp/(sigma/R)`).
This page is a **measurement-attempt record, not a result page**: it closes nothing and claims no new
number for the anchor. The dynamic resolution ladder registered in
`docs/xval_projection_laplace_dynamic_ladder_debt.md` was *submitted* this round and **did not
execute**; the reason and the price of the job are recorded here so the next unit does not re-derive
them and does not mistake a queued job for a measured one.

## 1) What was attempted

`benchmarks/laplace_sph/run_dynamic_ladder.py` declares 7 marched-projection jobs (6 unmeasured,
`dx8_nu01` already in the tree) at fixed physics R = 20 mm, sigma = 0.072 N/m, rho0 = 1000,
h/dx = 1.3, dt = 0.1 t_sigma, CG rel_tol 1e-6, max_iter 4000. The dynamic arm is the only arm of
this anchor in state WITHHELD, and its only available evidence is negative: the one-shot solenoidal
diagnostic fitted order -1.311 against a declared minimum of +0.50, and `dx8_nu01` gives
`max|u|/v_ref` = 0.263 against Adami's 0.1 line.

Submission this round: job `cj4334`, repo TensorSPH, `stage_dir` = this worktree, `--shard {i}/7`,
16 cores per shard. It stayed `queued` for the whole round (8.5 min at the last poll; both 128-core
nodes reported ok with load 48 / 43 and 65 jobs in flight cluster-wide) and returned no record, so
the ladder is **NOT_RUN this round**.

Local execution is not an alternative: this host's guard caps interactive work at <8 cores and
10 minutes, and the fine levels need two orders of magnitude more (table below).

## 2) Method: a cost plan calibrated on the record that exists

`--plan` (new this round) prints one row per declared job: the **exact** particle count from the same
`hex_block` lattice the runner uses, the measured wall time where a record exists, and elsewhere
`t = t_cal (n/n_cal)^1.5 (steps/steps_cal) (threads_cal/threads_now)`. That exponent is an
assumption (neighbour list linear in n, PCG iteration count of a Laplacian growing like n^0.5); it is
printed with the table, and every level that lands later replaces its estimate with its own measured
wall time. Calibration: `dx8_nu01`, 780 particles, 120 steps, 284 s on 8 threads.

Independent spot check against a fresh local march (`relax_projection_consistent`, R/dx = 8, four
threads): 10 steps in 17.6 s and 20 steps in 34.7 s, i.e. 1.7 s/step, so the plan is conservative by
about 1.4x rather than optimistic.

## 3) Cost table (injected from `results_dynamic_ladder/cost_plan.json`)

| job | role | R/dx | particles | steps | record | wall [s] |
|---|---|---|---|---|---|---|
| `dx8_nu01` | ladder | 8 | 780 | 120 | present | 284 (measured) |
| `dx12_nu01` | ladder | 12 | 1733 | 120 | **absent** | 1879 (estimate) |
| `dx16_nu01` | ladder | 16 | 3090 | 120 | **absent** | 4473 (estimate) |
| `dx24_nu01` | ladder | 24 | 6853 | 120 | **absent** | 14773 (estimate) |
| `dx16_nu03` | nu_sweep | 16 | 3090 | 120 | **absent** | 4473 (estimate) |
| `dx16_nu30` | nu_sweep | 16 | 3090 | 120 | **absent** | 4473 (estimate) |
| `dx12_horizon` | horizon | 12 | 1733 | 600 | **absent** | 9393 (estimate) |

Total on the calibration thread count (8): **39747 s = 11.0 h**; on 16 threads: **5.5 h**. The single finest level
(`dx24_nu01`, 6853 particles) is the bulk of it, which is why this arm is a queued cluster job and
not an interactive measurement.

## 4) Figure

`docs/figures/m22740_dynamic_ladder_cost_plan.png`: cost per job vs R/dx (log-log) with the one
measured point, the 600 s interactive cap and the per-job cluster limit drawn in.

## 5) Notes and limits

* **No new physics number for the anchor this round.** The dynamic gates (`spurious_decay_order_min`,
  `horizon_drift_hold`, `viscosity_independence_pct`) remain unbacked; the dynamic arm stays
  **WITHHELD with negative available evidence**.
* The cost exponent is an assumption and must never be quoted as a benchmark result.
* The shard submission is left in the queue. The runner writes one record per job, so `--summarize`
  consumes exactly the jobs that land and the gate suite `tests/test_xval_projection_laplace_dynamic_ladder.py`
  (currently skipped on the missing `summary.json`) activates with no further code change.
* `cost_plan.json` sits beside the job records; the summariser's declared-job filter keeps it from
  being mis-read as a march, and that test was re-run against the new file.

## 6) Error statement

No accuracy claim is made here, so the <3 % / <15 % question is **not applicable**: every number on
this page is wall-clock seconds or a particle count. The anchor's accepted numbers (static arm dp
error +0.97 / +0.61 / -0.21 % at R/dx = 10 / 13.3 / 20, audited in `docs/audit_m17633/`) are
unchanged by this round.
