# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST / 3-D arm: why the ladder had no rungs (m22082)

Mission m22082, repo TensorSPH, branch `agent/m22082`. Companion of
`docs/kleefsman_3d_ladder_m21869.md` (which built the ladder *tool*) and of
`docs/verification_kleefsman_3d_blockage_m21728.md` (which stated the blockage
hypothesis).

## 1. What this note reports

The 3-D arm of the Kleefsman case exists to *measure* an attribution: the 2-D
model blocks 100 % of the 1.00 m tank width while the experiment blocks 40 %
(a 0.403 m box), and every 2-D verdict over-predicts the in-front-of-box peak
(frozen slab operator, dx = 0.02: 0.4768 m vs 0.370 m reference, +28.9 %). The
3-D arm asks whether restoring the third dimension closes that gap.

The previous shift (m21869) shipped the ladder tool and reported the three
production rungs as **still running**, verdict WITHHELD, "drop the recovered
results into `results/` and the verdict follows". This shift checked that claim
against the job records instead of the queue status, and it was false:

| job | command | gate status | process rc | last sample | outcome |
|---|---|---|---|---|---|
| cj3661 | `run.py --dx 0.04 --t-end 2.6 --wall-layers 3` | `succeeded` | **255** | t = 2.036 s (slab 0.131 m) | glibc abort: `free(): corrupted unsorted chunks` |
| cj3662 | `run.py --dx 0.03 --t-end 2.6 --wall-layers 3` | `succeeded` | **255** | t = 0.579 s (slab 0.053 m) | same abort |
| cj3758 | `run.py --dx 0.08 --t-end 2.6 --wall-layers 3` | `succeeded` | **1** | — (0 s compute) | `ValueError: obstacle length 0.161 m is under-resolved at dx=0.08` |

Three findings, in order of severity.

* **F1 (blocking, methodological).** The compute gate's `succeeded` / `rc 0` is a
  *dispatch* state, not a run outcome: all three jobs are recorded as succeeded
  while the processes exited 255, 255 and 1. Gating evidence on the queue status
  is therefore unsound; the artifact (or the job log's `rc` + last sample) has to
  be read. This is the exact mechanism by which "still running" was written about
  three terminated failures.
* **F2 (blocking, design).** The coarsest rung dx = 0.08 is **inadmissible by the
  case's own validator**: a 0.161 m box cannot be spanned by 3 cells of 0.08 m.
  The 3-level ladder as designed (0.08 / 0.04 / 0.03) could never have produced a
  fit, whatever the fine rungs did. The admissible floor is
  `dx <= 0.161 / 3 = 0.053667 m`, so the coarse rung is dx = 0.05.
* **F3 (non-blocking, environment).** The heap abort is not case-specific:
  82 of 2812 recorded cluster job logs (2.9 %) carry `free(): corrupted unsorted
  chunks`, 38 on n315 and 44 on n330, across unrelated cases. At least one
  identical command is documented as aborting once (cj2937) and completing on
  retry (cj3094), so the working assumption is an intermittent node/allocator
  fault rather than a defect in `kleefsman_obstacle_3d`. It is *not* proven:
  no in-process reproducer was run, and the abort points differ between rungs
  (t = 2.04 s at dx = 0.04, t = 0.58 s at dx = 0.03), which a deterministic
  numerical bug would not do.

## 2. What was changed (defect fixes, not claims)

1. `src/tensorsph/kleefsman_obstacle_3d.py`
   * `run_case(..., on_sample=cb)` — the driver can now persist every stored
     sample as it is produced. Before this, a run that died at t = 2.04 s of a
     2.6 s window left *nothing* on disk, which is why two half-hour jobs
     produced zero evidence.
   * `to_dict()["fingerprint"]` — the protocol block (`t_end`, `sound_speed`,
     `wall_layers`, `obstacle_z`, `station_offset`, `gauge_aperture`,
     `arrival_threshold`) is written by the run itself. The ladder used to
     reconstruct it from keys the record did not have, so every key was skipped
     and the comparability guard passed on an empty comparison.
2. `benchmarks/kleefsman_obstacle_3d/run.py`
   * streams `results/<tag>_time_series.csv` row by row (flushed), writes
     `results/<tag>_meta.json` **before** the first step with `complete: false`,
     and on any exception records `aborted: true` + the error before re-raising;
     on success it stamps `complete: true`, steps and simulated time.
3. `benchmarks/kleefsman_obstacle_3d/ladder.py`
   * `rung_validity(dx)` / `valid_rungs(dxs)` — rung admissibility is now decided
     by the case's own `validate_case`, and `MAX_ADMISSIBLE_DX = 0.161/3` is
     reported with the ladder;
   * `analyse()` returns `RUNGS_REJECTED` (with `inadmissible_rungs`,
     `incomplete_runs`, `usable_rungs`) instead of extrapolating through a rung
     the validator rejects or a run that never reached `t_end`;
   * `check_protocol()` counts the keys it actually compared, flags keys missing
     from some records, and fails closed (`_fingerprint`) when fewer than two
     keys are comparable;
   * completeness is inferred from the series when a record does not declare it
     (`t_last >= t_end`), so old records stay readable but truncated ones cannot
     masquerade as levels.

`tests/test_kleefsman_obstacle_3d_rung_gate.py` (new, 13 tests) pins all of the
above with invariants, not smoke tests: the submitted coarse rung is rejected;
the admissible set is bounded by the 3-cell rule from both sides; two valid rungs
are not a three-level ladder; empty / partially-recorded fingerprints do not pass
the guard; an inadmissible rung or an aborted run stops the fit instead of
entering it; a clean valid ladder does fit; `to_dict()` carries the full
fingerprint; `on_sample` hands out exactly the stored samples. The pre-existing
ladder fixture was re-based from dx = 0.08 to dx = 0.05 (it encoded the
inadmissible rung set).

## 3. Production status (honest: WITHHELD)

The three valid rungs were re-submitted from this worktree with the fixed driver:

| job | rung | command |
|---|---|---|
| cj3887 | dx = 0.05 | `run.py --dx 0.05 --t-end 2.6 --wall-layers 3 --output-every 5 --threads 8 --tag dx0.05` |
| cj3888 | dx = 0.04 | `run.py --dx 0.04 --t-end 2.6 --wall-layers 3 --output-every 5 --threads 8 --tag dx0.04` |
| cj3889 | dx = 0.03 | `run.py --dx 0.03 --t-end 2.6 --wall-layers 3 --output-every 5 --threads 8 --tag dx0.03` |

At the end of this shift all three were still queued (~34 jobs ahead, ~8 cores
each), so **no 3-D peak or arrival number is claimed here**, and the blockage
hypothesis remains undecided. What has changed is that the decision can now
actually be reached: the coarse rung is valid, a crash can no longer erase a
series, and a rejected or truncated rung can no longer be averaged into a fit.
Recovering the verdict is mechanical:

```bash
python benchmarks/kleefsman_obstacle_3d/run.py --dx 0.05 ... --tag dx0.05   # or fetch
python benchmarks/kleefsman_obstacle_3d/ladder.py --tags dx0.05 dx0.04 dx0.03
```

`ladder.py` writes `results/ladder.json` with `peak_ladder.f_inf`,
`peak_vs_reference.verdict` ∈ {`BLOCKAGE_HYPOTHESIS_SUPPORTED`,
`BLOCKAGE_HYPOTHESIS_NOT_SUPPORTED`, `NOT_DECIDABLE`} and the same for arrival.

## 4. Error accounting against the acceptance

Acceptance for the anchor is peak-height and arrival-time error < 10 % vs
Kleefsman et al. (2005) Fig. 19 (H2), reference 0.370 m / 0.3591 s.

| quantity | value | vs reference | < 10 %? |
|---|---|---|---|
| 3-D peak (any rung) | not run to completion | — | **unknown (not claimed)** |
| 3-D arrival (any rung) | not run to completion | — | **unknown (not claimed)** |
| 2-D frozen slab peak, dx = 0.02 (existing, for context) | 0.4768 m | +28.9 % | no |
| 2-D arrival, dx = 0.02 (existing) | 0.4409 s | +22.8 % | no |

No figure is included: the only 3-D series that exists is a truncated log of
`slab` samples (t ≤ 2.036 s at dx = 0.04) from a failed job, and plotting it
against the experiment would invite reading a peak that the run never reached
(the record stops 0.67 s before the experimental peak instant, so any maximum on
it is censored by construction — the same censoring rule `score()` enforces).

## 5. Lessons for the next shift

* Never accept a cluster result on the strength of `state: succeeded`; read the
  job log `rc` and the artifact. `rc 255` above a `succeeded` line is the whole
  trap.
* Check a rung against `validate_case` *before* submitting it. `valid_rungs()`
  now does this for the ladder; the same one-line check applies to any sweep.
* `benchmarks/kleefsman_obstacle_3d/results/` holds only `ladder.json`
  (`status: NO_RUNS`) plus `rung_gate.json`; no run artifact has ever existed for
  this arm.