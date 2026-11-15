# Similarity-corrected FSI2 run: the measurement that `SIMILARITY_AND_REFERENCE_AUDIT.md` left open

**Anchor**: `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`
**Mission**: m21375 · **Repo**: TensorSPH · **Branch**: `agent/m21375`
**Driver**: `benchmarks/xval_fsi2/run_similarity_corrected.py`
**Tests**: `tests/test_fsi2_similarity_corrected_run.py` (5 tests)

---

## 1. What hole this closes (and what it does not yet)

Two previous rounds end with the same explicit open item:

* `docs/xval_fsi2/SIMILARITY_AND_REFERENCE_AUDIT.md` §5, last row:
  *tip amplitude, similarity-corrected case | 0.0082 m | "by construction |
  group only, **not yet run**"*;
* `docs/xval_fsi2/COUPLED_BENCHMARK_REPORT.md` §5: *"❌ Full long-time
  integration (t_end = 10 s) — requires cluster submission"*, with the only
  executed probe being 40 steps (t ≈ 0.04 s).

That is a **measurement** gap, not a modelling gap: the similarity-corrected
case (`U_mean = 0.2 m/s`, `Re = 100`, `Ae = 5600`, `rho_s/rho_f = 1`,
`E = 2.24e5 Pa`) is the parameter point at which Turek-Hron amplitudes are
comparable at all, and it had never been integrated past the inlet ramp.

This round delivers the **instrument** for that measurement — a restartable
driver with predeclared gates — and starts the long integration on the
scheduler. The amplitude numbers themselves are **not** in this document yet;
see §6 for the honest status.

## 2. Why the run is expensive (the reason it was never done)

| quantity | value | consequence |
|---|---|---|
| beam first bending period | `1.8751²/(2π)·√(EI/ρA L⁴)` → **2.53 s** | ≥ 3 periods ⇒ `t_end ≥ 8 s` |
| acoustic time step `0.25 h/c0` | **1.04e-3 s** (dx = 0.02 m, c0 = 5 m/s) | ≈ 7 700 steps for 8 s |
| interface sub-iterations per step | 1 (machine-floor residual at start) → **15–20** once the interface is nonlinear | ≈ 0.9 s/step measured |
| wall clock for 8 s | ≈ 2 h single process | exceeds one queue slot → **checkpoint/resume required** |

The measurement window is therefore *predeclared* as
`t ≥ max(0, inlet_ramp_time) + 2·T₁ = 6.07 s` by default (`--t-measure-start`),
because half of the first beam period is still inlet ramp and a half-range
amplitude taken over less than one period is not an amplitude.

## 3. Predeclared decision, metric, thresholds, stopping rule

Fixed in the driver docstring and in the JSON payload **before** any result was
inspected:

* **Decision** — is the beam-tip amplitude of the similarity-corrected case
  within the owner 15 % standard of Turek-Hron FSI2?
* **Metric** — half-range `(max−min)/2` of `tip_x`, `tip_y` over the measurement
  window; physical units [m]; reference `fsi2_reference.BENCHMARK_TIP`
  (`tip_x = 0.0028 m`, `tip_y = 0.0082 m`).
* **Thresholds** — `|rel| ≤ 15 %` → `pass_15` (acceptance), `|rel| ≤ 3 %` →
  `pass_3` (reporting target).
* **Stopping rule** — integrate to `--t-end`; abort on a non-finite value or a
  coupling abort and report it. No threshold, window or resolution is retuned
  after seeing the result.
* **Similarity gate first** — the driver asserts
  `fsi2_similarity.similarity_report(case)["passed"]` before the first step, so
  a mis-parametrised case fails in milliseconds instead of after two hours.

## 4. What is verified this round

### 4.1 Instrument checks (all green)

```
python3 -m pytest tests/test_fsi2_similarity_corrected_run.py -q
5 passed in 10.94s
```

| test | invariant |
|---|---|
| `test_beam_first_frequency_matches_closed_form` | helper reproduces an independently typed `1.8751²/(2π)√(EI/ρA L⁴)`; `T₁ ∈ (1.5, 4) s` pins the window scale |
| `test_window_stats_half_range_and_start_cut` | half-range/mean on a synthetic sine, with the `t ≥ t_start` cut and the degenerate zero series (no NaN) |
| `test_dominant_frequency_recovers_injected_sine` | spectral read-out recovers 0.4 Hz to 5 % |
| `test_checkpoint_round_trip_is_exact` | checkpoint restores `beam_nodes`, `beam_vel`, `fluid_vel`, `state.time` **bitwise** — the property a chained run needs |
| `test_short_coupled_run_is_finite_and_advances_time` | 12 strong-coupling steps at dx = 0.05 m are finite, `state.time` advances by exactly `12·dt`, similarity gate passes |

### 4.2 Observed on the checks that concern the coupled path

The 40-step probe in `COUPLED_BENCHMARK_REPORT.md` reported the strong path as
"converged in 1 iteration/step". That is reproducible but only for the first
steps, and the reason is visible in the residual history: the interface
residual is at the machine floor there (`||r|| < 1e-12·L`), so the step exits
on `abs_tol`, not on a converged interface. Once the flow develops the same
case needs 15–20 IQN-ILS iterations per step (recorded in
`coupling_iterations`, 0.62 → 1.29e-4 residual reduction). Reading the early
"1 iteration" as evidence of a cheap strong coupling is what made the 8 s run
look affordable; it is not.

## 5. How to reproduce

```bash
# instrument tests
python3 -m pytest tests/test_fsi2_similarity_corrected_run.py -q

# short smoke (≈1 min, startup transient only -- not a physics result)
python3 benchmarks/xval_fsi2/run_similarity_corrected.py --dx 0.02 --t-end 0.06 \
    --t-measure-start 0.0 --tag smoke

# the measurement (chained; each chunk resumes from the checkpoint)
python3 benchmarks/xval_fsi2/run_similarity_corrected.py --dx 0.02 --t-end 8.0 \
    --ckpt artifacts/xval_fsi2/ckpt_dx0.02.pt --ckpt-every 100
python3 benchmarks/xval_fsi2/run_similarity_corrected.py --dx 0.02 --t-end 16.0 \
    --ckpt artifacts/xval_fsi2/ckpt_dx0.02.pt --ckpt-every 100 --resume
```

Outputs (program-generated, nothing hand-typed): `artifacts/xval_fsi2/
similarity_corrected_<tag>.{json,csv,npz}` — amplitudes, gate rows, per-step
`tip_x/tip_y/drag/lift/max_speed/coupling_iterations/coupling_residual`.

## 6. Error judgement — honest status

| quantity | reference | this round | relative error | < 3 % ? |
|---|---|---|---|---|
| `tip_x_amplitude` | 0.0028 m | **not measured** (8 s run queued) | — | **no basis** |
| `tip_y_amplitude` | 0.0082 m | **not measured** (8 s run queued) | — | **no basis** |
| drag mean / lift amplitude | 3.2 / 0.22 | **not measured** | — | **no basis** |

The only numbers produced so far are from the 0.06 s startup transient
(`tip_y = −1.9e-5 m`, gate `−99.93 %`): that is the inlet ramp, **not** a
physics result, and it must not be quoted as one. Status =
**WITHHELD**: the instrument is verified, the measurement is submitted as
cluster job **cj3392** (8 s ≈ 7 700 steps, 8 cores, checkpoint every 100 steps,
`artifacts/xval_fsi2/ckpt_dx0.02.pt`). A follow-up mission should
`run_compute(action="fetch", job_id="cj3392")`, then read the `.json` and only
then fill the three rows above — resuming from the checkpoint if the slot was
cut at the timeout.

## 7. Notes, limitations, scope

1. **Resolution.** `dx = 0.02 m` gives 5 particles across the cylinder diameter
   (`D = 0.1 m`). At `Re = 100` that under-resolves the attached boundary layer
   and the shedding wavelength; an amplitude within 15 % at this resolution
   would be a pleasant surprise, a large error would *not* by itself condemn the
   solver. The honest ladder (dx = 0.02 → 0.01 → 0.005) is what the checkpoint
   format now makes affordable; only the first rung is budgeted here.
2. **`rho_s/rho_f = 1` is the hard point.** Loose coupling at that ratio is
   known to produce unphysical displacement (0.56 m in the 40-step probe). Only
   `strong-iqn-ils` (or Aitken) is meaningful; the driver defaults to IQN-ILS.
3. **`--ratio 10` is a different physics problem.** The OSS regression point
   (`Ae = 1.4e3`, `rho_s/rho_f = 10`) is reachable for reproduction runs and is
   flagged in the payload; it must not be compared to `BENCHMARK_TIP`.
4. **Reference provenance.** `BENCHMARK_TIP`/`BENCHMARK_FORCES` are the
   repository's own quotation of Turek & Hron (2006); the paper has still not
   been consulted from this work tree (unchanged caveat 1 of the similarity
   audit). The ±15 % gate is only as good as that transcription.
5. **Checkpoint identity.** `--resume` refuses a checkpoint written with a
   different `dx`/`c0`/`ratio`/solver/u_mean (`checkpoint/case mismatch`), so a
   chained run cannot silently mix discretisations.
6. **Nothing in TensorPlatform or the ship platform was touched** (2026-11-02
   control order); all changes are inside `TensorSPH`.