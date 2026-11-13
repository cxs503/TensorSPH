# Resolution ladder for the transport-velocity correction (round 2)

Anchor: `XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION` (Xiangyu-Hu/SPHinXsys
transport-velocity correction, ported in `src/tensorsph/transport_velocity_correction.py`).

Round 1 (`README.md` in this directory) graded the operator **REAL** but left a
`WITHHELD` claim behind:

> "improves convergence" — needs the `dx = 0.02/0.01` ladder, which is queued
> (cluster jobs `cj1623`, `cj1624`) and had not run when this report was written.

This round runs that ladder (four levels, `dx = 0.04 … 0.01`, dense time
sampling) and answers the claim. **The answer is negative for the claim as
stated, and positive for the accuracy standard at the grids that matter** —
both results are reported, the negative one is not softened.

## 1. What was run

| item | value |
|---|---|
| case | Martin & Moyce (1952) dam break, `a = 0.2 m`, `2a = 0.4 m`, dry bed, tank 2.0 x 0.75 m |
| solver | in-repo WCSPH (`src/tensorsph/dam_break.py`), `dt` from `dt_from_courant()` |
| resolutions | `dx = 0.04, 0.03, 0.02, 0.01 m` (3 doublings of the column particle count: 5 -> 20 across `a`) |
| simulated | 0.45 s, `tau = t sqrt(2g/a) -> 4.46` (experiment window ends at `tau = 4.294`) |
| variants | `baseline` (no correction), `tvc_a010_all` (`alpha = 0.10`, the round-1 calibration), plus an `alpha` sweep at `dx = 0.01` (`0.10 / 0.15 / 0.30`) |
| sampling | 240 time-history samples per run (`--output-samples 240`) so the level-to-level differences are not sampling-limited |
| command | `python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.04 0.03 0.02 0.01 --variants baseline tvc_a010_all --seconds 0.45 --output-samples 240 --out docs/xval_sphinx_tvc/data/ladder_dense.json` |
| analysis | `python benchmarks/xval_sphinx_tvc/convergence.py --inputs docs/xval_sphinx_tvc/data/grid_dx004.json docs/xval_sphinx_tvc/data/grid_dx003.json docs/xval_sphinx_tvc/data/ladder_dense.json` (the `dx = 0.02` rows are in `ladder_dense.json`; the earlier 7-variant `dx = 0.02` payload lived in `artifacts/`, which is `.gitignore`d, so only the committed ladder is the record) |

Gate: owner standard, mean relative error against the experiment **< 15 %**
(`OWNER_GATE_REL`), and the ledger absolute gate `MAE < 0.15` in `Z/a`.

## 2. Error against the experiment (digitised `Z/a(tau)`, +-0.05 uncertainty)

| `dx` [m] | baseline MAE | baseline rel. | TVC `alpha=0.10` MAE | TVC rel. | TVC gate (<15 %) |
|---|---|---|---|---|---|
| 0.04 | 0.591 | 46.8 % | 0.147 | 17.1 % | FAIL (marginal) |
| 0.03 | 0.621 | 51.0 % | **0.142** | **10.6 %** | **PASS** |
| 0.02 | 0.576 | 41.4 % | 0.156 | 12.5 % | **PASS** |
| 0.01 | 0.572 | 38.7 % | 0.280 | 16.4 % | FAIL |

Raw: `docs/xval_sphinx_tvc/data/ladder_dense.json` (per-`tau` rows, mass drift,
density compression, Courant peak, clumping metrics, environment).

* The **baseline** improves monotonically but saturates: 46.8 -> 38.7 % over a
  3x refinement, i.e. the residual is dominated by *model* error (front
  definition, 2-D vs the experiment's side-wall-free channel), not by `dx`.
* The **correction** is 2.4-4.8x better than the baseline at *every*
  resolution, and it meets the owner standard at `dx = 0.03` (10.6 %) and
  `dx = 0.02` (12.5 %). It does **not** improve monotonically: at `dx = 0.01`
  the same `alpha = 0.10` is back to 16.4 %.

## 3. Why the fixed-coefficient ladder is not convergent (measured + derived)

The per-application shift measured in the runs is set by
`|delta r| = alpha h^2 |grad-1|`, and the runs give `max shift/h = 0.155`
**at every resolution** for `alpha = 0.10` (`h = 1.3 dx`), i.e. the per-step
shift is resolution independent. But a run of the same *physical* duration has
`N = T/dt` applications with `dt ~ dx` (Courant), so the cumulative correction
rate scales as

```
|delta r| / dt  ~  alpha h^2 |grad-1| / dt  ~  alpha * dx        (h ~ dx, |grad-1| ~ 1/dx, dt ~ dx)
```

so **at fixed `alpha` the physical regularisation weakens as `dx` decreases**.
That is exactly what the ladder shows: `dx = 0.01` behaves like a
*weaker-corrected* run (16.4 %, closer to the `dx = 0.02, alpha = 0.05` result of
19.8 % than to `alpha = 0.10`).

The obvious repair — keep `alpha * dx` constant (`alpha = 0.30` at `dx = 0.01`,
`0.20` at `0.02`, `0.10` at `0.03`) — was tested at `dx = 0.01`:

| `dx = 0.01` | `alpha` | MAE | rel. | max shift/h |
|---|---|---|---|---|
| 0.01 | 0.10 | 0.280 | 16.4 % | 0.155 |
| 0.01 | **0.15** | **0.193** | **14.3 %** | 0.232 |
| 0.01 | 0.30 (constant-rate scaling) | 0.410 | 33.2 % | 0.464 |

`docs/xval_sphinx_tvc/data/sweep_dx001_alpha.json`.

* The optimum does move **up** with refinement (`0.10` at `dx = 0.03` ->
  `0.15` at `dx = 0.01`), which is the qualitative content of the scaling
  argument;
* but it moves by a factor 1.5 over a 3x refinement (`alpha_opt ~ dx^-0.37`),
  **not** the `dx^-1` of the constant-rate argument: the strong form of the
  hypothesis is **falsified** (at `alpha = 0.30` the correction overshoots and
  the run loses more than the baseline gains);
* the window is bounded above by stability, not by accuracy: `shift/h` grows
  linearly with `alpha` (0.155 / 0.232 / 0.464), and beyond ~0.3 the explicit
  position correction is a positive feedback loop (`README.md` (6), the
  sign-failure record).

## 4. Self-convergence (no experimental input)

Cauchy differences of `Z/a(tau)` between neighbouring levels on a common `tau`
grid; `p = log(e_coarse / e_fine) / log(r)`. This leg cannot be biased by the
digitisation uncertainty.

| variant | `0.04 -> 0.03` (r=1.33) | `0.03 -> 0.02` (r=1.50) | `0.02 -> 0.01` (r=2.00) | orders |
|---|---|---|---|---|
| baseline | 0.055 | 0.258 | 0.098 | -4.47, **+1.76** |
| TVC `alpha=0.10` | 0.235 | 0.119 | 0.190 | **+1.97**, **-0.86** |

(`L2` RMS of the level difference in `Z/a`; `docs/xval_sphinx_tvc/data/convergence_ladder.json`.)

Read honestly:

* **Neither variant is in its asymptotic range at `dx >= 0.01`.** The coarse
  baseline pairs are dominated by a single escaping front particle (the front
  metric is one particle wide, see `README.md` (6)-ii), which is why the
  baseline differences are non-monotone; the finest pair does show ~1.8 order.
* The TVC pair `0.03 -> 0.02` shows second order (`p = 1.97`), then the
  `0.02 -> 0.01` pair **reverses** (`p = -0.86`) — the same signature as the
  fixed-`alpha` decay in section 3, from an independent metric that never looks
  at the experiment.
* Richardson extrapolation of the two finest levels gives a limit that is still
  37.6 % from the experiment (baseline), i.e. refining this metric further will
  **not** close the gap: the remaining error is the front-definition/model
  error, not discretisation. The TVC limit is not extrapolable (`p < 0`).

## 5. Verdict (evidence grades)

| claim | grade | evidence |
|---|---|---|
| ported operator reproduces the upstream algorithm | **REAL** | round 1, transcription test `tests/test_transport_velocity_correction.py` |
| correction improves the dam-break front at `dx <= 0.03` | **REAL** | 10.6 % / 12.5 % mean rel. error, `ladder_dense.json` |
| owner 15 % standard met | **REAL** at `dx = 0.03` (10.6 %), `dx = 0.02` (12.5 %), `dx = 0.01` only with `alpha = 0.15` (14.3 %) | sections 2-3 |
| "the ported scheme improves convergence" (round-1 `WITHHELD`) | **FALSIFIED as stated** for a fixed `alpha` | sections 3-4: non-monotone error and reversed observed order |
| constant-`alpha*dx` scaling repairs it | **FALSIFIED** (strong form) / supported as a trend | `alpha_opt ~ dx^-0.37`, `alpha = 0.30` degrades to 33.2 % |

## 6. Practical rule (what a user of the port should do)

> `alpha = 0.10` is calibrated for `dx` in `[0.02, 0.03]` and `h = 1.3 dx`.
> Refining beyond that without touching `alpha` silently *weakens* the
> correction (it acts like a smaller `alpha`); the safe direction is a mild
> increase (`~0.15` at `dx = 0.01`) plus a `tvc_max_shift` cap, and the
> `max shift/h` diagnostic must stay below ~0.3 — at 0.464 the scheme is a
> net loss. Any reuse at a different `h/dx` or `dt` law needs its own
> calibration run: the coefficient is *not* resolution independent.

## 7. Limitations (what this round does not establish)

1. **Not a numerical diff against a live SPHinXsys binary.** The image has no
   Boost/TBB, so the OSS reference still cannot be built here (round-1 friction
   2, unchanged). The operator leg is exact-transcription; the physics leg is
   against the experiment.
2. **One case, one metric.** The verdict is the dam-break front position. The
   bulk metrics in the payloads (clumping, `median nn/dx`, density compression)
   were not turned into a ladder of their own; a scheme can be
   front-equivalent and distribution-different.
3. **Four levels, one case, no repeated runs.** `dx = 0.005` (the fifth level
   that would put the `0.02 -> 0.01` pair into an asymptotic group) was not run:
   wall clock and disk (the machine hit 100 % disk during this round, which is
   also why the first `dx = 0.01` payload was lost and re-run into `/dev/shm`).
4. **The scaling argument is dimensional, not derived.** Section 3 shows the
   measured consequence (non-monotone error, reversed order, optimum moving by
   1.5x) and offers the `alpha*dx` rate argument as the explanation; the strong
   form of that explanation is falsified by its own test, so the mechanism is
   *validated qualitatively, not derived*.

## 8. Reproduce

```
# ladder (four levels, dense sampling) ~5 min single core
python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.04 0.03 0.02 0.01 \
    --variants baseline tvc_a010_all --seconds 0.45 --output-samples 240 \
    --out docs/xval_sphinx_tvc/data/ladder_dense.json

# alpha sweep at the finest grid
python benchmarks/xval_sphinx_tvc/run.py --resolutions 0.01 \
    --variants tvc_a015_all tvc_a030_all --seconds 0.45 \
    --out docs/xval_sphinx_tvc/data/sweep_dx001_alpha.json

# Cauchy / Richardson analysis + figure
python benchmarks/xval_sphinx_tvc/convergence.py --inputs \
    docs/xval_sphinx_tvc/data/grid_dx004.json \
    docs/xval_sphinx_tvc/data/grid_dx003.json \
    docs/xval_sphinx_tvc/data/ladder_dense.json \
    --variants baseline tvc_a010_all \
    --out docs/xval_sphinx_tvc/data/convergence_ladder.json

# tests (16 analysis tests + 9 operator tests)
python -m pytest tests/test_convergence_ladder.py tests/test_transport_velocity_correction.py -q
```

Figure: `docs/xval_sphinx_tvc/figure_convergence_ladder.png` (left: front
history at four resolutions vs the experiment; right: Cauchy L2 and MAE against
`dx`, log-log, with slope-1 / slope-2 references).