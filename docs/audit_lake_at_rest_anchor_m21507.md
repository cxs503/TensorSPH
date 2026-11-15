# Adversarial review of `CASE-SPH-LAKE-AT-REST` evidence (m21507)

Anchor: **CASE-SPH-LAKE-AT-REST** — 算例池·静水湖面守恒（零速零波动），ref: 解析恒等，
tolerance: machine precision (`1e-9`). Ledger status at start of this review: `open`.

Scope of the review: the shipped evidence channel (`tests/data/lake_at_rest/`,
`docs/verification_lake_at_rest*.md`), the rest-horizon study added by m21337
(`benchmarks/lake_at_rest/stability.py`, `tests/data/lake_at_rest/stability.json`,
`tests/test_lake_at_rest_stability.py`), the band estimator added here, and the
manifest gate. Reviewed at commit `6b48aed3` (branch `agent/m21507`), clean tree.

Independence limitation, stated up front: this review was performed by an agent
of the same programme as the authors of the material reviewed, in the same
worktree, on the same machine. It is **not** an external audit. What it does
have is reproducibility: every finding below is a command that fails or a
number re-derived from shipped data, not an opinion.

## 1. Findings at `6b48aed3`, before any change

### F1 (blocking, evidence integrity) — the manifest gate was red at HEAD

```
$ python -m pytest tests/test_lake_at_rest_evidence.py -q
FAILED tests/test_lake_at_rest_evidence.py::test_manifest_matches_every_shipped_byte
AssertionError: unmanifested evidence files: ['channel.json', 'stability.json',
 'stability_band_fit.json', 'wall_gap.json']   (as measured; names accumulate)
```

`benchmarks/lake_at_rest/freeze.py` built `MANIFEST.sha256` from the two case
runs plus `report.json` only. Two sibling runners (`stability.py`, `wall_gap.py`)
later wrote records into the same shipped directory without re-freezing, so on
every fresh checkout the anchor's "the shipped numbers are tamper-evident" gate
failed. Consequence: the entire evidence channel for the anchor was
un-decisionable — a green-looking doc over a red gate.

Root cause: the manifest membership was hard-coded to `RUNS`, and re-freezing
re-runs the physics (bytes of the published numbers change), so nobody re-froze.

**Fix**: `EXTRA_RECORDS` declares the sibling records; they are hashed when
present, and `freeze.py --reship` re-hashes the frozen payload without re-running
the case, so the published numbers keep their bytes. The gate is now
`10 passed`.

### F2 (blocking, correctness of a stated control) — the seed-independence gate failed

```
$ python -m pytest tests/test_lake_at_rest_stability.py -q
5 failed (3 × KeyError: 'samples', 1 seed-ratio, 1 coarse re-derivation)
```

Three *different* defects were hiding behind one red file:

* **F2a schema drift.** `stability.py` moved the raw series from
  `arm["run"]["samples"]` to `arm["samples"]`; the gates and the doc generator
  still read the old path (`KeyError: 'samples'`). The shipped artefact is the
  arm-level form; consumers were never updated. Nothing in CI caught it because
  the file was simply red on `main` — a red test that everyone scrolls past is
  not a gate.
* **F2b estimator defect (the substantive one).** The study estimated the growth
  rate `lambda` by least squares over a *single absolute amplitude window*
  (`1e-10 <= max|v|/c0 <= 1e-5`) declared once for every arm. An arm that starts
  moving (the two seeded arms) enters that window at `t = dt`, so the fit
  averages the start-up transient: measured `lambda = 0.887` (`R^2 = 0.29`) and
  `lambda = 3.724` (`R^2 = 0.58`) against `16.299` (`R^2 = 1.00`) for the base
  arm. The doc's sentence "the rate is seed-independent" was therefore **not
  supported by its own fit**, and the gate that compared the two numbers
  (declared window `0.90 … 1.10`) correctly failed at `0.238`.
* **F2c false negative in the artefact-free gate.** `test_coarse_lattice_rederivation`
  ran `t_end = 0.30 s` at `dx = 0.04 m` and fitted the legacy window. At that
  resolution the departure needs ~1 s to leave the roundoff floor, so the fit
  returned a *negative* slope (`-0.118`) and the test failed for a reason that
  has nothing to do with the physics it claims to test.

**Fix (F2b, and the claim it decides).** A rate is only meaningful where the
departure is exponential. The declared band is `1e-6 … 1e-3`: seven decades
above the measured roundoff floor (`max|v|/c0 = 2.6e-14` at rest,
`max|a|/g = 1e-12 … 1e-13`) and one decade below the measured saturation
amplitude (`8.6e-3`). Re-derived from the **shipped raw series** (no physics
re-run, `benchmarks/lake_at_rest/band_fit.py`):

| arm | legacy window λ [1/s] | R² | band λ [1/s] | R² | legacy = rate? |
|---|---|---|---|---|---|
| base | 16.299 | 1.0000 | **16.470** | 1.0000 | yes |
| courant_half (dt/2) | 15.519 | 0.9989 | 16.558 | 1.0000 | yes |
| resolution dx=0.025 | 14.889 | 1.0000 | 14.916 | 1.0000 | yes |
| pairs_rebuilt | 16.391 | 1.0000 | 16.421 | 1.0000 | yes |
| seed 1e-6 dx | 0.887 | 0.2878 | 16.475 | 0.9997 | **no (transient)** |
| seed 1e-8 dx | 3.724 | 0.5753 | 16.600 | 1.0000 | **no (transient)** |
| saturation 8 s | 16.445 | 1.0000 | 16.563 | 1.0000 | yes |
| viscosity 1e-6 | 16.536 | 1.0000 | 16.634 | 1.0000 | yes |

In the band the seed ratio is `0.9925` (declared `0.90 … 1.10`) and the `dt`
ratio `1.0054` (declared `0.75 … 1.33`). **The claim survives; the estimator did
not.** This is the distinction that matters: the study's conclusion was right for
the wrong reason, and its own gate said so.

Anti-cherry-picking check (declared windows, base arm):
`1e-8…1e-4 → 16.394`, `1e-7…1e-3 → 16.446`, `1e-6…1e-3 → 16.470`,
`3e-6…3e-3 → 16.474` — spread < 1 %. The symmetric failure mode was measured
too: a window whose upper edge reaches into saturation
(`1e-5 … 1e-2`) returns `8.19`, i.e. the same bias at the other end. Both must be
kept out of the estimator, which is why the band has two edges and not one.

### F3 (non-blocking) — the horizon document never existed

`benchmarks/lake_at_rest/make_horizon_doc.py` crashed
(`KeyError: 'samples'`, same schema drift as F2a), and the artefacts it
generates figures for (`docs/figures/lake_stability_*.png`) were never written.
So the honest verdict of the m21337 study — *rest is not persistent; it decays
exponentially with a measured horizon* — was shipped as an artefact and a test
file with **no document**, which is the one place a reader looks.
Now rendered: `docs/verification_lake_at_rest_horizon.md` (+ both figures).

## 2. What the anchor's status is after this review

| statement | evidence | status |
|---|---|---|
| rest to `1e-9` over the published `t = 0.50 s`, laterally periodic tank | `tests/data/lake_at_rest/periodic_dx020.json`, gates G1–G5 | **holds** (`t* = 1.1805 s > 0.50 s`) |
| rest to `1e-9` as an *unconditional* claim | exponential departure, `λ = 16.47 1/s` (`R² = 1.0000`) | **FAIL** — honest, documented |
| wall-bounded tank | side-wall truncation `max|a_x|/g ≈ 0.12` | **FAIL** — documented, not refinable by wall pressure/mass/operator options (`docs/verification_lake_at_rest_wall_gap.md`) |
| departure is not an integrator artefact | `λ(dt/2)/λ(dt) = 1.0054` | supported |
| departure is linear (seed-independent) | `λ(1e-6)/λ(1e-8) = 0.9925` in the band | supported **now**; unsupported by the legacy fit |
| shipped numbers are tamper-evident | `MANIFEST.sha256` over 6 records | supported |

Anchor verdict: **FAIL as an unconditional machine-precision claim, PASS over the
certified horizon.** The FAIL is measured, reproducible and now decided on a
fresh checkout by gates that actually run.

## 3. Checks performed

* `pytest tests/test_lake_at_rest_stability.py` — 13 gates (was 5 failed / 5 passed).
* `pytest tests/test_lake_at_rest_evidence.py` — 10 gates, manifest tamper gate
  re-verified after re-freezing.
* Independent re-derivation of every band `λ` from the shipped raw series with the
  test file's own least squares (not the producer's).
* Artefact-free re-derivation at `dx = 0.04 m`, `dt` and `dt/2`, `t_end = 3.0 s`:
  `λ = 11.566` and `11.570` 1/s (`R² ≥ 0.99999`), ratio `1.0004`.
* Window sensitivity sweep, four declared windows plus two deliberately bad ones
  (into the transient, into saturation).

## 4. Checks NOT run (marked unknown, not passed)

* `freeze.py --verify` (full physics re-run against the frozen channel) was **not**
  executed in this session: it re-runs both case runs and the rounding-level
  quantities are only reproducible as a scale (documented spread ~1.3×). The
  manifest and gate re-grading were verified; the roundoff-digit reproduction was
  not.
* The 8 s saturation arm and the `dx = 0.025` arm were **not** re-run; their
  numbers are consumed as shipped.
* No attempt was made to identify the *mechanism* of the growing mode beyond the
  recorded dominant column wavenumber; the doc's mechanism statement stays as
  narrow as the spectrum allows.
* Local compute discipline: `load15 ≈ 56 / 128` at review time; only sub-minute
  runs were executed (the artefact-free gate above, 37 s).

## 5. Claims in the reviewed material that were overstated

1. "the rate is seed-independent" — true, but the quoted fits
   (`0.887` / `3.724` 1/s) did not show it; they contradicted it. Fixed by the
   band estimator, and the legacy numbers are retained in the doc marked as
   *not a rate*.
2. "shipped evidence is machine-decidable" — false at HEAD, because the manifest
   gate was red and the horizon document was missing.
3. The rest-horizon study was described as an "addendum to the PASS" while
   its document did not exist in the tree; the addendum is now the shipped
   `docs/verification_lake_at_rest_horizon.md`.

## 6. Disposition

**Request changes — done in this branch, approve with notes going forward.**

Blocking issues F1, F2a–F2c are fixed and gated. Non-blocking F3 is fixed. The
anchor's verdict is unchanged in direction (FAIL unconditional / PASS over
`t*`), and the one claim that was overstated (seed-independence evidence) is now
supported by an estimator that survives both a declared-window sweep and an
artefact-free re-derivation at another resolution. Residual risk: F2b shows that
"fit an exponential where it is exponential" was not obvious to two consecutive
missions on this case; the band and its rationale are therefore stated as a
reusable rule, not as a magic constant.

Remediation left open (not done here, needs a decision, not a patch):

* `freeze.py --verify` should be part of the case's acceptance path, not an
  optional flag.
* The rest of the repository's cases quote exponential/linear rates from
  hand-chosen amplitude windows; the band rule here should be checked against them
  before the same failure is discovered twice.