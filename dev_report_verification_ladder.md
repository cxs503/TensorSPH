# Dev report — TensorSPH zero-to-one verification ladder

Node `p83w1-n01-tensorsph-verification` (campaign DAG, physics domain).
Branch `agent/m7114`.  Artefacts: `artifacts/verification_ladder/`,
document: [`docs/benchmarks/verification-ladder.md`](docs/benchmarks/verification-ladder.md),
tests: `tests/test_vv_ladder.py`.

## 1. What was built

A new package `src/tensorsph/vv/` (verification & validation) that runs three
anchors and a negative-control panel and emits **one schema-pinned JSON**:

| module | content |
|---|---|
| `vv/tgv.py` | periodic 2-D Taylor-Green vortex on a **fixed neighbour list with the minimum-image convention**; spline kernel, Tait EOS and Morris Laplacian taken from the library, so the anchor measures the library's viscosity |
| `vv/hydrostatic.py` | square tank, banded time-averaged pressure against `rho g h`, kernel-support band excluded from the gate |
| `vv/dambreak.py` | the library's dam-break case plus two references (Ritter analytic inside its validity window; Martin & Moyce transcription as a cross-check) and jet-dispersion statistics |
| `vv/ladder.py` | runner, provenance (git sha/branch/dirty, torch, python), negative controls, campaign-criteria mapping |
| `vv/schema.py` | dependency-free validator: anchors, gates, verdict consistency, provenance |
| `vv/figures.py` | four figures generated **from the JSON only** |

Everything is deterministic: no random seeds, no fitted constants in the
harness, and the doc is generated from the JSON (`make_doc.py`), so no number in
the document can disagree with the artefact.

## 2. Results (numbers are the artefact's, not retyped)

| rung | gate | measured | verdict |
|---|---|---|---|
| C1 hydrostatic, `p = rho g h`, dx = 0.02 m, 4 s, mean over the last 25 % | max band error < 1 % | **0.924 %** (mean 0.41 %) | PASS |
| C2 Taylor-Green decay rate vs exact `4 nu k^2` = 2.5133 1/s, dx = 0.025 m, M = 0.02 | error < 3 % | **1.179 %** (R² = 0.999987) | PASS |
| C3 dam-break front vs Ritter inside `t < a/sqrt(gH)` | error < 5 % | **93.8 %** (front speed 31 % of `2c0`) | FAIL |
| C3 experimental cross-check (Martin & Moyce) | (not gated) | MAE **0.491** in `Z/a` = 49 % of `a` | reported |
| C4a gravity off | error must move | **100 %** | PASS |
| C4b viscosity off | decay rate must collapse | **0.17 %** of the analytic rate | PASS |
| C4c surface tension off | jet must disperse | front response **0.011 %** | INCONCLUSIVE |

Key mechanism behind C2's accuracy: the split-out viscous operator
(`viscous_acceleration`) reproduces the exact `lap u = -2 k^2 u` of the
Taylor-Green field with a **relative L2 error of 0.679 % at dx = 0.025 m and
2.651 % at dx = 0.05 m** — second-order convergence, and the same magnitude as
the decay-rate error. The rung therefore measures the discretisation of
viscosity rather than an aggregate of everything.

## 3. Honest failures and what they mean

**C3 (dam-break, < 5 %).** Two independent references both disagree with the
column, and in the same direction (the simulated front is *slower*): 93.8 %
against the Ritter shallow-water front inside its validity window, and an MAE of
0.49 `Z/a` against the transcribed experiment. Refinement does not fix it: the
observed order of the experimental MAE between dx = 0.04 and 0.02 m is **0.154**
and of the front-speed deficit **0.128**, i.e. the offset is essentially
resolution independent. This is consistent with the repository's own
ritter-shallow-limit ladder, which reaches 6.2 % only at `H/a = 0.1`,
dx = 0.005 m (573 s) and Richardson-extrapolates to 0 % — the error is
**physical** (a 2:1 column is outside shallow-water theory) plus reference
uncertainty (the bundled table is audited WITHHELD as a digitisation), not a
scheme defect that a 5 % single-resolution gate can detect. The criterion as
written is therefore not attainable; the verifiable statement is the
convergence order plus the front-speed ratio, and both are in the artefact.

**C4c (surface tension).** The campaign expects the dam-break jet to disperse
when `sigma = 0`. At the case's Weber number (~550) the capillary force is
dynamically negligible, and the measured response is 0.011 % — the test cannot
discriminate. Labelled INCONCLUSIVE rather than PASS, because claiming that the
mechanism is proven effective by a null response would be false. The correct
mechanism check for this library already exists elsewhere
(`benchmarks/laplace_pressure`, `sigma_eff/sigma = 0.70`).

## 4. TGV validity window (measured, reusable)

The anchor is only meaningful while the `O(rho U0^2)` pressure field that
balances the exact convective term stays above the summation-density noise of
the lattice. Measured at dx = 0.05 m, `c0 = 10` m/s:

| M = U0/c0 | decay-rate error |
|---|---|
| 0.002 | 2.45 % |
| 0.005 | 2.11 % |
| 0.010 | 2.18 % |
| 0.020 | 3.00 % |
| 0.050 | 174 % |
| 0.100 | 240 % |

So the plateau is set by the operator error (~2-3 % at dx = 0.05, 0.68 % at
dx = 0.025) and the anchor breaks down sharply above M ~ 0.05. **Any TGV
verification in this solver must be run at M <= 0.02**; at the library's usual
M ~ 0.1 the vortex decays five times too fast and the number says nothing about
the viscosity model. That is a trap worth writing down.

## 5. Audit chain

1. `benchmarks/verification_ladder/run.py` → `artifacts/verification_ladder/ladder.json`
   (provenance: git sha of the code commit, dirty flag, torch/python versions).
2. `make_doc.py` regenerates `docs/benchmarks/verification-ladder.md` from that
   JSON; every table cell is injected.
3. `vg_regime.py` → `tgv_mach_window.json` (section 4).
4. `pytest tests/test_vv_ladder.py` — 16 tests: schema acceptance/rejection,
   provenance, the discrete-Laplacian accuracy, the analytic identities, the
   hydrostatic rung and its `g = 0` control, the TGV decay and its `nu = 0`
   control, dam-break stability/mass conservation, front filtering, and a
   quick-mode end-to-end ladder run whose output is validated against the
   schema.

## 6. Limits

* Rung 2 is 2-D and periodic. The 3-D Taylor-Green vortex has no analytic
  solution; a 3-D rung would need reference DNS data and is not covered.
* Rung 3 uses free-slip walls, whereas the Martin & Moyce tank has walls; the
  dam-break wall condition is a documented, separate question in this repo.
* The hydrostatic rung's 1 % gate is met only with a 4 s run averaged over the
  last 25 % (a 2 s run leaves a residual sloshing mode and reads 2.3 % at
  dx = 0.02 m). The gate is a statement about a settled pool, not about an
  instantaneous profile.
* No turbulence model, no density diffusion, no surface tension in the gated
  dam-break run: each would add unverified physics to a verification case.