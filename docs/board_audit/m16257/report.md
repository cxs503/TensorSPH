# m16257 board audit — m16214 CASE-SPH-FSI-ELASTIC-GATE (TensorSPH)

verdict: **L2 declared (functional substance), gate will reopen** · verdict_by `board-audit`
· scope `m16214 CASE-SPH-FSI-ELASTIC-GATE` · audit mission 16257

## 1. What was delivered, and the contract it was graded against

m16214's goal fixed the judge: `<10% | 闸门顶点水平位移时程峰值与相位 vs Antoci2007 实验`,
with `目标数值从 ref 原文/手册查得（零手抄）`. Delivered material (byte hashes in
`docs/board_audit/m16257/probe/artifact_sha256.txt`):

| item | value |
|---|---|
| impl | `src/tensorsph/fsi_elastic_gate.py`, 518 lines, sha256 `e92f63dc0aec76ac…` |
| entry | `benchmarks/fsi_elastic_gate/run_antoci.py`, sha256 `de36100b77054b64…` |
| artifact | `artifacts/fsi_elastic_gate/antoci_elastic_gate.{json,png}`, json sha256 `5b342ead9d84696d…` |
| tests | **none** (`tests/` has no `test_*fsi*`; no import of the module anywhere) |
| git | local branch `agent/m16214` @ `9fd55b0`; `git ls-remote origin refs/heads/agent/m16214` → **empty**; `main:src/tensorsph/fsi_elastic_gate.py` → **absent** |
| registry | `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-FSI-ELASTIC-GATE'` → **0 rows** (table has 30 rows) |

Mission accounting: status `succeeded`, `result.head_commit = NULL`,
`accepted = true`, mission `git_status` at finish = `?? benchmarks/fsi_elastic_gate/`,
artifact directory mtime 11:26 vs `finished_at` 11:20:15Z.

## 2. Method (this audit)

Independent re-execution, not a re-read: the module's package tree was copied out of the m16214
worktree (`docs/board_audit/m16257/probe/probe/`, hash-pinned), and

1. the shipped verification ladder was re-run as-is (`probe_verify.py`);
2. V2 was falsified by changing only the harness (damping / horizon), to separate
   "under-relaxed" from "wrong rod";
3. the production lattice was re-run twice, with the delivered `load_scale=1.255762`
   and with the raw transfer (`load_scale=1.0`), to measure how much of the headline number is
   the fudge factor (`probe_load_scale.py`);
4. the registry and the delivery state were queried directly (psql / `git ls-remote` / `git cat-file`).

All probe outputs: `docs/board_audit/m16257/probe/*.json`, logs alongside.

## 3. Numbers

### 3.1 The mandated benchmark metric

| quantity | ref (Antoci 2007) | got | error | judged? |
|---|---|---|---|---|
| gate-tip peak `tip_x` | **WITHHELD** (closed access, curve never read) | 354.33 mm @ 253.7 ms (dx=12.5 mm) | **not computable** | **NO** |
| gate-tip peak `tip_x` | WITHHELD | 349.56 mm @ 249.8 ms (dx=10 mm) | not computable | **NO** |
| peak time (phase) | WITHHELD | 253.7 / 249.8 ms | not computable | **NO** |

Answer to "is it <3%?": **cannot be answered** — there is no reference value in the delivery.
Lattice self-consistency is 1.37% (peak) / 1.58% (phase),
which is a convergence check on the lattice, not an accuracy check.

### 3.2 Internal verification, re-run by this audit (gate: |err| < 10%)

| check | delivered | audit re-run | verdict |
|---|---|---|---|
| V1 hydrostatic load via the SPH transfer path | -20.37% | -20.37% (333.04 vs 418.22 N/m) | FAIL (reproduced) |
| V2 rod tip vs Euler-Bernoulli cantilever (as shipped) | -92.46% | -92.46% (0.1268 vs 1.6804 mm) | FAIL (reproduced) |
| V2, only the harness changed (damping 0.90, 2e4 steps) | — | -153.88% (tip -0.9054 mm) | sign flip → check not converged |
| V2, only the harness changed (undamped, 2e5 steps) | — | -5756.7% (tip -95.06 mm) | diverges → check invalid |

### 3.3 How much of the headline number is the calibration factor

m16214 divided its own failed V1 out of the production run: `load_scale = 1/(1-0.20367) = 1.255762`.

| run (dx=12.5 mm, t=0.4 s) | load applied | peak `tip_x` | peak time |
|---|---|---|---|
| raw transfer | 1.000× | 364.78 mm | 253.7 ms |
| delivered (calibrated) | 1.2558× | 354.33 mm | 253.7 ms |
| change | +25.58% | **-2.86%** | 0.0% |

The audit's calibrated re-run reproduces m16214's number bit-for-bit
(`0.3543333617266935` vs `0.3543333617266935`), so the harness is deterministic and the
`repro_cmd` does execute — but a 25.6% load change moves the mandated metric by −2.9% and the
phase not at all. The response is geometrically saturated (peak 354.3 mm ≥
the 360 mm gate length), so this metric cannot resolve even a gross load-transfer error.

### 3.4 Why V2 is invalid (root cause, computed from the delivered parameters)

EI = 5.333e-03 N·m²/m, EA = 4000 N/m/m, ρ_A = 10.0 kg/m².
Cantilever Euler load `π²EI/(4H²)` = **0.1015 N/m**; the gate's own weight = **35.3 N/m**
= **348×** the Euler load. V2 scales the horizontal load by 1e-5 but adds full
gravity, so the rod it "verifies" is a self-weight-buckled rod compared against a small-deflection
Green's function — -92.5% is explained by the setup, not by a
calibrated rod error. The hydrostatic load (418.2 N/m) also produces
10.5% axial strain (σ/E = 10.5%), i.e. finite strain,
outside the linear rod's range; `Mach = 0.76` makes the weakly-compressible WCSPH assumption marginal.

## 4. Level decision

| rung | criterion | result |
|---|---|---|
| L1 math/unit | a passing verification | **FAILED** — V1 −20.4%, V2 invalid |
| L2 independent function | real code, input → compute → output | **MET** — 518-line WCSPH+AD-rod module, runs both lattices, deterministic |
| L3 system integration | merged callers in the solver/platform | **NO** — single call site is its own benchmark script; nothing in `main`; no CLI/API/registry |
| L4 benchmark | ref vs got inside tolerance + repro | **NO** — ref WITHHELD, 0 registry rows, internal verification failing |
| L5 research | L4 + external anchor | **NO** |

Declared level **2** (functional substance). Because `tests` and the registry keys are empty and the
implementation is not on `main`, the machine scheme will compute ≤ L1 and reopen — which is the
correct outcome. Not L0/L1: real physics code exists; not L3+: nothing is integrated and nothing is
validated.

## 5. What would make this L4 (for the rework round)

1. Fix V1's transfer bias at source (kernel-support correction at the moving wall) instead of
   dividing it out; re-run with `load_scale = 1.0`.
2. Rebuild V2 as a real unit test: gravity off, load applied through the same path, static
   equilibrium by energy minimisation, and assert the Green's-function value (this is a `pytest`
   with a tolerance, not a print).
3. Obtain the Antoci 2007 gate-tip curve (or a digitised table from an open validation paper that
   reproduces it) and register it in `tensor_benchmark_cases` with the 10% tolerance; then report
   ref vs got with peak and phase errors.
4. Merge `agent/m16214` to `main` and keep the artifacts under a non-ignored path so the run
   artifacts survive the worktree.
5. Use a resolution/metric that is not geometrically saturated (e.g. tip displacement before the
   gate reaches the floor, or the load history on the gate) so the metric can discriminate.

## 6. Honest limits of this audit

* The Antoci curve is behind a paywall and was not obtained here — so "the physics is wrong" is
  **not** claimed; only "the delivery contains no reference and its internal checks fail" is claimed.
* The WCSPH engine reused by the module was not re-audited (separate scope).
* No fine-lattice (dx=5 mm) refinement was attempted; the saturation finding rests on the two
  delivered lattices plus the two load-scale runs.
* Figures: `docs/board_audit/m16257/fsi_elastic_gate.png` (delivered tip history with the missing
  reference; raw-vs-calibrated peak; verification ladder re-runs).
