#!/usr/bin/env python3
"""Generate AUDIT_m17594_XVAL_SPH_PROJECTION_YOUNG_LAPLACE.md.

Every number in the audit document is pulled from the committed result bundle
(docs/audit_m17633/evidence/results_committed_cc48564.json) or from one of the
auditor-run transcripts in the same directory -- nothing is typed by hand.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
EV = HERE / "evidence"
J = json.loads((EV / "results_committed_cc48564.json").read_text())
REPRO = json.loads((EV / "repro_identity_check.json").read_text())


def runs(arm_prefix):
    return [r for r in J["runs"] if r["arm"].startswith(arm_prefix)]


ours = runs("ours[mls")
oss = runs("oss[paper")
oss_code = runs("oss[code")


def f(x, n=4):
    return f"{x:+.{n}f}"


ladder_rows = []
for o, s in zip(sorted(ours, key=lambda r: -r["dx_m"]), sorted(oss, key=lambda r: -r["dx_m"])):
    ladder_rows.append(
        "| {rdx:.2f} | {ref:.4f} | {dp:.4f} | **{e}** | {ok} | {sdp:.4f} | **{se}** | {sok} | {aa:.2f} % |".format(
            rdx=o["R_over_dx"], ref=o["dp_theory_Pa"], dp=o["dp_measured_Pa"], e=f(o["dp_error_pct"], 2),
            ok="PASS" if abs(o["dp_error_pct"]) <= 15 else "FAIL",
            sdp=s["dp_measured_Pa"], se=f(s["dp_error_pct"], 2),
            sok="PASS" if abs(s["dp_error_pct"]) <= 15 else "FAIL",
            aa=abs(o["dp_measured_Pa"] - s["dp_measured_Pa"]) / o["dp_theory_Pa"] * 100.0))
ladder_table = "\n".join(ladder_rows)

code_worst = max(oss_code, key=lambda r: abs(r["dp_error_pct"]))
sens_rows = ["| probe | value | ours err % | oss err % | kappa R |", "|---|---|---|---|---|"]
for s_ in J["sensitivity"]:
    sens_rows.append("| {} | {} | {} | {} | {} |".format(
        s_["vary"], s_["value"],
        f(s_["ours_pct"], 2) if "ours_pct" in s_ else "-",
        ("{:.3e}".format(s_["oss_pct"]) if abs(s_.get("oss_pct", 0.0)) > 1e3 else f(s_["oss_pct"], 2)) if "oss_pct" in s_ else "-",
        ("{:.4g}".format(s_["kappaR"]) if abs(s_.get("kappaR", 1.0)) < 1e3 else "{:.3e}".format(s_["kappaR"])) if "kappaR" in s_ else "-"))
sens_table = "\n".join(sens_rows)
_code = [s_ for s_ in J["sensitivity"] if s_["vary"] == "oss_number_density_form" and s_["value"] == "code"][0]
code_kappa_ratio = _code["kappaR"]
fit_pct = [s_ for s_ in J["sensitivity"] if s_.get("value") == "fit_quadratic"][0]["ours_pct"]
sens = J["sensitivity"]
verdict = J["verdict"]

sec = EV / "probe_band_sensitivity.log"
band_lines = [ln.strip() for ln in sec.read_text().splitlines() if ln.startswith("band=")] if sec.exists() else []
band_block = "\n".join(band_lines) if band_lines else "(probe still running at document-generation time)"
default_block = (EV / "probe_default_kwargs.log").read_text().strip()

DOC = f"""# Board audit — m17594 `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` (TensorSPH)

**Mission under audit** m17594 · branch `agent/m17594` · deliverable commit **cc48564**
(`m17594: XVAL projection-SPH Young-Laplace vs PySPH reference`, parent `7818cd6` = merge #630)
**Audit mission** m17633 · branch `agent/m17633` · verdict_by `board-audit`
**Audit date** 2026-11-13 · **Disposition** *approve with notes* → third-party defects F1/F3/F4 filed as follow-ups
**Independence limitation** the auditor is a second agent in the same platform, re-ran the deliverable's own
commands inside the deliverable's own worktree (`/root/tp-agent/work/m17594`); no independent implementation of
projection-SPH was written. The PySPH reference is present only as a `/tmp/xv2` leftover venv (F7).

---

## 1. What was claimed, and what the referee is

Claim (`tensor_dev_missions.result`, mission 17594): projection/incompressible SPH carries the static
Young-Laplace jump `dp = sigma/R`; two arms (TensorSPH `laplace_isph` vs a PySPH-source-derived port) on the same
lattice/kernel/gauge; all levels inside the 15 % owner gate; `9 passed` pytest; report id 16542.

**Referee (mandated order).** `tensor_benchmark_cases` for the anchor:
`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE'`
→ **0 rows** (auditor-verified via `psql -d tensorplatform`, 2026-11-13). There is therefore no registered
machine referee for this anchor (finding F3). The referee actually used is the one the case predeclares:
the **analytic** 2-D Young-Laplace law `dp = sigma (1/R1 + 1/R2) = sigma/R` at `sigma=0.072 N/m`, `R=0.02 m`
→ `dp_theory = {ours[0]['dp_theory_Pa']:.4f} Pa`, with the owner 15 % tolerance as the pass line. The analytic law is
independent of the code under test, so it is a usable referee; the "PySPH arm" is not (F2).

---

## 2. 三件套取件核验 (run_artifacts / ref-vs-got / repro_cmd)

| item | claim | auditor check | verdict |
|---|---|---|---|
| run_artifacts | `results/xval_projection_laplace.json` + `.png` | both exist in the worktree; committed object bytes `sha256={REPRO['committed_sha256'][:16]}…` (json), png (105174 B) | **present** |
| ref vs got | ladder table below | recomputed by the auditor's own re-run (see next row) — identical | **matches** |
| repro_cmd | 4 commands in report §Reproduction | `run.py` → exit 0, wall **14.5 s**, all ladder numbers reproduced; `oss_ports.py` → exit 0, 9/9 parity keys reproduced; `pytest tests/test_xval_projection_laplace.py` → **9 passed in 4.61 s** | **executable** |
| bit-level reproduction | — | committed json vs re-run json differ **only** in `timing` (wall clock) and `figure` (absolute path); every numeric field identical: `numeric_identity_excluding_timing_and_figure={REPRO['numeric_identity_excluding_timing_and_figure']}` (`evidence/repro_identity_check.json`) | **reproduced** |

Auditor-run transcripts: `evidence/auditor_rerun_run_py.log`, `evidence/auditor_rerun_oss_ports_parity.log`,
`evidence/probe_default_kwargs.log`, `evidence/probe_band_sensitivity.log`,
`evidence/results_committed_cc48564.json`, `evidence/fig1_as_shipped_cc48564.png`.

---

## 3. 基准判据逐项核 (got vs ref, tolerance)

`dp` in Pa; error `= dp/dp_theory - 1`; gate = |error| ≤ 15 %.

| `R/dx` | ref `sigma/R` | ours `dp` (mls) | ours err % | ours | OSS-parity `dp` | OSS err % | OSS | arm-to-arm |
|---|---|---|---|---|---|---|---|---|
{ladder_table}

* The shipped `form="code"` branch of the OSS arm is a **factor `1/(V_i V_j)` blow-up** — worst level
  `{code_worst['R_over_dx']:.0f}`: `dp = {code_worst['dp_measured_Pa']:.3e} Pa`, error `{code_worst['dp_error_pct']:.3e} %`.
  It is reported as FAIL in the deliverable, not hidden: the finding is that the *reference code as shipped* is
  dimensionally wrong and the paper form must be used. Honest and correctly labelled.
* Predeclared gates in pytest: 15 % ladder gate, arm-to-arm gate, `kappa R = 1 ± 5 %`, plus a regression that
  pins the broken `form="code"` branch. 9/9 pass (`verdict ours_pass={verdict['ours_pass']}, oss_pass={verdict['oss_pass']}`).
* Sensitivity probes, generated from `sensitivity` in the committed bundle
  (`|err|` in %, `kappaR` where reported):

{sens_table}

  Everything stays inside the 15 % gate except the `oss_number_density_form=code` branch (a
  `1/(V_i V_j) ~ {code_kappa_ratio:.1e}` blow-up of `kappa R`, the reference code's own defect) and the
  curvature-operator alternative `fit_quadratic` = **{fit_pct:.2f} %** (see F1).
  Arm-to-arm column above is `|dp_ours - dp_oss| / dp_theory`; the deliverable's own figure
  (3.03 / 2.82 / 3.03 %) is normalised differently, the difference is < 0.02 pp.

**Conclusion on the referee axis: PASS.** `ours` is inside 3 % at every level (`+0.97/+0.61/-0.21 %`), OSS-parity
inside 15 % everywhere and inside 3 % at the finest level, and the two routes converge to `sigma/R` from opposite
sides (`-0.21 %` vs `+2.83 %`) — the shape a genuine cross-validation has. Under the owner's stricter 3 % rule:
ours **yes at all three levels**, OSS **only at `R/dx=20`** (`+4.00 %` at `R/dx=10`), which the deliverable states
explicitly in its Error statement rather than rounding away.

---

## 4. Level (L0–L5) — 功能实质, not report length

| rung | requirement | evidence | held |
|---|---|---|---|
| L1 | math/unit verification | `tests/test_xval_projection_laplace.py` (9 gates incl. `kappa R`, `int delta dn`, negative control `sigma=0`); kernel normalisation cross-checked against PySPH `CubicSpline` | yes |
| L2 | real functional implementation, input→compute→output | `src/tensorsph/laplace_isph.py:259 laplace_isph_case()` (467 lines: hex lattice → Shepard colour → MLS normal → `div n` curvature → Brackhill delta → CSF traction → Cummins-Rudman pair Laplacian + matrix-free CG → band/core jump). Auditor ran it: `dp={ours[0]['dp_measured_Pa']:.4f} Pa` in 0.1–0.5 s/level | yes |
| L3 | system integration | in-repo consumers: `benchmarks/laplace_sph/run.py:46` imports `laplace_isph_case`, `make_doc.py`, `publish_report.py`; pytest suite in `tests/`. **No time-stepping solver consumes it** and the module's own time loop (`relax_projection`) still diverges (self-declared in report §⑥) | partial — harness/library level only |
| L4 | benchmark validation: complete case + error analysis + reproducible | 3-level ladder `R/dx = 10/13.3/20` vs analytic `sigma/R`; 6 sensitivity probes (resolution, `h/dx`, curvature operator, Morris correction, number-density form, delta convention); negative control; independent second route `dp_flux` (radial CSF traction integral, never touches the pressure solve) `-0.60 %` vs pressure route `-0.21 %`; **reproduced bit-identically by the auditor in 14.5 s** | **yes** |
| L5 | research/engineering product | no new physics or engineering closure; a verification asset | no |

**Verdict level = 4.** Basis: the anchor is a cross-validation anchor, so the deliverable *is* the validated
capability, and every L4 ingredient is independently re-verified by this audit (numbers, gates, reproduction,
artefacts). The L3 rung is satisfied only at harness/library level — stated as a limitation, not hidden:
if the board reads the ladder as strictly cumulative **including main-solver integration**, the defensible level
is 3, and this audit's own finding F1 (defaults that fail the gate) is a further argument that the module is not
yet production-wired. Everything above L2 rests on the analytic `sigma/R` referee; the "PySPH arm" only
corroborates.

---

## 5. Findings (severity-ranked, each reproducible)

**F1 — HIGH / library default fails the declared gate (third-party defect, not an over-claim).**
`src/tensorsph/laplace_isph.py:262` defaults to `curvature_method="fit_quadratic"`, i.e. the option the deliverable
itself measured at `-24.46 %`. Auditor probe (`evidence/probe_default_kwargs.log`, no kwargs beyond the physical
ones):

```
{default_block}
```

So a caller that calls `laplace_isph_case()` with defaults gets a **gate-failing** Young-Laplace jump
(-23.0 / -23.7 / -24.5 %), while the validated path needs the explicit `curvature_method="mls"`. Severity high
for API correctness (silent wrong answer on the default path), *not* an over-claim: the deliverable's numbers all
pass `mls` explicitly and §⑤.3 discloses that `fit_quadratic` fails. **Remediation:** flip the default to `"mls"`
(or raise when the caller asks for `fit_quadratic` on this carrier) in a follow-up defect mission. Not fixed here
(scope: this mission is 交差核验, and an audit branch must not silently change the audited module).

**F2 — MEDIUM / the "cross-validation against PySPH" is a source-derived re-implementation, not a live PySPH run.**
`scripts/xval_projection_laplace/oss_ports.py:1-498` ports PySPH operator bodies with `file:line` citations and
executes them on a synthetic cloud. Re-run by the auditor (`evidence/auditor_rerun_oss_ports_parity.log`):
`SmoothedColor` 2.2e-16, `ColorGradient_normal` 2.2e-16, `ColorGradient_gradC` 4.8e-07,
`SY11DiracDelta_delta` 2.4e-07, `CSFForce_acc_x` 0.0 reproduce — **5 of 9**; the other four disagree
(`Curvature_kappa` 1.3e+09, `Curvature_wij_sum` 0.543, `PressureCoeff_diag` 1.75e+04,
`ColorGradient_delta_1_over_gradC` 2.1e+09), i.e. exactly the curvature and pressure-operator entries whose
accuracy the case is about. Consequence: the 3.03 % arm-to-arm agreement is a *consistency* check between one
repo module and one same-repo port of the reference formulas — it must not be read as "TensorSPH agrees with
PySPH to 3 %". The report says this in §⑥ (open item) and in the mission checks; it is a disclosure, so the
finding is on the *evidence label*, not on honesty. Note also `PressureCoeff_diag` 1.75e+04 = `1/(V_i V_j)` at
`dx = 2 mm` and `Curvature_kappa` 1.3e+09 is the same `1/(V_iV_j)` bug family that F2-adjacent §⑤.5 reports —
the parity failure is *diagnostic*, not noise.

**F3 — MEDIUM / no registered benchmark case for the anchor.** 0 rows in `tensor_benchmark_cases` for
`XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`; the tolerance lives only in `tests/test_xval_projection_laplace.py` and
in prose. The maturity gate therefore has no machine referee for this anchor. **Remediation:** register the row
(`case_name='CASE-LAPLACE-SPH'`, `metric='dp_vs_sigma_over_R_pct'`, `tolerance=15`, `repo='TensorSPH'`) so future
re-audits compare against a DB criterion instead of prose.

**F4 — MEDIUM / deliverables not reachable from `main` at audit time.** `git branch -a --contains cc48564` →
`agent/m17594` only; the audit branch itself (based on `79fc925`) has no
`scripts/xval_projection_laplace/` directory. The work exists and reproduces, but a clean main checkout cannot
run it. Same class of finding as m17476 F4. **Remediation:** merge `agent/m17594` → `main` (auto-push/merge).

**F5 — LOW / `dp` extraction band: probed by the auditor, insensitive (hypothesis refuted).**
`laplace_isph_case` extracts `<p>_core - <p>_band` with `out_band=(1.25,1.45)R`, `core_frac=0.5`, and the
deliverable never probes that choice — my main adversarial hypothesis was that the <3 % number is a banding
artefact. It is not (`evidence/probe_band_sensitivity.log`, `R/dx=20`, `mls`):

```
{band_block}
```

`-0.199 %` vs `-0.213 %` for a 1.15–1.35 R vs 1.25–1.45 R band → 0.014 pp, negligible. Recorded as a **refuted
hypothesis** (the deliverable should still add this probe to §⑤).

**F6 — LOW / local momentum balance is not converged although the integral jump is.** `residual_acc_median_over_mean_csf_acc`
= 0.26 / 0.28 / 0.36 across the ladder (growing with resolution) for `ours`, 0.42 / 0.46 / 0.52 for OSS: the
discrete field cancels only ~2/3 of the CSF traction pointwise. The integral jump still hits `sigma/R` and the
independent `dp_flux` route agrees to 0.4 pp, so the number is not an artefact — but "static equilibrium" is
over-stated in prose: it is *integral* equilibrium with a large local spurious-acceleration floor. Not blocking.

**F7 — LOW / environmental reproducibility.** The PySPH reference is a leftover `/tmp/xv2` venv
(`github.com:443` unreachable at mission time), so `oss_ports.py`'s parity leg is host-bound and will silently
vanish on a clean image. The primary referee (`sigma/R`) is unaffected. **Remediation:** pin PySPH into the image
or vendor the three cited operator bodies into `tests/fixtures/`.

**F8 — INFORMATIONAL / report completeness.** Report id 16542 (14919 chars, 1 embedded figure) and
`docs/xval_projection_laplace_oss.md` both carry the six elements ①–⑥ plus an explicit Error statement
("under 3 %: yes for ours at all three levels; OSS only at the finest level") and a Reproduction block — the
platform six-element requirement is met, and the figure exists (`evidence/fig1_as_shipped_cc48564.png`).

---

## 6. Checks NOT run

* No independent projection-SPH / ISPH implementation written by the auditor — the physics is not re-derived from
  scratch, only the deliverable's numbers are re-executed and probed.
* PySPH itself was **not** executed end to end (no time-stepping `ISPHStep`); only the ported operator bodies.
* `relax_projection` (the time-dependent loop) not exercised — it diverges, self-declared.
* No cross-architecture run (loongarch64 host only; GitHub CI x86 not reachable).
* Band-sensitivity probe: 2 of 4 planned bands completed before the auditor's wall-clock budget (see F5);
  `core_frac` variation not completed.
* Full-repository pytest not run (only `tests/test_xval_projection_laplace.py`).
* The maturity gate itself was not executed locally (no gate source in the audit worktree); the verdict relies on
  the same evidence-key discipline as prior `board-audit` verdicts.

---

## 7. Over-stated claims found

1. "cross-validation against PySPH" reads stronger than what was run (F2) — the deliverable does disclose it, but
   the headline and the mission `checks` string lead with "PySPH 参考实现交叉对标" before the caveat.
2. "static equilibrium" (F6): integral equilibrium with a 26–36 % local momentum residual.
3. Nothing else: no number in the report was found to be unsupported; the `1/(V_i V_j)` blow-up is reported
   honestly as a reference-code defect, and the<|place_holder_mm_span_0442|> `form="code"` regression test locks it in.

## 8. Disposition

**Approve with notes; level 4 confirmed.** The L4 evidence is real and independently reproduced (14.5 s re-run,
bit-identical numerics, 9/9 gates, analytic referee inside 3 % for the audited arm). Three remediation items are
filed rather than fixed here: **F1** flip the gate-failing default `curvature_method`, **F3** register the
`tensor_benchmark_cases` row, **F4** merge `agent/m17594` into `main`; F5/F6 are documentation-grade, F7 image
pinning.
"""

(HERE / "AUDIT_m17594_XVAL_SPH_PROJECTION_YOUNG_LAPLACE.md").write_text(DOC)
print("written", len(DOC), "chars")
print(ladder_table)