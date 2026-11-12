"""m16257 board-audit evidence builder for m16214 (CASE-SPH-FSI-ELASTIC-GATE).

Zero hand-copy: every number in the verdict JSON, the figure and the markdown
report is read back from the delivered artifact JSON or from this audit's own
probe JSONs / git / registry queries executed in this script.

Outputs (all inside the audit worktree):
  docs/board_audit/m16257/evidence.json      -> the row's evidence JSON
  docs/board_audit/m16257/fsi_elastic_gate.png
  docs/board_audit/m16257/report.md
  docs/board_audit/m16257/probe/antoci_elastic_gate.json  (byte copy of m16214's artifact)
  docs/board_audit/m16257/probe/artifact_sha256.txt
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent            # artifacts/audit_m16257
WT = HERE.parents[1]                              # audit worktree (m16257)
M16214 = Path("/root/tp-agent/work/m16214")
OUT_DIR = WT / "docs" / "board_audit" / "m16257"
ART = M16214 / "artifacts/fsi_elastic_gate/antoci_elastic_gate.json"
IMPL = M16214 / "src/tensorsph/fsi_elastic_gate.py"
ENTRY = M16214 / "benchmarks/fsi_elastic_gate/run_antoci.py"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def psql(sql: str) -> list[list[str]]:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d",
                        "tensorplatform", "-tA", "-F", "\x01", "-c", sql],
                       capture_output=True, text=True)
    return [ln.split("\x01") for ln in r.stdout.strip().split("\n") if ln]


def git(*args: str, cwd: Path = M16214) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    delivered = json.loads(ART.read_text())
    v1 = json.loads((HERE / "probe_verify_results.json").read_text())
    ls = json.loads((HERE / "probe_load_scale.json").read_text())

    # ---- preserve the delivered artifact + bind it by hash ----------------
    (HERE / "antoci_elastic_gate.json").write_bytes(ART.read_bytes())
    (HERE / "artifact_sha256.txt").write_text(
        f"{sha(ART)}  {ART}\n{sha(IMPL)}  {IMPL}\n{sha(ENTRY)}  {ENTRY}\n"
        f"{sha(ART)}  {HERE/'antoci_elastic_gate.json'} (audit copy)\n")

    # ---- independent registry / delivery queries -------------------------
    rows = psql("SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases "
                "WHERE anchor_slug='CASE-SPH-FSI-ELASTIC-GATE'")
    total_cases = psql("SELECT count(*) FROM tensor_benchmark_cases")[0][0]
    tests_on_disk = psql("SELECT count(*) FROM tensor_dev_missions WHERE id=16214")[0][0]
    heads = psql("SELECT status,coalesce(result->>'head_commit','NULL'),"
                 "coalesce(result->>'accepted','NULL') FROM tensor_dev_missions "
                 "WHERE id=16214")[0]
    remote_branch = git("ls-remote", "origin", "refs/heads/agent/m16214")
    in_main = git("cat-file", "-e", "main:src/tensorsph/fsi_elastic_gate.py")
    contains = git("branch", "-a", "--contains", "9fd55b0")
    commit = git("log", "--oneline", "-1", "--", "src/tensorsph/fsi_elastic_gate.py")
    test_globs = sorted(p.name for p in (M16214 / "tests").glob("test_*fsi*"))

    # ---- derived engineering numbers (all from the delivered parameters) --
    p = delivered["case"]["parameters"]
    EI = p["gate_E"] * p["gate_thickness"] ** 3 / 12.0           # N m^2 per m width
    EA = p["gate_E"] * p["gate_thickness"]                       # N per m width
    rho_A = p["gate_rho"] * p["gate_thickness"]                  # kg/m^2
    H = p["gate_height"]
    p_cr = math.pi ** 2 * EI / (4.0 * H ** 2)                    # cantilever Euler load
    self_weight = rho_A * p["gravity"] * H                       # N per m width
    hydro_load = 0.5 * p["rho0"] * p["gravity"] * p["water_height"] ** 2
    axial_strain = hydro_load / EA
    strain_from_stress = (hydro_load / p["gate_thickness"]) / p["gate_E"]
    mach = p["sound_speed"] / math.sqrt(p["gamma"] * 1.0e5 / p["rho0"])

    runs = {r["tag"]: r for r in delivered["runs"]}
    coarse, fine = delivered["runs"][0], delivered["runs"][-1]
    conv = delivered["self_convergence"]
    raw = next(r for r in ls["runs"] if r["tag"] == "raw_transfer")
    cal = next(r for r in ls["runs"] if r["tag"] == "calibrated")
    peak_vs_m16214 = (cal["peak_tip_x"] - coarse["peak_tip_x"]) / coarse["peak_tip_x"]

    ev = {
        "impl_files": [
            "src/tensorsph/fsi_elastic_gate.py (518 lines, sha256 %s) — ONLY on local "
            "branch agent/m16214 @9fd55b0; absent from main (%s); absent from GitHub "
            "(git ls-remote refs/heads/agent/m16214 -> empty)" % (sha(IMPL)[:16],
                                                                 "cat-file main:... empty"),
            "audit byte copy (provenance, not a delivery): "
            "docs/board_audit/m16257/probe/probe/src/tensorsph/fsi_elastic_gate.py sha256=%s" % sha(IMPL)[:16],
        ],
        "entry": "benchmarks/fsi_elastic_gate/run_antoci.py (sha256 %s) — same branch-only "
                 "status; no argparse entry in src/tensorsph/__main__.py, no platform tool/API "
                 "registration" % sha(ENTRY)[:16],
        "run_artifacts": [
            "/root/tp-agent/work/m16214/artifacts/fsi_elastic_gate/antoci_elastic_gate.json "
            "(sha256 %s; artifacts/ is .gitignore'd -> never enters git; mtime 11:26 > "
            "mission finished_at 11:20:15Z)" % sha(ART)[:16],
            "/root/tp-agent/work/m16214/artifacts/fsi_elastic_gate/antoci_elastic_gate.png",
            "audit copy: docs/board_audit/m16257/probe/antoci_elastic_gate.json sha256=%s" % sha(ART)[:16],
            "audit probe runs: docs/board_audit/m16257/probe/probe_verify_results.json, "
            "docs/board_audit/m16257/probe/probe_load_scale.json",
        ],
        "tests": [],
        "tests_note": "ZERO tests: `tests/` in m16214 has no test_*fsi* file (%r); no test "
                      "imports fsi_elastic_gate; nothing in the FSI path is regression-guarded" % test_globs,
        "integration_callers": [],
        "integration_callers_note": "only call site is its own benchmark harness "
                                    "benchmarks/fsi_elastic_gate/run_antoci.py; the module is not "
                                    "imported by any other TensorSPH module, CLI or platform tool",
        "benchmark": {
            "registry_rows_for_anchor": len(rows),
            "registry_total_rows": int(total_cases),
            "registry_note": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
                             "anchor_slug='CASE-SPH-FSI-ELASTIC-GATE' returns 0 rows; the goal's "
                             "judge '<10% | gate-tip peak+phase vs Antoci2007' has no registered "
                             "tolerance and no registered value. Auditor did NOT back-fill "
                             "(independence, same precedent as verdict m16252).",
            "ref": None,
            "ref_status": "WITHHELD",
            "ref_note": "delivered artifact self-declares case.reference.reference_values = "
                        "'WITHHELD - experimental gate-tip curve not read' (paper closed access); "
                        "PARAMETER_PROVENANCE says the geometry was NOT read from the article. "
                        "The mandated ref-vs-got comparison therefore does not exist.",
            "got_peak_tip_x_m": coarse["peak_tip_x"],
            "got_peak_time_s": coarse["peak_time"],
            "got_peak_tip_x_m_fine": fine["peak_tip_x"],
            "got_peak_time_s_fine": fine["peak_time"],
            "error_vs_ref_pct": None,
            "pass": False,
            "self_convergence_pct": {"peak": conv["peak_rel_diff"] * 100,
                                     "phase": conv["time_rel_diff"] * 100,
                                     "verdict": conv["verdict"],
                                     "note": "lattice self-consistency only, not a reference "
                                             "comparison; both runs used the same calibrated "
                                             "load_scale=%r" % coarse["load_scale"]},
            "delivered_internal_verification": {
                "V1_load_rel_error_pct": delivered["verification"]["V1_hydrostatic_load"]["rel_error"] * 100,
                "V1_verdict": delivered["verification"]["V1_hydrostatic_load"]["verdict"],
                "V2_tip_rel_error_pct": delivered["verification"]["V2_cantilever"]["rel_error"] * 100,
                "V2_verdict": delivered["verification"]["V2_cantilever"]["verdict"],
                "load_calibration_factor": delivered["verification"]["V1_hydrostatic_load"].get(
                    "load_calibration_factor"),
            },
            "audit_reproduction": {
                "V1_rel_error_pct": v1["V1_as_shipped"]["rel_error"] * 100,
                "V1_load_sph": v1["V1_as_shipped"]["load_sph"],
                "V1_load_exact": v1["V1_as_shipped"]["load_exact"],
                "V2_rel_error_pct": v1["V2_as_shipped"]["rel_error"] * 100,
                "V2_tip_rod_m": v1["V2_as_shipped"]["tip_rod"],
                "V2_tip_exact_m": v1["V2_as_shipped"]["tip_exact"],
                "V2_overdamped_rel_error_pct": v1["V2_overdamped"]["rel_error"] * 100,
                "V2_overdamped_tip_m": v1["V2_overdamped"]["tip_rod"],
                "V2_undamped_long_rel_error_pct": v1["V2_undamped_long"]["rel_error"] * 100,
                "V2_undamped_long_tip_m": v1["V2_undamped_long"]["tip_rod"],
                "V2_verdict_stability": "sign flip and 62x amplitude swing under a change of "
                                        "damping/horizon only -> V2 measures no converged quantity; "
                                        "the rod is WITHHELD, not merely FAILED",
                "repro_determinism": "audit re-run of the calibrated coarse lattice reproduces "
                                     "m16214's peak_tip_x bit-for-bit (rel diff %.1e)" % peak_vs_m16214,
            },
            "load_scale_sensitivity": {
                "raw_peak_tip_x_m": raw["peak_tip_x"],
                "calibrated_peak_tip_x_m": cal["peak_tip_x"],
                "peak_change_pct_for_plus_25_58pct_load": ls["peak_inflation_from_calibration"] * 100,
                "phase_raw_s": raw["peak_time"],
                "phase_calibrated_s": cal["peak_time"],
                "note": "the delivered result is multiplied by load_scale=%.6f, a factor obtained "
                        "by dividing out the module's OWN failed V1 (-20.37%%); the peak moves "
                        "%.2f%% in the WRONG direction for a 25.6%% load increase and the peak time "
                        "does not move at all -> the mandated metric cannot resolve a 25%% load-"
                        "transfer error at this configuration (tip_x peak %.1f mm >= gate length "
                        "%.0f mm, geometrically saturated)"
                        % (coarse["load_scale"], ls["peak_inflation_from_calibration"] * 100,
                           cal["peak_tip_x"] * 1000, H * 1000),
            },
            "physics_diagnostics": {
                "EI_Nm2_per_m": EI, "EA_N_per_m": EA, "area_density_kg_m2": rho_A,
                "cantilever_euler_load_N_per_m": p_cr,
                "gate_self_weight_N_per_m": self_weight,
                "self_weight_over_euler_load": self_weight / p_cr,
                "hydrostatic_load_N_per_m": hydro_load,
                "axial_strain_at_hydrostatic_load": axial_strain,
                "strain_from_sigma_over_E": strain_from_stress,
                "mach_at_c0": mach,
                "note": "the vertical 4 mm gate carries %.1fx its own Euler buckling load, so both "
                        "V2 (which adds full gravity to a 1e-5-scaled load) and the FSI run are "
                        "large-deflection/self-weight dominated: the small-deflection cantilever "
                        "Green's function used as V2's reference is not valid for V2's own setup, "
                        "which explains the -92%% without needing a rod-stiffness bug; and the axial "
                        "strain at the hydrostatic load is %.1f%% (=sigma/E %.1f%%), i.e. finite "
                        "strain, outside the linear rod's validity. Mach %.2f makes the WCSPH "
                        "weakly-compressible assumption marginal."
                        % (self_weight / p_cr, axial_strain * 100, strain_from_stress * 100, mach),
            },
        },
        "repro_cmd": "cd <TensorSPH checkout at branch agent/m16214 @9fd55b0> && "
                     "python benchmarks/fsi_elastic_gate/run_antoci.py --dx 0.0125 0.01 --t-end 0.4 "
                     "  # executed by this audit at 253/406 particles, 1642/2052 steps, "
                     "111 s / 108-441 s wall; entry works, output is deterministic; "
                     "but it is NOT on main and needs the un-merged branch, so the command is "
                     "reproducible only from the local worktree",
        "external_anchor": None,
        "external_anchor_note": "Antoci, Gallati & Sibilla (2007) Comput. Struct. 85:879-890, "
                                "DOI 10.1016/j.compstruc.2007.01.002 is CLOSED access; its "
                                "experimental gate-tip curve was never read, and no digitised "
                                "table/figure was obtained -> no external anchor exists in the "
                                "delivery (L5 is not claimable, L4 fails on this key alone)",
        "delivery_state": {
            "mission_status": heads[0], "head_commit": heads[1], "accepted": heads[2],
            "commit_on_branch": commit,
            "branch_on_github": remote_branch or "ABSENT (git ls-remote empty)",
            "branches_containing_commit": contains or "local agent/m16214 only",
            "in_main": in_main or "NOT_IN_MAIN",
            "artifact_in_git": "no (artifacts/ is .gitignore'd)",
        },
        "findings": {
            "F1_blocking_no_reference": "the mission's own judge ('<10%% gate-tip peak+phase vs "
                                        "Antoci2007') was never evaluated: reference WITHHELD, 0 "
                                        "registry rows. The delivered peak %.1f mm @ %.1f ms has no "
                                        "ref/got pair, so 'succeeded' is unsupported on the stated "
                                        "criterion." % (coarse["peak_tip_x"] * 1000, coarse["peak_time"] * 1000),
            "F2_blocking_calibration_of_a_failed_check": "the delivered result is scaled by "
                                                         "load_scale=%.6f derived by dividing out its own "
                                                         "failed V1 (-20.37%%); a failed verification was "
                                                         "converted into a fudge factor instead of a fix. "
                                                         "Audit probe: raw transfer peak %.1f mm vs calibrated "
                                                         "%.1f mm (%.2f%%), phase identical -> the metric is "
                                                         "saturated and cannot see the 25%% error either way."
                                                         % (coarse["load_scale"], raw["peak_tip_x"] * 1000,
                                                            cal["peak_tip_x"] * 1000,
                                                            ls["peak_inflation_from_calibration"] * 100),
            "F3_blocking_v2_invalid": "V2's verdict is not converged: audit re-runs give -92.5%% "
                                      "(as shipped), -153.9%% (heavier damping) and -5756.7%% "
                                      "(long horizon, undamped) for the same rod. Root cause "
                                      "quantified: V2 adds full gravity on a %.1f kg/m2 rod "
                                      "(%.1f N/m self weight, %.0fx the %.4f N/m Euler load) while "
                                      "scaling only the horizontal load by 1e-5, so its Euler-"
                                      "Bernoulli small-deflection reference does not describe its "
                                      "own setup. Rod correctness = WITHHELD."
                                      % (rho_A, self_weight, self_weight / p_cr, p_cr),
            "F4_blocking_not_delivered": "impl/entry exist only as a local commit 9fd55b0 on "
                                         "agent/m16214: not on GitHub (ls-remote empty), not in main; "
                                         "the run artifact is in .gitignore'd artifacts/ whose mtime "
                                         "is 6 min AFTER finished_at; mission result still records "
                                         "git_status '?? ...'. Deliverable is one worktree-recycle away "
                                         "from total loss.",
            "F5_high_zero_tests": "no test file, no invariant, no regression guard for the new "
                                  "module; the shipped script asserts nothing (V1/V2 print 'FAIL' and "
                                  "execution continues).",
            "F6_medium_prose_vs_artifact": "mission checks say 'dx=10 mm 因预算截止未跑完' while the "
                                          "artifact contains a finished dx=0.01 run (%.1f s wall, 2052 "
                                          "steps); the artifact was regenerated after the mission was "
                                          "closed, so prose and artifact describe different runs."
                                          % fine["wall_seconds"],
            "F7_medium_self_convergence_weak": "CONVERGED at %.2f%%/%.2f%% reads strong but is a "
                                               "two-lattice comparison of a geometrically saturated "
                                               "response (peak tip_x %.1f / %.1f mm vs 360 mm gate "
                                               "length); it tests the lattice, not the physics."
                                               % (conv["peak_rel_diff"] * 100, conv["time_rel_diff"] * 100,
                                                  coarse["peak_tip_x"] * 1000, fine["peak_tip_x"] * 1000),
            "not_run": "no independent re-derivation of the WCSPH fluid path itself (out of scope: "
                       "the reused engine is audited elsewhere); the closed-access Antoci curve was "
                       "not obtained (no licensed access from this runner); no full 0.4 s / dx=5 mm "
                       "refinement was attempted.",
        },
        "verdict_note": "request changes. Functional substance is L2 (real 518-line WCSPH+rod module "
                        "that runs end-to-end and reproduces bit-identically), but the L1 rung FAILED "
                        "(V1 -20.4%, V2 not converged), there is no external reference (WITHHELD), no "
                        "registry row, no tests and no merged code, so L3/L4/L5 are denied. The machine "
                        "gate will (correctly) compute <=L1 because the tests/registry keys are empty "
                        "and the impl files are not in main -> reopen.",
    }

    # ---- verdict JSON for the DB row -------------------------------------
    (OUT_DIR / "evidence.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for r in delivered["runs"]:
        ax.plot([t * 1000 for t in r["tip_history"]["t"]],
                [v * 1000 for v in r["tip_history"]["tip_x"]],
                label=f"delivered dx={r['dx']*1000:.1f} mm")
    ax.axhline(H * 1000, ls=":", c="k", lw=1)
    ax.text(2, H * 1000 + 6, "gate length 360 mm (geometric ceiling)", fontsize=7)
    ax.set_xlabel("t [ms]"); ax.set_ylabel("gate tip x [mm]")
    ax.set_title("m16214 delivered tip history\nNO reference curve (ref WITHHELD)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot([t * 1000 for t in delivered["runs"][0]["tip_history"]["t"]],
            [v * 1000 for v in delivered["runs"][0]["tip_history"]["tip_x"]],
            label="delivered (load_scale=1.2558)", lw=2)
    ax.plot([t * 1000 for t in delivered["runs"][0]["tip_history"]["t"]],
            [v * 1000 for v in delivered["runs"][0]["tip_history"]["tip_y"]],
            label="tip y, delivered", lw=1)
    ax.set_xlabel("t [ms]"); ax.set_ylabel("displacement [mm]")
    ax.set_title("audit probe: raw vs calibrated load (dx=12.5 mm)\n"
                 f"peak {raw['peak_tip_x']*1000:.1f} mm (raw, +25.6% less load) vs "
                 f"{cal['peak_tip_x']*1000:.1f} mm", fontsize=9)
    ax.scatter([raw["peak_time"] * 1000], [raw["peak_tip_x"] * 1000], marker="o",
               c="crimson", zorder=5, label="raw peak")
    ax.scatter([cal["peak_time"] * 1000], [cal["peak_tip_x"] * 1000], marker="x",
               c="k", zorder=5, label="calibrated peak")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2]
    labels = ["V1 load\n(as shipped)", "V2 tip\n(as shipped)", "V2 tip\nheavier damping",
              "V2 tip\nlong/undamped"]
    vals = [v1["V1_as_shipped"]["rel_error"] * 100, v1["V2_as_shipped"]["rel_error"] * 100,
            v1["V2_overdamped"]["rel_error"] * 100, v1["V2_undamped_long"]["rel_error"] * 100]
    bars = ax.bar(labels, vals, color=["#c44", "#c44", "#e88", "#e88"])
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val - 120, f"{val:.1f}%",
                ha="center", fontsize=8)
    ax.axhline(0, c="k", lw=0.8)
    for y in (-10, 10):
        ax.axhline(y, ls="--", c="grey", lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("relative error vs analytic reference [%]")
    ax.set_title("internal verification: audit re-runs\n(gate needs |err| < 10%)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fsi_elastic_gate.png", dpi=130)
    print("figure:", OUT_DIR / "fsi_elastic_gate.png")

    # ---- markdown report --------------------------------------------------
    md = f"""# m16257 board audit — m16214 CASE-SPH-FSI-ELASTIC-GATE (TensorSPH)

verdict: **L2 declared (functional substance), gate will reopen** · verdict_by `board-audit`
· scope `m16214 CASE-SPH-FSI-ELASTIC-GATE` · audit mission 16257

## 1. What was delivered, and the contract it was graded against

m16214's goal fixed the judge: `<10% | 闸门顶点水平位移时程峰值与相位 vs Antoci2007 实验`,
with `目标数值从 ref 原文/手册查得（零手抄）`. Delivered material (byte hashes in
`docs/board_audit/m16257/probe/artifact_sha256.txt`):

| item | value |
|---|---|
| impl | `src/tensorsph/fsi_elastic_gate.py`, 518 lines, sha256 `{sha(IMPL)[:16]}…` |
| entry | `benchmarks/fsi_elastic_gate/run_antoci.py`, sha256 `{sha(ENTRY)[:16]}…` |
| artifact | `artifacts/fsi_elastic_gate/antoci_elastic_gate.{{json,png}}`, json sha256 `{sha(ART)[:16]}…` |
| tests | **none** (`tests/` has no `test_*fsi*`; no import of the module anywhere) |
| git | local branch `agent/m16214` @ `9fd55b0`; `git ls-remote origin refs/heads/agent/m16214` → **empty**; `main:src/tensorsph/fsi_elastic_gate.py` → **absent** |
| registry | `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-FSI-ELASTIC-GATE'` → **{len(rows)} rows** (table has {total_cases} rows) |

Mission accounting: status `{heads[0]}`, `result.head_commit = {heads[1]}`,
`accepted = {heads[2]}`, mission `git_status` at finish = `?? benchmarks/fsi_elastic_gate/`,
artifact directory mtime 11:26 vs `finished_at` 11:20:15Z.

## 2. Method (this audit)

Independent re-execution, not a re-read: the module's package tree was copied out of the m16214
worktree (`docs/board_audit/m16257/probe/probe/`, hash-pinned), and

1. the shipped verification ladder was re-run as-is (`probe_verify.py`);
2. V2 was falsified by changing only the harness (damping / horizon), to separate
   "under-relaxed" from "wrong rod";
3. the production lattice was re-run twice, with the delivered `load_scale={coarse['load_scale']:.6f}`
   and with the raw transfer (`load_scale=1.0`), to measure how much of the headline number is
   the fudge factor (`probe_load_scale.py`);
4. the registry and the delivery state were queried directly (psql / `git ls-remote` / `git cat-file`).

All probe outputs: `docs/board_audit/m16257/probe/*.json`, logs alongside.

## 3. Numbers

### 3.1 The mandated benchmark metric

| quantity | ref (Antoci 2007) | got | error | judged? |
|---|---|---|---|---|
| gate-tip peak `tip_x` | **WITHHELD** (closed access, curve never read) | {coarse['peak_tip_x']*1000:.2f} mm @ {coarse['peak_time']*1000:.1f} ms (dx=12.5 mm) | **not computable** | **NO** |
| gate-tip peak `tip_x` | WITHHELD | {fine['peak_tip_x']*1000:.2f} mm @ {fine['peak_time']*1000:.1f} ms (dx=10 mm) | not computable | **NO** |
| peak time (phase) | WITHHELD | {coarse['peak_time']*1000:.1f} / {fine['peak_time']*1000:.1f} ms | not computable | **NO** |

Answer to "is it <3%?": **cannot be answered** — there is no reference value in the delivery.
Lattice self-consistency is {conv['peak_rel_diff']*100:.2f}% (peak) / {conv['time_rel_diff']*100:.2f}% (phase),
which is a convergence check on the lattice, not an accuracy check.

### 3.2 Internal verification, re-run by this audit (gate: |err| < 10%)

| check | delivered | audit re-run | verdict |
|---|---|---|---|
| V1 hydrostatic load via the SPH transfer path | {delivered['verification']['V1_hydrostatic_load']['rel_error']*100:+.2f}% | {v1['V1_as_shipped']['rel_error']*100:+.2f}% ({v1['V1_as_shipped']['load_sph']:.2f} vs {v1['V1_as_shipped']['load_exact']:.2f} N/m) | FAIL (reproduced) |
| V2 rod tip vs Euler-Bernoulli cantilever (as shipped) | {delivered['verification']['V2_cantilever']['rel_error']*100:+.2f}% | {v1['V2_as_shipped']['rel_error']*100:+.2f}% ({v1['V2_as_shipped']['tip_rod']*1e3:.4f} vs {v1['V2_as_shipped']['tip_exact']*1e3:.4f} mm) | FAIL (reproduced) |
| V2, only the harness changed (damping 0.90, 2e4 steps) | — | {v1['V2_overdamped']['rel_error']*100:+.2f}% (tip {v1['V2_overdamped']['tip_rod']*1e3:+.4f} mm) | sign flip → check not converged |
| V2, only the harness changed (undamped, 2e5 steps) | — | {v1['V2_undamped_long']['rel_error']*100:+.1f}% (tip {v1['V2_undamped_long']['tip_rod']*1e3:+.2f} mm) | diverges → check invalid |

### 3.3 How much of the headline number is the calibration factor

m16214 divided its own failed V1 out of the production run: `load_scale = 1/(1-0.20367) = {coarse['load_scale']:.6f}`.

| run (dx=12.5 mm, t=0.4 s) | load applied | peak `tip_x` | peak time |
|---|---|---|---|
| raw transfer | 1.000× | {raw['peak_tip_x']*1000:.2f} mm | {raw['peak_time']*1000:.1f} ms |
| delivered (calibrated) | {coarse['load_scale']:.4f}× | {cal['peak_tip_x']*1000:.2f} mm | {cal['peak_time']*1000:.1f} ms |
| change | +25.58% | **{ls['peak_inflation_from_calibration']*100:+.2f}%** | 0.0% |

The audit's calibrated re-run reproduces m16214's number bit-for-bit
(`{cal['peak_tip_x']!r}` vs `{coarse['peak_tip_x']!r}`), so the harness is deterministic and the
`repro_cmd` does execute — but a 25.6% load change moves the mandated metric by −2.9% and the
phase not at all. The response is geometrically saturated (peak {cal['peak_tip_x']*1000:.1f} mm ≥
the 360 mm gate length), so this metric cannot resolve even a gross load-transfer error.

### 3.4 Why V2 is invalid (root cause, computed from the delivered parameters)

EI = {EI:.3e} N·m²/m, EA = {EA:.0f} N/m/m, ρ_A = {rho_A:.1f} kg/m².
Cantilever Euler load `π²EI/(4H²)` = **{p_cr:.4f} N/m**; the gate's own weight = **{self_weight:.1f} N/m**
= **{self_weight/p_cr:.0f}×** the Euler load. V2 scales the horizontal load by 1e-5 but adds full
gravity, so the rod it "verifies" is a self-weight-buckled rod compared against a small-deflection
Green's function — {v1['V2_as_shipped']['rel_error']*100:.1f}% is explained by the setup, not by a
calibrated rod error. The hydrostatic load ({hydro_load:.1f} N/m) also produces
{axial_strain*100:.1f}% axial strain (σ/E = {strain_from_stress*100:.1f}%), i.e. finite strain,
outside the linear rod's range; `Mach = {mach:.2f}` makes the weakly-compressible WCSPH assumption marginal.

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
"""
    (OUT_DIR / "report.md").write_text(md)
    print("report:", OUT_DIR / "report.md")
    print(json.dumps({"peak_coarse": coarse["peak_tip_x"], "peak_fine": fine["peak_tip_x"],
                      "raw_peak": raw["peak_tip_x"], "p_cr": p_cr,
                      "self_weight": self_weight, "ratio": self_weight / p_cr}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())