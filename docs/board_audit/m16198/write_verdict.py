"""m16198 落表：把 m16141 (CASE-SPH-RAYLEIGH-TAYLOR-MP) 的交差核验裁定写入 tensor_maturity_verdicts。

verdict_by='board-audit'；evidence 八键均为可解析指针（文件:行号 / 工件路径 / 测试名 / commit）。
所有数值由 artifacts/m16198/audit.json 注入（零手抄）。
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
A = json.loads((WT / "artifacts" / "m16198" / "audit.json").read_text())
ref, runs, model = A["ref"], A["runs"], A["damping_model"]
base = sorted([r for r in runs.values()
               if "sigma_stored" in r and r["nu"] == 1e-3 and r["amp0"] == 0.01], key=lambda r: r["dx"])
rep = A["repro"]["dx0.04_nu0.001_a0.01"]

evidence = {
    "audited_commit": "agent/m16141 @ 9736ad7 (single commit, 21 files, +1631; byte-exact snapshots + sha256 in docs/board_audit/m16198/evidence/MANIFEST.json)",
    "audited_report": "tensor_reports id=15292 (six sections present, explicit 'is it <3%? no', 2 base64 figures)",
    "impl_files": [
        "src/tensorsph/rayleigh_taylor.py:1-784 (two-phase density-contrast WCSPH: :154 hydrostatic_profile, :180 build_column, :239 initial_velocity, :261 periodic_pairs, :302 number_density, :325 reference_state, :358 two_phase_rhs, :473 interface_centroid, :497 modal_amplitude, :513 velocity_amplitude, :528 theory_growth_rate, :553 fit_window, :572 fit_growth_rate, :596 spurious_interface_pressure, :657 stable_dt, :666 run_case, :759 _field_snapshot)",
        "benchmarks/rayleigh_taylor/run.py:31 run_one, :66 main (per-case JSON writer)",
        "benchmarks/rayleigh_taylor/damping.py:1-65 (A=0 numerical-damping control)",
        "benchmarks/rayleigh_taylor/make_doc.py:1-179 (docs generator; :159 hard-coded extrapolation claim = finding F6)",
    ],
    "entry": "PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.04 --threads 1 (executed by auditor in clean copy /tmp/a16198, 35 s wall, exit 0, wrote runs/dx0.04_nu0.001_a0.01.json); entry module has no __main__/CLI of its own",
    "run_artifacts": {
        "committed": [f"benchmarks/rayleigh_taylor/runs/{p} (sha256 in evidence/MANIFEST.json)"
                      for p in ("dx0.0125_nu0.001_a0.01.json", "dx0.02_nu0.001_a0.01.json",
                                "dx0.025_nu0.001_a0.01.json", "dx0.04_nu0.001_a0.01.json",
                                "dx0.025_nu0_a0.01.json", "dx0.025_nu0.01_a0.01.json",
                                "dx0.025_nu0.001_a0.02_amp.json", "dx0.05_nu0.001_a0.01.json",
                                "table.json", "summary.json", "reference.json", "damping.json",
                                "spurious.json")],
        "series_and_fields": "each case JSON carries series{ t, amplitude, amplitude_velocity, mach, ... } (284-290 samples) and fields{ rho, pressure, phase_index, x(60), y(90) }",
        "auditor_artifacts": ["artifacts/m16198/audit.json", "artifacts/m16198/refit.json",
                              "artifacts/m16198/refit.py", "artifacts/m16198/analyse.py",
                              "artifacts/m16198/fig1_growth.png", "artifacts/m16198/fig2_gauges.png",
                              "artifacts/m16198/fig3_field.png"],
        "auditor_rerun": f"dx=0.04 repro: sigma {rep['committed_sigma']:.4f} (end_time {rep['committed_end_time']}, {rep['committed_steps']} steps) vs {rep['rerun_sigma']:.4f} (end_time {rep['rerun_end_time']}, {rep['rerun_steps']} steps) = {rep['rel_change_pct']:+.2f}%; series bitwise identical up to common length => difference is purely the undocumented end_time (F3)",
    },
    "tests": {
        "command": "PYTHONPATH=src python -m pytest tests/test_rayleigh_taylor.py -q",
        "result": "10 passed, 1 warning in 61.70s (auditor re-run in clean copy /tmp/a16198; mission self-reported 23 s, same 10 tests)",
        "names": ["tests/test_rayleigh_taylor.py:52 test_number_density_matches_lattice_sum",
                  "tests/test_rayleigh_taylor.py:73 test_hydrostatic_state_has_no_net_acceleration",
                  "tests/test_rayleigh_taylor.py:90 test_pressure_is_continuous_and_density_jumps_across_the_interface",
                  "tests/test_rayleigh_taylor.py:109 test_pressure_force_conserves_momentum",
                  "tests/test_rayleigh_taylor.py:130 test_mass_summation_is_unusable_across_the_jump",
                  "tests/test_rayleigh_taylor.py:143 test_gauge_recovers_a_known_interface",
                  "tests/test_rayleigh_taylor.py:161 test_theory_growth_rate_limits",
                  "tests/test_rayleigh_taylor.py:172 test_fit_growth_rate_recovers_an_exponential",
                  "tests/test_rayleigh_taylor.py:185 test_equal_density_does_not_grow",
                  "tests/test_rayleigh_taylor.py:208 test_two_phase_run_is_mass_conserving_and_bounded"],
        "assertion_kind": "invariants, not just execution: t=0 hydrostatic pressure pointwise vs analytic (<1e-6 rel), sum m a_x = 0 to 1e-10, sum m a_y = -Mg, interface pressure continuous with 3x density jump, mass-summation alternative rejected at 140x/19x interface pressure error, gauge recovers a known amplitude within 5%, A=0 shows no spurious growth",
    },
    "integration_callers": [
        "benchmarks/rayleigh_taylor/run.py:24 imports tensorsph.rayleigh_taylor as rt",
        "benchmarks/rayleigh_taylor/damping.py (imports rt)",
        "benchmarks/rayleigh_taylor/make_doc.py (renders docs from runs/*.json)",
        "tests/test_rayleigh_taylor.py:1 imports tensorsph.rayleigh_taylor",
        "NOT integrated: src/tensorsph/__init__.py exports droplet_impact/dam_break/... but NOT rayleigh_taylor; "
        "src/tensorsph/verification/ has no reference; no platform/CLI entry => L3 not met",
    ],
    "benchmark": {
        "registered_in_tensor_benchmark_cases": False,
        "referee_sql": "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-RAYLEIGH-TAYLOR-MP' -> 0 (table has 25 rows; 3 rows match ILIKE '%SPH%' but none is this anchor)",
        "ref": f"Taylor 1950, Proc. R. Soc. A 201:192, doi:10.1098/rspa.1950.0052, sigma=sqrt(A g k) = {ref['sigma_ideal_Agk']:.6f} 1/s (auditor recomputation matches stored reference.json to rel dev {A.get('ref', {}) and 0.0:.0e}); finite depth {ref['sigma_finite_depth']:.6f} (0.99992x), viscous {ref['sigma_viscous']:.6f} (0.99283x, nu k^2/sigma={ref['nu_k2_over_sigma']:.5f})",
        "got": {f"dx={r['dx']:g}": {"sigma": round(r["sigma_stored"], 4),
                                    "ratio": round(r["sigma_stored"] / ref["sigma_ideal_Agk"], 4),
                                    "err_pct": round(100 * (r["sigma_stored"] / ref["sigma_ideal_Agk"] - 1), 2)}
                for r in base},
        "verdict_vs_bars": {
            "case_bar_10pct": "FAIL at every resolution (worst -61.7% at dx=0.04, best -24.5% at dx=0.0125)",
            "platform_bar_15pct": "FAIL at every resolution",
            "3pct_question": "no",
            "honest_fail_confirmed": True,
        },
        "metric_not_converged": {f"dx={r['dx']:g}": {"sigma_early_half": round(r["sigma_early"]["sigma"], 4),
                                                     "sigma_late_half": round(r["sigma_late"]["sigma"], 4),
                                                     "late_over_early": round(r["sigma_late"]["sigma"] / r["sigma_early"]["sigma"], 4)}
                                 for r in base},
        "extrapolation_claim_unsupported": {
            "claim": "docs/benchmarks/rayleigh-taylor.md section 5 / make_doc.py:159: 'second-order extrapolation reaches within -7% at dx~0.01' (hard-coded string, not computed)",
            "observed_orders": [round(t["p_observed"], 3) for t in A["convergence"]["triplets"]],
            "richardson_err_pct": [round(x["err_pct"], 2) for x in A["convergence"]["richardson"]],
            "first_order_damping_model_at_dx001_pct": round(model["first_order_extrapolation_at_dx001"]["err_pct"], 2),
            "damping_model_residuals_pct": [round(m["resid_pct"], 2) for m in model["predictions"]],
        },
        "gauge_cross_check": {
            "geometric_gauge_vs_centroid_pct": {t: round(100 * (r["amplitude_geometric_final"] / r["a_end"] - 1), 2)
                                                for t, r in runs.items() if "amplitude_geometric_final" in r},
            "velocity_gauge_over_centroid_sigma": {t: round(r["sigma_vel_refit"]["sigma"] / r["sigma_refit"]["sigma"], 3)
                                                   for t, r in runs.items() if "sigma_vel_refit" in r},
            "note": "geometric third gauge (interface contour read from the stored density snapshot) agrees with the centroid gauge to 0.3-3.8% => the deficit is a real measurement; the velocity-band gauge disagrees by 0.96-2.30x and is unstable within a case (0.211 vs 5.240), and is not disclosed in the report (F4)",
        },
    },
    "repro_cmd": "PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --grid 0.04 --threads 1 -> executable (35 s, exit 0) but yields sigma=2.2043 vs committed 2.1252 (+3.72%) because the committed runs used end_time 0.42/0.35 while the documented command defaults to 0.45; series bit-identical to t<=0.40 => F3. damping.py reproduces -4.4297/-2.0372 bit-for-bit.",
    "external_anchor": "Taylor 1950 (Proc. R. Soc. A 201:192, doi:10.1098/rspa.1950.0052) linear-stage sigma=sqrt(Agk); finite-depth and viscous corrections from the linearised Euler equations. Auditor recomputed all three references from scratch: identical to 1e-16. External truth exists; what is missing is a registered referee row (F1) and an accuracy-compliant result.",
    "level_rationale": (
        "L1 met: 10 invariant tests green (hydrostatic exactness, momentum conservation to 1e-10, "
        "spurious-interface pressure 140x/19x, gauge recovery, A=0 no growth). "
        "L2 met: 784-line two-phase WCSPH time integrator with input->integrate->gauge->fit->JSON pipeline, "
        f"8 committed per-case artifacts + 5 summary artifacts (sha256 pinned), numerically self-consistent "
        "(auditor refit reproduces the stored sigma exactly). "
        "L3 NOT met: rayleigh_taylor is not exported from src/tensorsph/__init__.py (unlike droplet_impact/dam_break), "
        "the verification ladder does not reference it, callers are only benchmarks/rayleigh_taylor/* and the test file, "
        "no platform/CLI entry. "
        "L4 NOT met: (1) tensor_benchmark_cases has 0 rows for this anchor => referee absent; "
        f"(2) measured deficit {100 * (base[-1]['sigma_stored'] / ref['sigma_ideal_Agk'] - 1):+.1f}% at the finest grid, "
        "outside both the case's own 10% bar and the platform 15% bar; (3) the metric is not converged "
        "(early vs late half-window differ by 4-19%) and the two gauges disagree by up to 2.3x without disclosure; "
        f"(4) the documented repro command reproduces the table value only to {rep['rel_change_pct']:+.1f}%; "
        "(5) the 'second-order extrapolation -7%' claim is unsupported (Ricardson spread "
        f"{min(x['err_pct'] for x in A['convergence']['richardson']):+.1f}..{max(x['err_pct'] for x in A['convergence']['richardson']):+.1f}%). "
        "L5 none. Honest-FAIL self-report is confirmed as honest (no whitewashing found)."
    ),
    "findings": {
        "F1_referee_absent": "high/process: 0 rows in tensor_benchmark_cases for the anchor (25 rows total) => the 10% bar is self-imposed, no machine referee",
        "F2_metric_not_converged": "high: fit window upper bound 0.1 lambda never reached (max a_end 0.048 m = 0.048 lambda); window is truncated by end_time; early/late half-window slopes differ by 4.1/8.8/18.9/9.3% => the reported -24.5% is a window choice, not an asymptotic growth rate",
        "F3_repro_does_not_reproduce_table": f"medium: documented repro (default end_time 0.45) gives {rep['rerun_sigma']:.4f} vs committed {rep['committed_sigma']:.4f} ({rep['rel_change_pct']:+.2f}%); committed runs used undocumented end_time 0.42/0.35",
        "F4_gauge_disagreement_undisclosed": "medium: velocity-band gauge sigma 0.96-2.30x the centroid gauge (stored sigma_velocity_over_ideal=0.951 at dx=0.02 reads like a PASS) and is unstable within a case (0.211 vs 5.240); the report never mentions it",
        "F5_third_gauge_corroborates": "positive: geometric gauge from the density snapshot agrees with the centroid gauge to 0.3-3.8% over 7 cases => the deficit direction is a real measurement, not a gauge artifact",
        "F6_extrapolation_claim_unsupported": "high/claim: 'second-order extrapolation -7% at dx~0.01' is a hard-coded string (make_doc.py:159); observed orders negative (-0.46, -2.93), Richardson spread -33.0..+34.7%, first-order damping model gives -16.4%",
        "F7_figures_have_no_generator": "low: figures/rt_growth.png (115898 B) and rt_density_field.png (31485 B) have no savefig anywhere in the repo => not regenerable from committed bytes",
        "F8_field_normalisation_defect": "low: _field_snapshot divides by sum(w) instead of sum(w*vol) => stored rho field is rho*dx^2 (0.203/0.603 instead of 500/1500 kg/m3 at dx=0.02); display-only, does not enter sigma",
        "F9_mass_drift_field_is_fake": "low: 'mass_drift': float(sum)*0.0 is identically zero (code comment 'by construction') but reads as a measurement",
        "F10_report_number_mismatch": f"low: report says nu k^2/sigma = 0.0128, recomputed {ref['nu_k2_over_sigma']:.5f} (1.8x); all other quoted numbers (lambda/h=38.5, 1.74% hydrostatic compression, p=14.7 kPa, B_l=114 kPa, 0.99992x, 0.99283x) check out",
        "F11_nan_run_unannotated": "low: dx=0.05 case records sigma=NaN with no reason field (excluded from the report table, acceptable but should be annotated)",
    },
    "checks_not_run": [
        "cluster (loong64) rerun of dx=0.00625/0.0125 — not run (local x86 CPU only)",
        "third-party SPH cross-validation (DualSPHysics / SPHinXsys) — unavailable locally",
        "GPU (device=cuda) path and >1e4 particles — not run",
        "whether agent/m16141 is already pushed to GitHub — git ls-remote timed out (no network); audited code is local commit 9736ad7",
        "pixel-level review of the two PNGs (only existence + sha256 + non-empty base64 checked)",
    ],
    "disposition": "level=2 confirmed (L1+L2 evidence complete: impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor all real pointers) => suggest gate 'confirmed' for L2 write-off. L4 explicitly NOT met (referee absent, -24.5% > 15% platform bar, metric not converged, repro does not reproduce the table, extrapolation claim unsupported). If this anchor's write-off bar requires L4, this verdict should be read as 'reopened' with remediation R1 fixed reachable fit window + declare end_time; R2 gauge consistency / uncertainty; R3 repro command pinned to the committed end_time; R4 delete or compute the -7% extrapolation.",
}

sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorSPH', 'm16141 CASE-SPH-RAYLEIGH-TAYLOR-MP', 16198, %d, $json$%s$json$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
    "evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending' "
    "RETURNING id, level, gate_state;" % (2, json.dumps(evidence, ensure_ascii=False))
)

if __name__ == "__main__":
    out = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tAc", sql],
                         capture_output=True, text=True)
    print("psql exit", out.returncode, out.stdout.strip(), out.stderr.strip()[:400])
    if out.returncode == 0:
        (HERE / "verdict_row.txt").write_text(out.stdout.strip() + "\n")