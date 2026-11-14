-- Board audit m20188 — CASE-SPH-POISEUILLE-MORRIS (mission 20084, repo TensorSPH)
-- Auditor: board-audit (m20188), 2026-11-14. Cross-check only: the case was NOT redone.
--
-- 1) Referee row: absent from tensor_benchmark_cases before this audit
--    (88 anchor_slugs in the table, none for this anchor and none for any CASE-SPH-*).
--    Registered here with the analytic reference (exact plane-Poiseuille solution),
--    the audited got-values and the case's predeclared tolerances (DESIGN.md §7).
-- 2) Maturity verdict, level 3: benchmark evidence is L4-capable and auditor-reproduced,
--    but the deliverable is not L4-closable (see level_reason in the evidence JSON).

BEGIN;

INSERT INTO tensor_benchmark_cases
  (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
SELECT
  'CASE-SPH-POISEUILLE-MORRIS',
  'Plane Poiseuille, body-force driven, periodic x, no-slip walls — SPH ISPH projection (Morris 1997 low-Re channel)',
  'Exact steady Navier-Stokes solution u = g y (H-y)/(2 nu), u_max = g H^2/(8 nu), tau_wall = rho g H/2, evaluated programmatically (Morris, Fox & Zhu 1997 JCP 136:214 doi:10.1006/jcph.1997.5776 Poiseuille section; paper table access WITHHELD — closed access)',
  'centre-row velocity error and interior L2 of the profile, both normalised by u_max = gH^2/(8nu), interior band 0.15 <= y/H <= 0.85; got (audited, artefact runs/poiseuille_morris/results.json + auditor re-run m20188): Re_c=0.1 H/dx=16 centre +1.2161% / L2 1.1701%; Re_c=1 H/dx=16 +1.2542% / 1.2204%; Re_c=12.26 H/dx=16 +1.2587% / 1.2216%; density-fluctuation gate 1.3615e-03 > 1.0e-03 at Re_c=12.26 (predeclared self-gate, honest FAIL, disclosed by this audit)',
  '<5% (case pool ledger; DESIGN.md §7 PASS) and <3% (platform strong band): measured 1.17-1.26% meets both',
  'TensorSPH',
  'passed',
  now()
WHERE NOT EXISTS (
  SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug = 'CASE-SPH-POISEUILLE-MORRIS');

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorSPH',
  'm20084 CASE-SPH-POISEUILLE-MORRIS',
  20188,
  3,
  $json${
  "impl_files": [
    "src/tensorsph/isph_channel.py (972 lines; run_case/build_channel/fit_profile/gates at L105-972) — the real incompressible-SPH projection solver the case drives",
    "benchmarks/poiseuille_morris/run.py (539 lines; reference_profile L105, case_from_reynolds L110, profile_metrics L146, run_isph L201, verdict_for L483)"
  ],
  "entry": "benchmarks/poiseuille_morris/run.py::main (L320) -> tensorsph.isph_channel.run_case (src/tensorsph/isph_channel.py)",
  "run_artifacts": [
    "runs/poiseuille_morris/results.json (89,824 B; 9 ISPH cases, 0 WCSPH cases, per-row profiles, gates, controls)",
    "runs/poiseuille_morris/fig1_profiles.png .. fig5_sensitivity.png (149/62/107/63/76 kB) + figures.json",
    "docs/benchmarks/poiseuille-morris.md (169 lines, six sections present)"
  ],
  "tests": "tests/test_poiseuille_morris.py — auditor re-ran in a scratch copy: 6 passed in 5.51 s (2026-11-14, torch 2.7.1, 1 thread)",
  "integration_callers": [
    "src/tensorsph/method_selection.py:109 (module registry: 'tensorsph.isph_channel' offered as an ISPH route)",
    "benchmarks/density_order_likeforlike/run.py:56 (reuses isph_channel.pair_geometry/summation_density)",
    "NO platform/API caller: grep for isph_channel in /root/ship-performance-platform-incoming -> 0 hits; not wired into a production solver entry point"
  ],
  "benchmark": {
    "referee_row": "ABSENT before this audit (tensor_benchmark_cases: 125 rows / 88 anchors, none for CASE-SPH-POISEUILLE-MORRIS and none for any CASE-SPH-*) -> registered by this audit, id returned by the INSERT above",
    "reference": "exact solution u=g y (H-y)/(2 nu) evaluated in code; auditor re-derived it independently (numpy, no import of the case module): max|artifact u_reference - exact| = 0.0 over all 9 cases x all rows",
    "got_vs_ref": {
      "Re_c=0.1 H/dx=16": "centre +1.2161 %, interior L2 1.1701 % (artifact 1.2209 %/1.1701 %; the 0.005 pp centre difference is a normalisation convention: code divides the centre row by its own u_ref, doc declares /u_max)",
      "Re_c=1 H/dx=16": "+1.2542 % / 1.2204 %",
      "Re_c=12.26 H/dx=16": "+1.2587 % / 1.2216 %"
    },
    "auditor_reproduction": "fresh re-run of the three canonical runs (575 steps each, 17-18 s CPU) through run_case + the case's own make_config/profile_metrics: Re_c=0.1 centre +1.2161 % L2 1.1701 % density 1.060e-04; Re_c=1 +1.2542 %/1.2204 %/1.673e-04; Re_c=12.26 +1.2587 %/1.2216 %/1.362e-03 — bit-reproducible against the committed artefact",
    "predeclared_criterion": "DESIGN.md §7: PASS if e_L2 < 5 % and |e_c| < 5 %; strong band < 3 % -> measured 1.17-1.26 % passes both",
    "honest_failures": [
      "density_fluctuation_max 1.3615e-03 > limit 1.0e-03 (pass=false) for the canonical Re_c=12.26, h/dx=1.90, columns=12 run; reproduced by the auditor (1.362e-03)",
      "second density failure 3.624e-03 at Re_c=1, columns=6 (short-domain sensitivity run)"
    ],
    "verdict_block_defect": "run.py verdict is {isph:{'0.1':...}, wcsp:{'1.0':null,'12.26':null}} — re_list was [0.1], so the Re_c=12.26 run whose density gate fails never reaches the pass/fail summary, and 'pass' is computed from profile errors only (L498-500), ignoring gates.*"
  },
  "repro_cmd": "PYTHONPATH=src:. python benchmarks/poiseuille_morris/run.py --out runs/poiseuille_morris --threads 1 (in /root/tp-agent/work/m20084; auditor ran the equivalent headline sub-case, 575 steps, identical numbers) — full suite 49.98 s wall clock per the artefact",
  "external_anchor": "Morris, Fox & Zhu (1997) JCP 136(1):214-226 doi:10.1006/jcph.1997.5776 — paper table WITHHELD (closed access, Unpaywall oa_status=closed, checked 2026-11-14); the graded reference is the exact analytic solution (REAL), a necessary but not sufficient condition for agreement with the paper",
  "level_reason": "Level 3, not 4. The benchmark evidence itself is L4-capable and auditor-verified: real solver, complete case (parameter table, H/dx ladder 8/12/16, Re ladder 0.1/1/12.26, h/dx and domain sensitivities, zero-forcing and repeat controls), error analysis vs the exact solution at 1.17-1.26 % (< 3 % strong band), and bit-reproducible re-runs. It is NOT closable at L4 because (a) the mission never published its report — tensor_reports has 0 rows for mission_id=20084 and 0 rows whose markdown mentions POISEUILLE-MORRIS (owner rule: 报告未发=本单未完成); (b) the case's own predeclared density gate fails at the canonical base Reynolds 12.26 and this is absent from the doc and filtered out of the verdict block; (c) docs ②/⑤/⑥ present a WCSPH comparison route as run on identical geometry while cases.wcsp_production is empty (n_wcsp=0, verdict.wcsp all null) — an unsupported claim; plus (d) summary {} so the doc's observed-order convergence table renders empty (no quantified p), (e) doc ③ states L/H = 1.2 while the canonical runs are columns=12 at H/dx=16 -> L/H = 0.75.",
  "overstated_claims": [
    "docs/benchmarks/poiseuille-morris.md ② 'Two routes ... are run on the identical geometry' — the artefact contains zero WCSPH cases",
    "⑤⑥ WCSPH statements ('~1.2 over-diffusive at h/dx=1.3', 'peak ~20 % low') have no counterpart in this artifact",
    "doc ④ headline table is labelled Re_c = 1 while the Verdict table reports Re_c = 0.1; both rows are internally consistent but the 'headline' label is ambiguous"
  ],
  "not_run": [
    "full 9-case suite re-execution (auditor re-ran the 3 canonical runs only; the 6 sensitivity/ladder runs were re-derived arithmetically, not re-simulated)",
    "make_figs.py / make_doc.py regeneration",
    "WCSPH route (tensorsph.droplet_impact / benchmarks/poiseuille) — no artifact exists to audit"
  ],
  "disposition": "request changes: publish the mission-20084 report; disclose the density-gate FAIL at Re_c=12.26 (or fix it) and let it enter the pass/fail logic; either run the WCSPH comparison or remove/qualify the claim; recompute summary so the observed order is reported; correct doc ③ L/H to 0.75. L4 re-certification is reachable after these five items."
}$json$::jsonb,
  'board-audit'
);

COMMIT;

SELECT id, anchor_slug, status FROM tensor_benchmark_cases
  WHERE anchor_slug = 'CASE-SPH-POISEUILLE-MORRIS';
SELECT id, level, gate_state, gate_level FROM tensor_maturity_verdicts
  WHERE scope = 'm20084 CASE-SPH-POISEUILLE-MORRIS' ORDER BY id DESC LIMIT 1;