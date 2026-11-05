"""Extract the two protocols' ladder numbers from the conflict artifacts.

Run inside the merge worktree (branch agent/m3163 + main merged) *before* the
union rewrite would be enough, but running it afterwards is better: it proves
the preserved block really holds main's numbers.

    python docs/merge/pr103/extract_ladder_compare.py <out.json>

Writes one record per rung with both protocols' read-outs, so the report figure
and table are programmatically injected (no hand-copied numbers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path("benchmarks/laplace_law/convergence")
RUNGS = ["relax_dx0.0025.json", "relax_dx0.0020.json", "relax_dx0.0015.json"]


def main() -> int:
    out = Path(sys.argv[1])
    rows = []
    for name in RUNGS:
        rep = json.loads((BENCH / name).read_text())
        sup = rep["provenance"]["superseded_protocol_run"]
        rows.append({
            "artifact": name,
            "dx_m": rep["setup"]["dx_m"],
            "n_particles": rep["setup"]["n_particles"],
            "primary": {
                "protocol": "t/t_sigma = 1.0 (agent/m3163, run_ladder.py)",
                "steps": rep["setup"]["steps"],
                "t_over_t_sigma": rep["setup"]["t_over_t_sigma"],
                "bandA_err_pct": rep["laplace"]["rel_error_pct"],
                "absolute_err_pct": rep["laplace"]["rel_error_absolute_pct"],
                "spurious_ratio": rep["spurious_current"]["ratio"],
            },
            "preserved": {
                "protocol": "fixed 900 steps (main)",
                "steps": sup["setup"]["steps"],
                "t_over_t_sigma": sup["setup"]["t_over_t_sigma"],
                "bandA_err_pct": sup["laplace"]["rel_error_pct"],
                "spurious_ratio": sup["spurious_current"]["ratio"],
            },
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reference_delta_p_Pa": 360.0, "rungs": rows},
                              indent=2) + "\n")
    print(f"wrote {out} ({len(rows)} rungs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
