"""Build the slamming cost table + figure from the recorded probe logs.

Every number in ``docs/slamming_evidence_blocker_m14940.md`` comes from this
script reading ``docs/slamming_cost/logs/*.log``: the probe prints
``[probe] step k/n t=... wall_s`` every 20 steps and `run.py`'s profile run
prints the per-step time; nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
STEP_RE = re.compile(r"\[probe\] step (\d+)/(\d+) t=(-?[\d.]+) ([\d.]+) s")


def parse(path: Path) -> dict:
    rows = [(int(k), int(n), float(t), float(w))
            for k, n, t, w in STEP_RE.findall(path.read_text())]
    out = {"log": path.name, "rows": rows, "n_steps": rows[0][1] if rows else None}
    if len(rows) >= 2:
        seg = [(rows[i + 1][0] - rows[i][0], rows[i + 1][3] - rows[i][3])
               for i in range(len(rows) - 1)]
        out["segment_s_per_step"] = [round(dw / dk, 3) for dk, dw in seg]
    return out


def main() -> int:
    runs = {p.name: parse(p) for p in sorted(LOGS.glob("probe_*.log"))}
    prof = (LOGS / "profile_dx4mm.log").read_text()
    m = re.search(r"40 steps ([\d.]+) s \(([\d.]+) s/step\)", prof)
    flat = {"log": "profile_dx4mm.log", "steps": 40,
            "wall_s": float(m.group(1)), "s_per_step": float(m.group(2)),
            "particles": 3000, "dx_mm": 4.0}
    table = {"runs": runs, "flat_dx4mm": flat}
    (HERE / "cost_table.json").write_text(json.dumps(table, indent=1))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, r in runs.items():
        if not r["rows"]:
            continue
        ax.plot([r0[0] for r0 in r["rows"]], [r0[3] for r0 in r["rows"]],
                marker="o", label=f"{name.replace('probe_', '').replace('.log', '')}"
                                 f" (n={r['n_steps']} steps)")
    ax.set_xlabel("time step completed")
    ax.set_ylabel("cumulative wall clock [s]")
    ax.set_title("TensorSPH slamming probe: pre-impact cost blow-up\n"
                 "(8 threads, CPU float64; load15 ~85 on a 128-core box)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "slamming_cost_growth.png", dpi=150)

    print(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
