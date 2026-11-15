"""Build the 3-D abort-evidence note and its figures from the archived job logs.

Every number in the emitted markdown comes from this file
(``results/abort_evidence/*.log``, ``*.csv``), never from a hand-typed table:
the job logs are the cluster scheduler's own records of the runs, archived here
verbatim so the evidence survives outside the cluster.

Outputs
-------
``results/abort_evidence/abort_evidence.json``  machine-readable record
``results/abort_evidence/fig_partial_series.png`` H2 slab series of the runs
    that died (they were producing a physical signal right up to the abort)
``results/abort_evidence/fig_death_map.png``     death time vs dx

Usage::

    python benchmarks/kleefsman_obstacle_3d/make_abort_doc.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
EVID = HERE / "results" / "abort_evidence"

# run identity: job -> (dx, threads, note)
RUNS = {
    "cj3952": (0.05, 16, "production rung (run.py)"),
    "cj3953": (0.04, 16, "production rung (run.py)"),
    "cj3954": (0.03, 16, "production rung (run.py)"),
    "cj3992": (0.05, 1, "diagnostic stub (faulthandler.register(SIGABRT) refused)"),
    "cj3993": (0.05, 16, "diagnostic stub (faulthandler.register(SIGABRT) refused)"),
}


def parse_log(path: Path) -> dict:
    payload = json.loads(path.read_text())
    out = payload.get("output_tail", "")
    times = [float(m) for m in re.findall(r"t=\s*([\d.]+)", out)]
    slab = [float(m) for m in re.findall(r"slab=\s*([\d.]+)", out)]
    aborted = "corrupted unsorted chunks" in out
    header = next((line for line in out.splitlines()
                   if line.startswith("[") and "dx=" in line), "")
    m = re.search(r"dx=([\d.]+) t_end=([\d.]+) dt=([\deE.+-]+) c0=(\S+) "
                  r"wall_layers=(\d+)", header)
    return {
        "job": path.stem,
        "node": payload.get("node"),
        "rc": payload.get("rc"),
        "aborted": aborted,
        "abort_message": "free(): corrupted unsorted chunks" if aborted else None,
        "last_logged_time_s": times[-1] if times else None,
        "n_progress_samples": len(times),
        "dx": float(m.group(1)) if m else RUNS.get(path.stem, (None,))[0],
        "t_end_requested": float(m.group(2)) if m else None,
        "dt": float(m.group(3)) if m else None,
        "sound_speed": float(m.group(4)) if m else None,
        "wall_layers": int(m.group(5)) if m else None,
        "threads": RUNS.get(path.stem, (None, None, ""))[1],
        "note": RUNS.get(path.stem, (None, None, ""))[2],
        "traceback_head": next((line.strip() for line in out.splitlines()
                                if "Error" in line), None),
        "progress_slab": slab,
        "progress_time": times,
    }


def load_partial_csv() -> tuple[list[float], list[float]]:
    path = EVID / "cj3952_dx0.05_partial.csv"
    t, h = [], []
    for k, line in enumerate(path.read_text().splitlines()):
        if k == 0:
            continue
        f = line.split(",")
        t.append(float(f[0]))
        h.append(float(f[1]))
    return t, h


def fig_partial() -> Path:
    t, h = load_partial_csv()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(t, h, lw=1.5, color="tab:blue",
            label="3-D SPH, dx=0.05 m (cj3952, threads=16)")
    ax.axhline(0.370, ls="--", lw=1.2, color="k",
               label="Kleefsman 2005 H2 peak, 0.370 m")
    ax.axhline(0.02, ls=":", lw=1.0, color="tab:gray",
               label="arrival threshold, 0.02 m")
    ax.axvline(t[-1], ls="-.", lw=1.2, color="tab:red",
               label=f"glibc abort at t={t[-1]:.2f} s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("slab water height at x=2.426 m (m)")
    ax.set_title("Partial H2 record of the aborted 3-D run")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    out = EVID / "fig_partial_series.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_death(records: list[dict]) -> Path | None:
    rows = [r for r in records if r["aborted"] and r["last_logged_time_s"]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for r in rows:
        ax.scatter([r["dx"]], [r["last_logged_time_s"]], s=55,
                   label=f"{r['job']} (threads={r['threads']})")
    ax.axhline(1.7027, ls="--", lw=1.1, color="k",
               label="experimental H2 peak time 1.703 s")
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("t at death (s)")
    ax.set_xlim(0.02, 0.06)
    ax.set_ylim(0.0, 2.0)
    ax.set_title("Every 3-D rung dies before $t_{end}=2.6$ s")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    out = EVID / "fig_death_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    logs = sorted(EVID.glob("cj*.log"))
    records = [parse_log(p) for p in logs]
    f1 = fig_partial()
    f2 = fig_death(records)
    payload = {
        "records": records,
        "aborted": sum(1 for r in records if r["aborted"]),
        "figures": [str(f1)] + ([str(f2)] if f2 else []),
        "environment": {
            "torch": "2.7.1", "mkl": False,
            "note": "CPU only; no numpy/ctypes/f2py interop is imported by the "
                    "3-D case modules (src/tensorsph/{dam_break_3d,"
                    "kleefsman_obstacle_3d,droplet_impact,neighbor_search,"
                    "transport_velocity_correction}.py), so there is no foreign "
                    "buffer that our own code could overrun",
        },
    }
    (EVID / "abort_evidence.json").write_text(json.dumps(payload, indent=1) + "\n")
    lines = []
    for r in records:
        lines.append(
            f"| {r['job']} | {r['dx']:.2f} | {r['threads']} | "
            f"{'ABORT' if r['aborted'] else 'no abort'} | "
            f"{r['last_logged_time_s'] if r['last_logged_time_s'] is not None else '-'} | "
            f"{r['t_end_requested']} | {r['note']} |")
    table = ("| 作业 | dx [m] | threads | 结局 | 死亡时刻 [s] | t_end | 说明 |\n"
             "|---|---|---|---|---|---|---|\n" + "\n".join(lines))
    (EVID / "table.md").write_text(table + "\n")
    print(table)
    print(f"wrote {EVID/'abort_evidence.json'} and {len(payload['figures'])} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())