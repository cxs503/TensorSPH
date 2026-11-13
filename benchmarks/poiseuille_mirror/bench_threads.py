#!/usr/bin/env python3
"""Micro-benchmark: does threading help the Poiseuille mirror kernel?

The envelope study has two points (``P3``: ``c_s`` doubled, ``P4``: ``t_end``
doubled) whose step count is twice the baseline.  Before spending wall clock on
them it is worth knowing whether the per-step cost is thread-bound or
op-overhead bound -- the problem has only a few hundred particles.

    PYTHONPATH=src:. python benchmarks/poiseuille_mirror/bench_threads.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from benchmarks.poiseuille.run import make_config  # noqa: E402
from benchmarks.poiseuille_mirror.run import run_case_mirror  # noqa: E402
from tensorsph.droplet_impact import resolve_device  # noqa: E402


def bench(threads: int, cs: float, t_end: float = 0.03) -> dict:
    torch.set_num_threads(threads)
    dev = resolve_device("cpu")
    dx = 0.1 / 12
    cfg = make_config(dx, 0.01, 1.0, dev)
    if cs != cfg.sound_speed:
        import dataclasses
        cfg = dataclasses.replace(cfg, sound_speed=cs)
    t0 = time.time()
    case = run_case_mirror(cfg, 0.1, 12 * dx, 9.81, t_end, 0.01, dev)
    wall = time.time() - t0
    return {"threads": threads, "cs": cs, "steps": case["steps"],
            "wall_clock_s": wall, "ms_per_step": 1000.0 * wall / case["steps"],
            "u_max": max(p["u_mean"] for p in case["profile"])}


def main() -> int:
    rows = [bench(t, 40.0) for t in (1, 2, 4)]
    for r in rows:
        print(f"threads={r['threads']}  {r['ms_per_step']:.2f} ms/step  "
              f"({r['wall_clock_s']:.1f}s for {r['steps']} steps)")
    base = rows[0]["ms_per_step"]
    out = {"rows": rows,
           "speedup_vs_1thread": {str(r["threads"]): base / r["ms_per_step"]
                                  for r in rows}}
    print(json.dumps(out["speedup_vs_1thread"]))
    path = REPO / "benchmarks" / "poiseuille_mirror" / "out" / "bench_threads.json"
    path.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())