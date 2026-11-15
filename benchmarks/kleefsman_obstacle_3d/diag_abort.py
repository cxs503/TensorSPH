"""Diagnostic wrapper for the 3-D case's ``free(): corrupted unsorted chunks`` abort.

The production driver (:mod:`run`) dies intermittently with a glibc heap abort
after 1.5-2 s of simulated time, throwing away the run and leaving the case
verdict ``WITHHELD``.  ``run.py`` cannot catch it: the abort happens inside the
C allocator, not in Python code, so ``except BaseException`` never sees it.

This wrapper adds nothing to the physics; it adds the instrumentation needed to
attribute the abort:

* ``faulthandler.enable()`` plus a ``faulthandler``-style ``SIGABRT`` dump --
  CPython refuses ``faulthandler.register(SIGABRT)`` ("signal 6 cannot be
  registered", verified: jobs cj3992/cj3993 died on that call), so the abort
  backtrace is obtained by re-raising it through ``signal.signal`` and letting
  ``faulthandler.dump_traceback`` run on the Python frame that was executing;
* ``MALLOC_CHECK_`` / ``MALLOC_PERTURB_`` / ``MALLOC_ARENA_MAX`` /
  ``OMP_NUM_THREADS`` are recorded (and honoured by glibc if set), so strict
  glibc checking and arena pinning are runnable without editing this file;
* every output sample is streamed to CSV exactly as ``run.py`` does, so a run
  that dies still leaves a comparable partial record (the ladder only uses
  complete runs).

Usage::

    python benchmarks/kleefsman_obstacle_3d/diag_abort.py --dx 0.05 \
        --t-end 2.6 --threads 1 --tag t1 --out-dir .../diag
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.kleefsman_obstacle_3d import (  # noqa: E402
    Kleefsman3DConfig,
    run_case,
    score,
)

HERE = Path(__file__).resolve().parent


def _install_abort_dump() -> None:
    """Dump every thread's Python stack, then let the abort take its course."""
    faulthandler.enable()

    def _handler(signum, _frame):
        sys.stderr.write(f"\n[diag] caught signal {signum}, dumping stacks\n")
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(128 + int(signum))

    try:
        signal.signal(signal.SIGABRT, _handler)
    except (OSError, ValueError) as exc:      # pragma: no cover - platform
        print(f"[diag] SIGABRT handler unavailable: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--t-end", type=float, default=2.6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sound-speed", type=float, default=20.0)
    ap.add_argument("--wall-layers", type=int, default=3)
    ap.add_argument("--output-every", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    _install_abort_dump()

    tag = args.tag or f"diag_dx{args.dx:g}_t{args.threads}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Kleefsman3DConfig(dx=args.dx, t_end=args.t_end,
                           sound_speed=args.sound_speed,
                           wall_layers=args.wall_layers,
                           output_every=args.output_every)
    cfg.dt = 0.9 * cfg.dt_from_courant()

    env = {k: os.environ.get(k) for k in
           ("MALLOC_CHECK_", "MALLOC_PERTURB_", "MALLOC_ARENA_MAX",
            "OMP_NUM_THREADS", "PYTHONFAULTHANDLER")}
    print(f"[{tag}] dx={cfg.dx:g} dt={cfg.dt:.3e} threads={args.threads} "
          f"torch={torch.__version__} env={env}", flush=True)

    csv_path = out_dir / f"{tag}_time_series.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_s", "slab_height_m", "height_operator_m",
                         "front_x_m", "n_bypass"])
        fh.flush()

        def on_sample(_i: int, row: dict) -> None:
            writer.writerow([f"{row['t_s']:.6f}", f"{row['slab_height_m']:.6f}",
                             f"{row['height_operator_m']:.6f}",
                             f"{row['front_x_m']:.6f}",
                             f"{row['n_bypass']:.0f}"])
            fh.flush()

        t0 = time.time()
        try:
            run = run_case(cfg, device="cpu", on_sample=on_sample)
        except BaseException:                              # noqa: BLE001
            print(f"[{tag}] ABORTED after {time.time() - t0:.0f} s", flush=True)
            traceback.print_exc()
            return 1

    verdict = score(run)
    verdict.update({"tag": tag, "complete": True,
                    "threads": args.threads,
                    "wall_clock_s": round(time.time() - t0, 1),
                    "env": env, "diagnostic": True})
    (out_dir / f"{tag}.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(f"[{tag}] completed t={run.simulated_time:.3f} s steps={run.steps} "
          f"peak={verdict['run']['peak_height_m']:.4f} m "
          f"wall={verdict['run']['wall_clock_s']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())