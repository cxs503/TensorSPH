"""Isolate the ``free(): corrupted unsorted chunks`` abort seen on the wedge ladder.

Background (m22495).  Two cluster shards of the dyadic ladder died with
``free(): corrupted unsorted chunks`` (SIGABRT, rc=134):

* cj3803 — ``dx = 0.5 mm``, ``--threads 16``
* cj4103 — ``dx = 1 mm``,   ``--threads 8``   (compact digest path)
* cj4106 — ``dx = 2 mm``,   ``--threads 8``   (``--dump-frames`` path)

The previous reading of that evidence was "finest grids are memory-fragile /
resolution-dependent".  That reading rests on an untested attribution: the
crash was never separated from the **thread count**.  Every committed artefact
of this case was produced at ``--threads 4`` (the default); every abort
happened at ``--threads >= 8``.  This probe makes that confound checkable
instead of assumed:

1. ``libs``   — how many OpenMP runtimes are mapped into the process (a
   duplicated libgomp/libomp is the textbook cause of exactly this abort);
2. ``torch``  — a *minimal* torch workload (no SPH, no wedge code at all) at
   each thread count: if this aborts, the defect is in the runtime, not in the
   solver;
3. ``wedge``  — the real shard command at the same thread counts, for the
   counter-check.

Each scenario runs in its own child process with ``-X faulthandler`` and
``MALLOC_CHECK_=3``, so one abort does not hide the others, and the child's
return code is the datum (0 = clean exit, negative = killed by signal,
134 = glibc abort).  Nothing here is a physics run: the point is attribution.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/heap_probe.py \
        --out docs/artifacts/case_sph_wedge_entry_2d_heap_probe/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Thread counts of interest: 4 is the default that produced every committed
# artefact, 8/16 are the two counts that aborted.
DEFAULT_THREADS = (1, 2, 4, 8, 16)


def _child_env() -> dict:
    env = dict(os.environ)
    # glibc's own checker: turns a latent heap overflow into an immediate,
    # attributed abort instead of a later confusing one.
    env["MALLOC_CHECK_"] = "3"
    env["MALLOC_PERTURB_"] = "165"
    env.pop("OMP_NUM_THREADS", None)  # keep the runtime free to follow torch
    return env


def _minimal_torch_src(threads: int) -> str:
    return (
        "import torch\n"
        f"torch.set_num_threads({threads})\n"
        "x = torch.randn(1200, 1200,dtype=torch.float64)\n"
        "acc = torch.zeros(1, dtype=torch.float64)\n"
        "for _ in range(8):\n"
        "    x = (x @ x) * 1e-3\n"
        "    acc = acc + x.diagonal().sum().reshape(1)\n"
        "print('ok', float(acc))\n"
    )


def _wedge_src(dx: float, frames: int, threads: int) -> str:
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import ensemble_peak as ep\n"
        "raise SystemExit(ep.main(['--dx', repr(%r), '--frames', '%d',"
        " '--threads', '%d', '--out', '/tmp/heap_probe_wedge']))\n"
        % (dx, frames, threads)
    )


def _run(name: str, src: str, timeout: float, extra_env: dict | None = None) -> dict:
    env = _child_env()
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run([sys.executable, "-X", "faulthandler", "-c", src],
                              cwd=str(REPO), env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = "timeout"
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    tail = [ln for ln in err.strip().splitlines() if ln.strip()][-4:]
    return {"scenario": name, "rc": rc, "stdout_tail": out.strip()[-300:],
            "stderr_tail": tail,
            "heap_corruption": any("corrupted" in ln or "double free" in ln
                                   or "invalid pointer" in ln for ln in tail),
            "signal": (rc if isinstance(rc, str) else ("SIGABRT" if rc == -6 else None))}


def openmp_map() -> dict:
    """Which OpenMP runtimes are mapped, and how many distinct copies."""
    probe = (
        "import os, torch, torch.utils\n"
        "torch.set_num_threads(8)\n"
        "maps = open('/proc/self/maps').read()\n"
        "hits = [l.split()[-1] for l in maps.splitlines()\n"
        "        if 'libgomp' in l or 'libomp' in l or 'libiomp' in l]\n"
        "print('\\n'.join(sorted(set(hits))))\n"
    )
    env = _child_env()
    env["LD_DEBUG"] = "libs"
    proc = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO),
                          env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, timeout=600)
    libs = []
    for line in proc.stderr.splitlines():
        if "libgomp" in line or "libomp" in line:
            libs.append(line.strip()[-160:])
    return {
        "mapped_libs": [ln for ln in proc.stdout.strip().splitlines() if ln.strip()],
        "loader_lines": libs[:6],
        "parallel_info": subprocess.run(
            [sys.executable, "-c",
             "import torch;print(torch.__config__.parallel_info())"],
            cwd=str(REPO), env=_child_env(), stdout=subprocess.PIPE,
            text=True, timeout=300).stdout,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/artifacts/case_sph_wedge_entry_2d_heap_probe/probe.json")
    ap.add_argument("--threads", type=int, nargs="+", default=list(DEFAULT_THREADS))
    ap.add_argument("--dx", type=float, default=2.0e-3, help="grid for the wedge counter-check")
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--skip-wedge", action="store_true",
                    help="runtime-probe only (no physics run)")
    args = ap.parse_args(argv)

    report: dict = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/heap_probe.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "range": {},
        "runtime": openmp_map(),
        "scenarios": [],
    }

    for t in args.threads:
        report["scenarios"].append(
            _run(f"torch_minimal@threads={t}", _minimal_torch_src(t), args.timeout))
    for t in args.threads:
        report["scenarios"].append(
            _run(f"import_only@threads={t}",
                 f"import torch; torch.set_num_threads({t}); print('ok')", args.timeout))
    if not args.skip_wedge:
        for t in args.threads:
            report["scenarios"].append(
                _run(f"wedge@dx={args.dx:g},frames={args.frames},threads={t}",
                     _wedge_src(args.dx, args.frames, t), args.timeout))

    # Attribution: the same runtime probe that aborts, with the OpenMP runtime
    # pinned to a single thread, is the candidate remedy - test it explicitly.
    report["candidate_remedies"] = []
    for env in ({"OMP_NUM_THREADS": "1"}, {"OMP_WAIT_POLICY": "PASSIVE"},
                {"MALLOC_CHECK_": "0", "MALLOC_PERTURB_": "0"}):
        rec = _run(f"torch_minimal@threads=8 with {env}", _minimal_torch_src(8),
                   args.timeout, extra_env=env)
        rec["env"] = env
        report["candidate_remedies"].append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    # Compact digest: cluster logs are tail-truncated at 8 kB.
    print(json.dumps({k: report[k] for k in ("machine", "python")}))
    print("openmp mapped:", report["runtime"]["mapped_libs"])
    for s in report["scenarios"] + report["candidate_remedies"]:
        print(f"[{s['rc']!s:>8}] {s['scenario']}"
              f"{'  <-- HEAP CORRUPTION' if s['heap_corruption'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())