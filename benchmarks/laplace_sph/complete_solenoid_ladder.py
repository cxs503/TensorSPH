"""Complete the one-shot solenoidal-CSF diagnostic at the missing resolutions.

``diagnose_csf_solenoid.py`` measures, for **all four** ladder levels in one
process, the part of the lattice CSF traction that the projection operator of
the same lattice cannot annihilate.  The shipped record only holds ``R/dx`` = 8
and 12 (the process that produced it was stopped), so its fitted decay order
``-1.311`` rests on a two-point fit -- far too few points to support the
negative conclusion "the spurious-current source does not refinish away".

This tool runs only the *missing* levels and merges them into the existing
record through ``diagnose_csf_solenoid.summarise`` (the authoritative gate
logic -- no number on the page is computed here):

    PYTHONPATH=src python benchmarks/laplace_sph/complete_solenoid_ladder.py \
        --levels 16,24 --threads 8

Records are **per level** (``results_dynamic_ladder/solenoid_levels/Rdx<N>.json``)
so a level that dies or is killed leaves every earlier level usable, and a
re-run skips what is already measured.  ``--levels 0`` re-merges without
measuring anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnose_csf_solenoid as D  # noqa: E402

LEVELDIR = HERE / "results_dynamic_ladder" / "solenoid_levels"


def load_existing() -> list[dict]:
    """Levels already recorded by the original one-shot diagnostic."""
    if not D.JSON_OUT.exists():
        return []
    return list(json.loads(D.JSON_OUT.read_text()).get("levels", []))


def load_level(level: int) -> dict | None:
    p = LEVELDIR / ("Rdx%d.json" % level)
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="16,24")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    LEVELDIR.mkdir(parents=True, exist_ok=True)
    for lvl in [int(t) for t in args.levels.split(",") if t.strip()]:
        if load_level(lvl) is not None:
            print("[R/dx=%d] already measured, skipping" % lvl, flush=True)
            continue
        rec = D.measure_level(lvl)
        (LEVELDIR / ("Rdx%d.json" % lvl)).write_text(json.dumps(rec, indent=1))

    levels = {float(r["R_over_dx"]): r for r in load_existing()}
    for lvl in D.LEVELS:
        got = load_level(lvl)
        if got is not None:           # a fresh measurement wins over the old copy
            levels[float(got["R_over_dx"])] = got
    merged = [levels[k] for k in sorted(levels)]
    s = D.summarise(merged)
    D.OUTDIR.mkdir(parents=True, exist_ok=True)
    D.JSON_OUT.write_text(json.dumps(s, indent=1))
    print("wrote", D.JSON_OUT, "levels", s["setup"]["levels_R_over_dx"],
          "not measured", s["levels_not_measured"], flush=True)
    for k, v in s["gate_verdicts"].items():
        print("  %-28s measured=%-12.4g declared=%-8.4g pass=%s"
              % (k, v["measured"], v["declared"], v["pass"]), flush=True)
    if not args.no_doc:
        D.make_figure(s, D.FIG)
        D.make_doc(s)


if __name__ == "__main__":
    main()