"""Run the TensorSPH zero-to-one verification ladder.

    python benchmarks/verification_ladder/run.py --out artifacts/verification_ladder
    python benchmarks/verification_ladder/run.py --quick

Writes ``ladder.json`` (schema-pinned, see ``tensorsph.vv.schema``) plus four
PNGs, and exits non-zero when any rung or negative control fails. The doc
``docs/benchmarks/verification-ladder.md`` is regenerated from the JSON by
``make_doc.py``; no number is typed by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.vv.figures import make_all                    # noqa: E402
from tensorsph.vv.ladder import (LadderConfig, provenance,     # noqa: E402
                                 run_ladder, write_ladder)
from tensorsph.vv.schema import validate_ladder_json           # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "artifacts" / "verification_ladder"))
    parser.add_argument("--quick", action="store_true",
                       help="coarse parameters, for a smoke run")
    parser.add_argument("--hydro-dx", type=float, default=None)
    parser.add_argument("--tgv-dx", type=float, default=None)
    parser.add_argument("--dam-dx", type=float, default=None)
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = LadderConfig().quickify() if args.quick else LadderConfig()
    for attr, value in (("hydro_dx", args.hydro_dx), ("tgv_dx", args.tgv_dx),
                        ("dam_dx", args.dam_dx)):
        if value is not None:
            setattr(cfg, attr, value)
    if args.no_controls:
        cfg.run_controls = False

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    doc = run_ladder(cfg, command=" ".join(["verification_ladder/run.py"] + sys.argv[1:]))
    doc["wall_clock_s"] = time.time() - started
    errors = validate_ladder_json(doc)
    if errors:  # pragma: no cover - run_ladder already validates
        print("schema errors:", errors, file=sys.stderr)
        return 2
    path = write_ladder(doc, out_dir / "ladder.json")

    figures = {}
    if not args.no_figures:
        figures = {k: str(v) for k, v in make_all(doc, out_dir).items()}
        doc["figures"] = figures
        write_ladder(doc, path)

    summary = doc["summary"]
    print(json.dumps({"ladder": str(path), "figures": figures,
                      "summary": summary,
                      "errors_pct": summary["errors_pct"]}, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())