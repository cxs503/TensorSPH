"""Render ``docs/verification_lake_at_rest_bi.md`` from the frozen JSON artefact.

No number in the document is typed by hand: every table row is read from
``tests/data/lake_at_rest_bi/bi_wall.json``, which
``benchmarks/lake_at_rest_bi/run.py`` writes programmatically, and the gate table
is the JSON's own ``gates`` block.  Figures are re-generated from the same study
document by the runner's ``make_figures``.

    PYTHONPATH=src python3 benchmarks/lake_at_rest_bi/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (REPO / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lake_at_rest_bi_doc import render  # noqa: E402  (local helper, same dir)


def main() -> int:
    src = REPO / "tests" / "data" / "lake_at_rest_bi" / "bi_wall.json"
    study = json.loads(src.read_text())
    text = render(study)
    out = REPO / "docs" / "verification_lake_at_rest_bi.md"
    out.write_text(text)
    print("wrote", out, f"({len(text.splitlines())} lines)")
    from run import make_figures
    make_figures(study, REPO / "docs" / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())