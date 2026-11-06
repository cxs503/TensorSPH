"""Re-render ``laplace_pressure_jump.png`` from the committed JSON report.

The figure is a *derived* artefact: every number in it is injected from
``validation_report.json`` (which is the committed evidence), so the plot can be
rebuilt without paying for the 4 min SPH run that produced the report.  Keeping
this path separate from ``run.py`` means a fresh checkout can always materialise
the figure referenced by ``docs/verification_laplace_pressure_jump.md`` --
``tests/test_laplace_pressure_jump.py::test_figure_exists`` uses it as a
self-heal when the binary is missing (see m4786: the PNG was never committed
although the document already linked it, which turned main red).

    PYTHONPATH=src python benchmarks/laplace_pressure_jump/make_figure.py

Rendering is deterministic: ``make_figure`` only reads the report and uses the
Agg backend, so the output depends on the JSON, not on run order.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / "validation_report.json"
FIGURE = HERE / "laplace_pressure_jump.png"


def _runner():
    """Import the sibling benchmark runner (single source of truth for the plot)."""
    spec = importlib.util.spec_from_file_location("lpj_run_figure", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render(out: Path = FIGURE, report: Path = REPORT) -> Path:
    """Write the figure and return its path (``out`` is created if absent)."""
    data = json.loads(report.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    return _runner().make_figure(data, out_dir=out.parent)


def main() -> None:
    path = render()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    print(f"ROOT={ROOT}")


if __name__ == "__main__":
    main()
