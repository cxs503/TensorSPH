"""Run the WCSPH tests under coverage without tripping torch's import guard.

``torch`` raises ``RuntimeError: function '_has_torch_function' already has a
docstring`` when its C extension is imported while a ``sys.settrace`` tracer is
active (the tracer changes the frame at import time).  Importing torch *before*
coverage starts sidesteps it, so the measurement stays a real measurement.

Usage: python coverage_run.py <json-out> <test files...>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # noqa: F401  (must precede coverage for the reason above)


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    files = argv[1:]
    import os
    import coverage
    import pytest

    package = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "..", "src", "tensorsph", "wcsph"))
    cov = coverage.Coverage(source=[package])
    cov.start()
    code = pytest.main([*files, "-q", "--no-header", "-p", "no:cacheprovider"])
    cov.stop()
    cov.json_report(outfile=str(out))
    data = json.loads(out.read_text())
    percent = (data.get("totals", {}).get("percent_covered")
                       or data.get("totals", {}).get("percent", 0.0)) / 100.0
    print(json.dumps({"pytest_exit": int(code), "coverage": percent,
                      "files": files}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
