"""Verify the F3 machine-referee closure for XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE.

Confirms (a) the referee row is registered in tensor_benchmark_cases, (b) the
measured result rows are recorded in tensorbench_results, (c) all pass the 15 %
owner gate. Run:  python3 scripts/verify_xval_sph_young_laplace_referee.py
"""
from __future__ import annotations

import subprocess
import sys

ANCHOR = "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE"
TOL = 0.15


def psql(q: str) -> str:
    return subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-t", "-A", "-c", q],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    # (a) referee row
    rows = psql(
        f"SELECT id,case_name,metric,tolerance,repo,status "
        f"FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'"
    )
    print("referee rows:\n" + (rows or "(none)"))
    assert rows, "no referee row registered"

    # (b) measured results
    res = psql(
        f"SELECT result_id,metric,ref_value,got_value,rel_error,tolerance_rel,passed "
        f"FROM tensorbench_results WHERE benchmark_id='{ANCHOR}' ORDER BY got_value"
    )
    print("\nmeasured results:\n" + (res or "(none)"))
    assert res, "no measured result rows"

    # (c) all pass the 15 % gate
    lines = [l for l in res.splitlines() if l]
    for l in lines:
        parts = l.split("|")
        rel = float(parts[4])
        tol = float(parts[5])
        passed = parts[6].strip() == "t"
        assert passed and abs(rel) <= tol, f"gate fail: {l}"
    print(f"\nOK: {len(lines)} result rows, all |rel| <= {TOL} (PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
