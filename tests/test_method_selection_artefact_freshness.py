"""Every published method-selection artefact must be a fresh build of the registry.

Why this exists
---------------
The anchor ``XVAL-SPH-METHOD-SELECTION-2026-11-12`` publishes *four* derived
artefacts from one source of truth (``docs/reports/method_selection_registry.json``):

===============================  =========================================
producer                          artefacts
===============================  =========================================
``build_registry.py``             registry, matrix, tables, report card
``audit_quantity_coverage.py``    coverage audit json + md + png
``audit_decision_fragility.py``   fragility json (leave-one-record-out)
``strict_bound_profiles.py``      strict-bound json + md + png
===============================  =========================================

Fragility grew an anti-drift check in mission 21797 and the registry in 18056,
but the other two had none.  Measured on the merged tree at mission 22608: when
the registry grew from 29 to 30 records (the CSF quasi-static 45.69 % record),
``audit_quantity_coverage.py`` kept claiming 29 records and
``strict_bound_profiles.py`` kept quoting the *superseded* 23.8 % as
``csf_surface_tension``'s worst live accuracy record, while the registry said
45.69 %.  The number a reader saw in
``docs/xval_method_selection_strict_bound_m21537.md`` was therefore not the
number the selector was using, and nothing in the test suite noticed.

This test closes the class, not the instance: it runs **every** producer in its
own ``--check`` mode, so adding a fifth artifact means adding one row here
instead of hoping the next author remembers.

The check is a subprocess because the producers are scripts with side effects;
``--check`` is their documented "write nothing, exit non-zero on drift" mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: (producer, expected marker in stdout when the artefacts are fresh)
PRODUCERS = (
    ("benchmarks/method_selection/build_registry.py", "up to date"),
    ("benchmarks/method_selection/audit_quantity_coverage.py", "OK"),
    ("benchmarks/method_selection/audit_decision_fragility.py", "OK"),
    ("benchmarks/method_selection/strict_bound_profiles.py", "up to date"),
)


def _run_check(script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, str(ROOT / script), "--check"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.parametrize("script,marker", PRODUCERS, ids=[p[0] for p in PRODUCERS])
def test_producer_artefacts_are_a_fresh_build(script: str, marker: str) -> None:
    proc = _run_check(script)
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, (
        f"{script} reports stale artefacts (rc={proc.returncode}); rerun it "
        f"without --check and commit the result.\n{combined[-2000:]}"
    )
    assert marker.lower() in combined.lower(), (
        f"{script} exited 0 but did not report freshness ({marker!r}):\n"
        f"{combined[-2000:]}"
    )


def test_registry_record_count_is_the_one_the_audits_quote() -> None:
    """The two stale artefacts above differed only by a record count.

    Pin the cross-reference explicitly so a *silent* count mismatch cannot hide
    behind a producer that happens not to print it.
    """
    import json

    registry = json.loads(
        (ROOT / "docs/reports/method_selection_registry.json").read_text()
    )
    coverage = json.loads(
        (ROOT / "docs/reports/method_selection_coverage_audit_m20432.json").read_text()
    )
    assert coverage["registry_records"] == len(registry["records"])