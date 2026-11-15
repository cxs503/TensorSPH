"""Invariants of the dynamic surface-tension containment screen (mission 22526).

These are *invariant* assertions on the committed artefact, not a re-run of the
sweep: the sweep costs minutes, the artefact is the evidence.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = ROOT / "runs" / "droplet_tvc_containment"
DOC = ROOT / "docs" / "xval_method_selection_csf_dynamic_m22526.md"
SUMMARY = ROOT / "docs" / "reports" / "droplet_tvc_containment_m22526.json"

sys.path.insert(0, str(ROOT / "src"))
from tensorsph.capillary_oscillation import rayleigh_2d  # noqa: E402

#: the reading committed by the previous round (docs/xval_droplet_oscillation)
COMMITTED_BASELINE_ERR_PCT = -9.7609
COMMITTED_BASELINE_ESCAPED = 39


def _arms() -> list[dict]:
    files = sorted(ARMS_DIR.glob("arm_*.json"))
    if not files:
        pytest.skip("containment screen artefacts not present")
    return [json.loads(f.read_text()) for f in files]


def test_reference_period_is_the_exact_rayleigh_value():
    t = 2.0 * math.pi / rayleigh_2d(0.072, 1000.0, 0.05, 2)
    assert abs(t - 3.3798156) < 1e-6


def test_baseline_reproduces_the_committed_reading():
    """The screen must start from the same physics, not a re-tuned variant."""
    base = [a for a in _arms() if a["arm"] == "repro_baseline_no_tvc"]
    assert len(base) == 1
    a = base[0]
    assert a["tvc_alpha"] == 0.0
    assert a["escaped_particles"] == COMMITTED_BASELINE_ESCAPED
    assert abs(a["containment_max_r_over_R"] - 4.2608) < 0.01
    # 1 % reproduction gate on the period reading
    assert abs(a["period_err_pct"] - COMMITTED_BASELINE_ERR_PCT) < 0.1


def test_no_tvc_strength_produces_a_usable_measurement():
    """The falsified hypothesis, as an invariant: TVC never keeps the drop.

    If a future change makes transport-velocity correction contain the drop,
    this test fails and the claim in the report must be revisited -- that is the
    point of pinning it.
    """
    tvc = [a for a in _arms() if a["tvc_alpha"] > 0.0]
    assert len(tvc) >= 9

    def finite(x):
        return isinstance(x, float) and math.isfinite(x)

    for a in tvc:
        assert a["usable_measurement"] is False, a["arm"]
        assert a["gates"]["contained"] is False, a["arm"]
        if finite(a["containment_max_r_over_R"]):
            assert a["containment_max_r_over_R"] > 1.05, a["arm"]


def test_tvc_never_increases_containment_over_the_baseline():
    """Every TVC arm escapes at least as many particles as the no-TVC run."""
    arms = _arms()
    base = next(a for a in arms if a["tvc_alpha"] == 0.0)
    for a in arms:
        if a["tvc_alpha"] > 0.0:
            assert a["escaped_particles"] >= base["escaped_particles"], a["arm"]


def test_report_is_generated_from_the_artefact():
    """Anti-drift gate: the doc and summary are a fresh build, byte for byte."""
    if not DOC.exists():
        pytest.skip("report not built")
    r = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "surface_tension" /
                             "report_tvc_containment.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr