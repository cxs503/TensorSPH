"""Guard the Wagner reference column of the committed slamming evidence bundle.

The bundle ``docs/slamming_campaign_report.json`` is the only committed evidence
of campaign node p129w1r2-n08.  Its reference column printed ``C_s,Wagner =
85.89`` (beta = 20 deg) and ``48.37`` (beta = 30 deg) while the module that
defines the reference returns ``pi^2/tan(beta) = 27.12`` and ``17.09``; the two
rows also scaled by 1.776 where ``1/tan(beta)`` demands 1.587, so no formula in
the repository could have produced them (see
``docs/audit_slamming_reference_bundle_m16961.md``).  These tests make that
class of defect fail loudly instead of shipping inside a report.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmarks" / "slamming" / "audit_wagner_reference.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("slamming_ref_audit", AUDIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("beta_deg", [5.0, 10.0, 20.0, 30.0, 45.0, 60.0])
def test_module_reference_is_pi_squared_over_tan(beta_deg):
    """``slamming_force_coefficient`` == ``pi^2/tan(beta)`` to machine accuracy."""
    from tensorsph.slamming import slamming_force_coefficient

    got = slamming_force_coefficient(1000.0, 2.0, math.radians(beta_deg), 1.0e-3)
    want = math.pi ** 2 / math.tan(math.radians(beta_deg))
    assert got == pytest.approx(want, rel=1e-12)


def test_committed_bundle_reference_column_matches_the_module():
    """No committed reference cell may disagree with the code (tol 0.1 %)."""
    mod = _audit_module()
    rep = mod.audit()
    assert rep["rows"], "the bundle's force table was not found by the audit"
    assert rep["verdict"]["mismatch"] == []
    assert rep["verdict"]["unattributable_rows"] == []


def test_committed_bundle_figures_exist_in_tracked_paths():
    """A figure link must resolve to a committed file, not a git-ignored artifact."""
    mod = _audit_module()
    rep = mod.audit()
    assert rep["figures"], "the bundle declares no figures"
    for f in rep["figures"]:
        assert f["claimed_path"] is not None, f"{f['name']} is not linked"
        assert "/artifacts/" not in f["claimed_path"], (
            f"{f['name']} points at a git-ignored artifacts/ path")
        assert f["claimed_path_exists"], f"{f['claimed_path']} is missing"
    assert rep["verdict"]["missing_figure"] == []


# NOTE: the negative control ("a doctored reference cell must be reported") was
# exercised by hand on the pre-fix bundle: audit() returned
# mismatch = [20.0, 30.0] for the printed 85.89 / 48.37, and --fix then rewrote
# both cells to 27.12 / 17.09 (see docs/audit_slamming_reference_bundle_m16961.md
# section 2). It is not kept as a unit test because it monkeypatches the module's
# DOC path, which this pytest layout does not isolate reliably.
