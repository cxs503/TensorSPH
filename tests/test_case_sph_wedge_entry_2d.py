"""Tests for CASE-SPH-WEDGE-ENTRY-2D (driver, reference handling, grading).

These tests do not run SPH: they pin the parts that can be wrong silently --
the analytic reference values, the provenance contract of ``refs.json`` and the
grading logic (a grader that cannot fail is worthless, so the negative control
pumps in a +50 % bias and requires FAIL).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("case_wedge_run", CASE / "run.py")
refs = json.loads((CASE / "refs.json").read_text())


# --------------------------------------------------------------------------- #
# 1. analytic reference (independent closed form vs in-repo implementation)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("beta_deg", [10.0, 15.0, 20.0, 30.0, 45.0])
def test_wagner_reference_matches_closed_form(beta_deg):
    """Cp_max = pi^2/(4 tan^2 beta) + 1, Cp_apex = pi/(2 tan beta), from scratch."""
    from tensorsph.wedge_entry import wagner_peak_over_q, wagner_p_centre
    beta = math.radians(beta_deg)
    C = math.pi / (2.0 * math.tan(beta))
    assert wagner_p_centre(1.0, beta, 1000.0) == pytest.approx(1000.0 * C, rel=1e-12)
    assert wagner_peak_over_q(beta) == pytest.approx(
        math.pi ** 2 / (4.0 * math.tan(beta) ** 2) + 1.0, rel=1e-10)


def test_refs_derived_is_computed_not_typed():
    ref = run.refs_derived()
    beta = math.radians(run.BETA_DEG)
    C = math.pi / (2.0 * math.tan(beta))
    assert ref["wagner"]["C"] == pytest.approx(C, rel=1e-12)
    assert ref["wagner"]["cp_max"] == pytest.approx(C * C + 1.0, rel=1e-12)
    assert ref["wagner"]["cp_centre"] == pytest.approx(C, rel=1e-12)
    assert ref["zhao_faltinsen_1993"]["status"] == "WITHHELD"
    assert ref["zhao_faltinsen_1993"]["cp_max"] is None


# --------------------------------------------------------------------------- #
# 2. provenance contract: an unread reference may never carry a number
# --------------------------------------------------------------------------- #

def test_provenance_contract():
    by_id = {r["id"]: r for r in refs["references"]}
    zf = by_id["zhao_faltinsen_1993"]
    assert zf["status"] in ("REAL", "WITHHELD")
    assert zf["table_or_figure_value_beta_30"] is None      # no fabricated number
    assert "10.1017/s002211209300028x" in zf["citation"].lower()
    assert zf["bibliographic_verification"]["crossref"]
    assert zf["withheld_reason"] and zf["what_would_close_it"]
    w = by_id["wagner_1932"]
    assert w["status"] == "REAL" and w["source_of_formula"]
    # a REAL reference must not be a placeholder
    assert "wedge_entry.py" in w["source_of_formula"]


def test_case_list_is_predeclared_and_ratio_two():
    cases = run.case_list()
    ladder = [c for c in cases if c["kind"] == "ladder"]
    assert len(ladder) >= 3
    dxs = [c["dx"] for c in ladder]
    assert dxs == sorted(dxs, reverse=True)
    for a, b in zip(dxs, dxs[1:]):
        assert a / b == pytest.approx(2.0)
    assert any(c["kind"] == "speed_control" for c in cases)
    assert all(c["beta_deg"] == run.BETA_DEG for c in cases)


# --------------------------------------------------------------------------- #
# 3. grading: positive control on the reference field, negative control on +50 %
# --------------------------------------------------------------------------- #

def _wagner_cp(tau: float, beta: float) -> float:
    C = math.pi / (2.0 * math.tan(beta))
    t = min(max(tau, 0.0), 0.999)
    s = 1.0 / math.sqrt(max(1.0 - t * t, 1e-12))
    return 2.0 * C * s - s * s + 1.0


def _synthetic_case(beta_deg: float, dx: float, scale: float = 1.0,
                    kind: str = "ladder", velocity: float = 1.0) -> dict:
    beta = math.radians(beta_deg)
    C = math.pi / (2.0 * math.tan(beta))
    taus = [0.005 + 0.01 * k for k in range(120)]
    cps = [scale * _wagner_cp(t, beta) for t in taus]
    frame = {"t": 0.02, "c_sph": C * velocity * 0.02,
             "cp_max_footprint": max(cps), "cp_max_interior": max(cps),
             "cp_centre": scale * C, "x_jet": C * velocity * 0.02,
             "max_rho_ratio": 1.01, "profile_tau": taus, "profile_cp": cps}
    return {"case_id": f"syn_{kind}_{dx}", "kind": kind,
            "config": {"beta_deg": beta_deg, "velocity": velocity, "dx": dx},
            "steps": 1, "wall_seconds": 0.1, "series": [frame] * 6}


def test_grade_passes_on_exact_reference_field():
    arts = {f"c{i}": _synthetic_case(30.0, dx) for i, dx in
            enumerate((4e-3, 2e-3, 1e-3))}
    res = run.grade(arts, run.refs_derived())
    worst = max(abs(r["err_band_peak"]) for r in res["rows"])
    # bins are 0.01 wide in tau, so the sampled peak sits a hair below the
    # analytic maximum: 1e-3 is the discretisation floor of the test field.
    assert worst < 5e-3, res["rows"]
    assert all(r["verdict_band_peak"] == "PASS" for r in res["rows"])
    assert all(r["verdict_centre"] == "PASS" for r in res["rows"])


def test_grade_fails_on_fifty_percent_bias():
    """Negative control: the grader must be able to fail."""
    arts = {f"c{i}": _synthetic_case(30.0, dx, scale=1.5) for i, dx in
            enumerate((4e-3, 2e-3, 1e-3))}
    res = run.grade(arts, run.refs_derived())
    assert all(r["verdict_band_peak"] == "FAIL" for r in res["rows"])
    assert all(r["verdict_centre"] == "FAIL" for r in res["rows"])


def test_grade_withholds_when_profile_missing():
    arts = {"c": _synthetic_case(30.0, 1e-3)}
    arts["c"]["series"] = [{**arts["c"]["series"][0], "profile_cp": [None] * 120}]
    arts["c"]["series"][0]["cp_band_peak"] = None
    res = run.grade(arts, run.refs_derived())
    assert res["rows"][0]["verdict_band_peak"] == "WITHHELD"


def test_ensemble_profile_averages_only_finite_bins():
    taus = [0.0, 0.5, 1.0]
    a = {"series": [{"profile_tau": taus, "profile_cp": [1.0, None, 3.0]},
                    {"profile_tau": taus, "profile_cp": [3.0, 2.0, 5.0]}]}
    t, cp, n = run.ensemble_profile(a)
    assert t == taus and n == 2
    assert cp[0] == pytest.approx(2.0)       # mean of 1 and 3
    assert cp[1] == pytest.approx(2.0)       # None skipped
    assert cp[2] == pytest.approx(4.0)


def test_gci_shrinks_with_observed_order():
    from tensorsph.wedge_entry import gci
    assert gci(1.0, 1.1, 2.0, order=1) > gci(1.0, 1.1, 2.0, order=2)
    assert gci(1.0, 1.0, 2.0, order=1) == pytest.approx(0.0)