"""CASE-SPH-POISEUILLE-MORRIS: independent-audit and document-binding tests.

These tests guard the *evidence chain* of the anchor rather than the solver:

* `audit_independent.recompute` must reproduce analytic metrics from raw rows
  to machine precision, and must **fail** when the recorded metrics are
  tampered with (otherwise the audit would be vacuous);
* `make_doc.assert_coverage` must refuse to render a document from an
  incomplete (chunked, partially merged) sweep instead of emitting tables with
  a header and no rows;
* when the production artefact is present, the independent audit must pass and
  every number in the rendered document must bind back to the artefact.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from benchmarks.poiseuille_morris import audit_independent as audit
from benchmarks.poiseuille_morris.make_doc import assert_coverage

RESULTS = Path("runs/poiseuille_morris/results.json")
DOC = Path("docs/benchmarks/poiseuille-morris.md")

H, NU, G = 0.1, 1.0, 800.0        # the Re_c = 0.1 channel used by the case
RES = 8


def synthetic_case(e_scale: float = 1.0) -> dict:
    """A case dict with rows *built* from the closed form plus a known error.

    The measured profile is the exact solution scaled by (1 + e_scale*1e-3), so
    every metric has a value this test can compute by hand.
    """
    dx = H / RES
    rows = []
    for i in range(RES):
        y = (i + 0.5) * dx
        u_ref = audit.u_exact(y, H, G, NU)
        rows.append({
            "row": i,
            "y": y,
            "y_over_H": y / H,
            "u_measured": u_ref * (1.0 + e_scale * 1e-3),
            "u_reference": u_ref,
        })
    return {
        "route": "isph_projection",
        "reynolds_c": (G * H ** 2 / (8 * NU)) * H / NU,
        "resolution_H_over_dx": RES,
        "h_over_dx": 1.90,
        "columns": 12,
        "dx": dx,
        "viscosity": NU,
        "body_force": G,
        "metrics": {
            "u_max_reference": G * H ** 2 / (8 * NU),
            "centre_error_pct": 0.0,          # filled by the test where needed
            "interior": {"l2_over_umax_pct": 0.0, "n_rows": 0,
                         "max_abs_over_umax_pct": 0.0},
            "rows": rows,
        },
        "gates": {},
    }


def test_recompute_matches_analytic_to_machine_precision():
    case = synthetic_case(e_scale=1.0)
    r = audit.recompute(case)
    assert r["height"] == pytest.approx(H, rel=1e-12)
    assert r["row_ref_max_dev"] == 0.0
    # measured = u_ref*(1+1e-3)  ->  err/u_max is +1e-3 * u_ref/u_max everywhere
    u_max = G * H ** 2 / (8 * NU)
    interior = [row for row in case["metrics"]["rows"] if 0.15 <= row["y_over_H"] <= 0.85]
    expect = 100.0 * math.sqrt(sum(((1e-3 * row["u_reference"]) / u_max) ** 2
                                   for row in interior) / len(interior))
    assert r["l2_over_umax_pct"] == pytest.approx(expect, rel=1e-12)
    assert r["n_interior_rows"] == len(interior)
    assert r["max_abs_over_umax_pct"] == pytest.approx(
        100.0 * max(abs(1e-3 * row["u_reference"]) / u_max for row in interior), rel=1e-12)


def test_recompute_catches_an_offset_inside_the_scored_band():
    """A rigid offset (the classic wall-slip artefact) must show up as error.

    The offset is placed on a row inside the scored interior band, so the
    metric cannot absorb it: an audit that only looks at y/H = 0.5 would miss a
    wall slip, which is why the band metric is the one that is bound.
    """
    case = synthetic_case(e_scale=0.0)
    mid = min(case["metrics"]["rows"], key=lambda r: abs(r["y_over_H"] - 0.5))
    mid["u_measured"] += 0.05 * case["metrics"]["u_max_reference"]
    r = audit.recompute(case)
    assert r["max_abs_over_umax_pct"] > 4.9


def test_audit_is_not_vacuous_on_tampered_artefact(tmp_path):
    """Perturb one recorded metric: the audit must raise a blocking finding."""
    case = synthetic_case(e_scale=1.0)
    rec = audit.recompute(case)
    case["metrics"]["centre_error_pct"] = rec["centre_error_pct"]
    case["metrics"]["interior"] = {
        "n_rows": rec["n_interior_rows"],
        "l2_over_umax_pct": rec["l2_over_umax_pct"],
        "max_abs_over_umax_pct": rec["max_abs_over_umax_pct"],
    }
    payload = {
        "anchor": "CASE-SPH-POISEUILLE-MORRIS",
        "settings": {"reynolds_ladder": [case["reynolds_c"]]},
        "cases": {"isph_projection": [case], "wcsp_production": []},
        "controls": {"zero_forcing": {"max_abs_u_measured": 0.0, "pass": True},
                     "repeat_headline": {"profiles_identical": True}},
        "verdict": {"isph": {str(case["reynolds_c"]): {
            "H_over_dx": RES,
            "centre_error_pct": rec["centre_error_pct"],
            "l2_interior_pct": rec["l2_over_umax_pct"],
            "pass": True}}, "wcsp": {}},
        "summary": {},
    }
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload))
    good = audit.run(results, None, tmp_path / "audit_ok.json")
    assert good.blocking() == [], good.blocking()

    tampered = copy.deepcopy(payload)
    tampered["cases"]["isph_projection"][0]["metrics"]["interior"]["l2_over_umax_pct"] *= 0.5
    results.write_text(json.dumps(tampered))
    bad = audit.run(results, None, tmp_path / "audit_bad.json")
    assert bad.blocking(), "audit accepted a metric that disagrees with the raw rows"


def test_make_doc_guard_rejects_incomplete_sweep():
    case = synthetic_case()
    payload = {
        "settings": {"reynolds_ladder": [case["reynolds_c"]]},
        "cases": {"isph_projection": [case, dict(case, resolution_H_over_dx=16)],
                  "wcsp_production": []},
        "controls": {"zero_forcing": {}, "repeat_headline": {}},
        "summary": {},
        "verdict": {"isph": {}, "wcsp": {}},
    }
    with pytest.raises(SystemExit) as exc:
        assert_coverage(payload)
    msg = str(exc.value)
    assert "incomplete sweep" in msg
    assert "zero_forcing" in msg and "verdict[isph]" in msg


def test_make_doc_guard_accepts_a_complete_payload():
    """Both predeclared routes present, every ladder cell scored -> renders."""
    case = synthetic_case()
    re_v = case["reynolds_c"]
    coarse = dict(copy.deepcopy(case), resolution_H_over_dx=16)
    vec = {"H_over_dx": 16, "centre_error_pct": 1.0, "l2_interior_pct": 1.0, "pass": True}
    payload = {
        "settings": {"reynolds_ladder": [re_v]},
        "cases": {"isph_projection": [case, coarse],
                  "wcsp_production": [dict(copy.deepcopy(case), route="wcsp_production",
                                           h_over_dx=1.3),
                                      dict(copy.deepcopy(coarse), route="wcsp_production",
                                           h_over_dx=1.3)]},
        "controls": {"zero_forcing": {"max_abs_u_measured": 0.0},
                     "repeat_headline": {"profiles_identical": True}},
        "summary": {"isph_re0.1": {"convergence": {"n_points": 2, "order": 2.0}},
                    "wcsp_re0.1": {"convergence": {"n_points": 2, "order": 1.0}}},
        "verdict": {"isph": {str(re_v): dict(vec)}, "wcsp": {str(re_v): dict(vec)}},
    }
    assert_coverage(payload)          # must not raise


@pytest.mark.skipif(not RESULTS.exists(), reason="production artefact not generated")
def test_production_artefact_passes_independent_audit(tmp_path):
    A = audit.run(RESULTS, DOC if DOC.exists() else None, tmp_path / "audit.json")
    assert A.blocking() == [], A.blocking()


@pytest.mark.skipif(not (RESULTS.exists() and DOC.exists()),
                    reason="artefact or document missing")
def test_document_binds_to_artefact():
    payload = json.loads(RESULTS.read_text())
    A = audit.Audit()
    audit.audit_doc(payload, DOC.read_text(), {}, A)
    assert A.blocking() == [], A.blocking()