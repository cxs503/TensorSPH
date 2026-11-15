"""Tests for the *uncalibrated* grading of the ported transport-velocity correction.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

What is being protected
-----------------------
Rounds 1-2 of the cross-validation grade TVC with ``alpha = 0.10``, a value
calibrated against the reference it is scored on.  ``benchmarks/xval_sphinx_tvc/
uncalibrated.py`` re-grades the scheme with the two coefficients SPHinXsys
*actually ships* (``0.2`` dual criteria, ``0.2/4 = 0.05`` single acoustic step)
and writes ``docs/xval_sphinx_tvc/UNCALIBRATED_DEFAULTS.md`` from the raw
payloads.  These tests pin the three ways that report could silently go wrong:

1. the "default" coefficients drifting away from the upstream constants
   (a fitted value smuggled back in under a default-sounding name);
2. the grading code becoming a *second*, hand-written implementation that
   disagrees with ``run.py:score`` -- so the report would grade a different
   metric than the one the ladder recorded;
3. the numbers in the markdown being typed by hand instead of derived from the
   payload (regenerating the report must reproduce them).

They are pure post-processing: no SPH run, sub-second.

::

    pytest tests/test_xval_tvc_uncalibrated_defaults.py -q
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "xval_sphinx_tvc"
DATA = REPO / "docs" / "xval_sphinx_tvc" / "data"
EXP = REPO / "benchmarks" / "dam_break" / "martin_moyce_1952.json"

for _p in (REPO / "src", BENCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str):
    """Import a benchmark module by path (``benchmarks/`` is not a package)."""
    path = BENCH / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_xval_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def unc():
    return _load("uncalibrated")


@pytest.fixture(scope="module")
def runner():
    return _load("run")


@pytest.fixture(scope="module")
def experiment():
    return json.loads(EXP.read_text())


# --------------------------------------------------------------------------- #
# 1. the coefficients are the upstream ones, not a fit
# --------------------------------------------------------------------------- #
def test_upstream_defaults_are_the_shipped_constants(unc):
    alphas = unc.upstream_default_alphas()
    from tensorsph.transport_velocity_correction import (
        SPHINXSYS_DEFAULT_COEFFICIENT,
        SPHINXSYS_SINGLE_STEP_FRACTION,
    )

    assert alphas["dual_criteria"] == pytest.approx(SPHINXSYS_DEFAULT_COEFFICIENT)
    assert alphas["dual_criteria"] == pytest.approx(0.2)
    assert alphas["single_acoustic_step"] == pytest.approx(
        SPHINXSYS_DEFAULT_COEFFICIENT * SPHINXSYS_SINGLE_STEP_FRACTION)
    assert alphas["single_acoustic_step"] == pytest.approx(0.05)
    # the calibrated round-1/2 value is *not* one of them: that is the point
    assert 0.10 not in alphas.values()


def test_variant_table_pins_them(runner, unc):
    alphas = unc.upstream_default_alphas()
    assert runner.VARIANTS["tvc_a005_all"]["tvc_alpha"] == pytest.approx(
        alphas["single_acoustic_step"])
    assert runner.VARIANTS["tvc_a020_all"]["tvc_alpha"] == pytest.approx(
        alphas["dual_criteria"])
    # the port's own single-step default is what ``alpha=None`` means
    from tensorsph.transport_velocity_correction import (
        transport_velocity_correction as tvc,
    )
    import torch

    pos = torch.zeros((2, 2), dtype=torch.float64)
    pos[1, 0] = 10.0                      # out of support -> zero integral
    vol = torch.ones(2, dtype=torch.float64)
    res = tvc(pos, vol, 0.1)
    assert res.alpha == pytest.approx(alphas["single_acoustic_step"])


# --------------------------------------------------------------------------- #
# 2. the grading is the recorded metric, not a re-implementation
# --------------------------------------------------------------------------- #
def test_regrade_reproduces_the_ladder_scores(unc, experiment):
    payload = json.loads((DATA / "ladder_dense.json").read_text())
    checked = 0
    for rec in payload["records"]:
        g = unc.grade(rec["tau"], rec["Z_over_a"], experiment, band=0.0)
        assert g["points"] == rec["score"]["points"]
        assert g["mae"] == pytest.approx(rec["score"]["mae"], rel=0, abs=1e-12)
        assert g["mean_rel_error"] == pytest.approx(
            rec["score"]["mean_rel_error"], rel=0, abs=1e-12)
        assert g["gate_rel_pass"] == rec["score"]["gate_rel_pass"]
        assert g["gate_abs_pass"] == rec["score"]["gate_abs_pass"]
        checked += 1
    assert checked >= 4


def test_band_bracket_bounds_the_nominal_error(unc, experiment):
    payload = json.loads((DATA / "ladder_dense.json").read_text())
    for rec in payload["records"]:
        g = unc.grade(rec["tau"], rec["Z_over_a"], experiment, band=0.05)
        lo, hi = g["rel_band"]
        assert lo <= g["mean_rel_error"] <= hi, (rec["variant"], rec["dx"])
        lo_a, hi_a = g["mae_band"]
        assert lo_a <= g["mae"] <= hi_a
        # a nominal pass that the band can break must be labelled marginal
        if g["robust_pass"]:
            assert g["verdict"] == "PASS"
        if (g["gate_rel_pass"] and g["gate_abs_pass"]) and not g["robust_pass"]:
            assert g["verdict"] in {"PASS_marginal", "FAIL_marginal"}


def test_a_shifted_curve_fails_and_a_perfect_one_passes(unc, experiment):
    """Gate logic sanity: the verdict must respond to the curve, not to noise."""
    data = experiment["data"]
    taus = [d["tau"] for d in data]
    perfect = [d["Z_over_a"] for d in data]
    g_perfect = unc.grade(taus, perfect, experiment)
    assert g_perfect["mae"] == pytest.approx(0.0, abs=1e-15)
    assert g_perfect["verdict"] == "PASS" and g_perfect["robust_pass"]

    g_high = unc.grade(taus, [z * 1.5 for z in perfect], experiment)
    assert not g_high["gate_rel_pass"] and g_high["verdict"].startswith("FAIL")

    # a constant offset of one band is exactly the band's own scale
    g_off = unc.grade(taus, [z + 0.05 for z in perfect], experiment)
    assert g_off["mae"] == pytest.approx(0.05, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. the delivered payload and report are uncalibrated and generated
# --------------------------------------------------------------------------- #
PAYLOAD = DATA / "ladder_upstream_defaults.json"
LEVEL5 = DATA / "level_dx0005.json"
REPORT = REPO / "docs" / "xval_sphinx_tvc" / "UNCALIBRATED_DEFAULTS.md"


def test_payload_contains_only_uncalibrated_coefficients(unc):
    assert PAYLOAD.exists(), f"missing deliverable payload {PAYLOAD}"
    payload = json.loads(PAYLOAD.read_text())
    alphas = unc.upstream_default_alphas()
    seen = set()
    for rec in payload["records"]:
        alpha = rec["params"]["tvc_alpha"]
        seen.add(round(alpha, 12))
        assert round(alpha, 12) in {round(a, 12) for a in alphas.values()}
    assert seen == {0.05, 0.2}
    assert payload["gates"]["owner_rel"] == pytest.approx(0.15)


def test_report_numbers_are_regenerated_from_the_payload(unc, experiment, tmp_path):
    """The committed report must be the generated one, not hand-edited numbers."""
    assert REPORT.exists(), f"missing deliverable report {REPORT}"
    inputs = [DATA / "ladder_dense.json", DATA / "grid_dx004.json"]
    if PAYLOAD.exists():
        inputs.append(PAYLOAD)
    if LEVEL5.exists():
        inputs.append(LEVEL5)
    rows = unc.load_rows(inputs)
    committed_json = json.loads((DATA / "uncalibrated_defaults.json").read_text())
    report = unc.build_report(rows, experiment,
                              pending=committed_json.get("pending"))
    doc_path = tmp_path / "regen.md"
    fig_path = tmp_path / "regen.png"
    unc.write_doc(report, doc_path, fig_path)
    regenerated = doc_path.read_text()
    committed = REPORT.read_text()
    assert regenerated.replace(fig_path.name, "-") == committed.replace(
        "figure_uncalibrated_defaults.png", "-")
    for row in report["table"]:
        g = row["grade"]
        assert row["variant"] in committed
        if g.get("mean_rel_error") is not None:
            assert f"{100.0 * g['mean_rel_error']:.1f} %" in committed


def test_cauchy_orders_are_finite_where_defined(unc):
    payload = json.loads((DATA / "ladder_dense.json").read_text())
    levels = {float(r["dx"]): r for r in payload["records"]
              if r["variant"] == "tvc_a010_all"}
    pairs = unc.cauchy_pairs(levels)
    assert len(pairs) >= 2
    for p in pairs:
        assert p["cauchy_l2"] >= 0.0
        assert math.isfinite(p["cauchy_l2"])
    assert any(p["order"] is not None for p in pairs)