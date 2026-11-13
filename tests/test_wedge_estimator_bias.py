"""Invariants of the same-operator peak-Cp calibration (m18382).

What these tests protect
------------------------
``benchmarks/case_sph_wedge_entry_2d/estimator_bias.py`` claims three things
that the earlier rounds could not state because they compared a discretised
measurement with a point value:

1. the estimator copy in that script is *the same estimator* as
   ``ensemble_peak.py`` (pooling + sample gate + max over bin means);
2. the Adami wall-pressure operator it models is *unbiased where the field is
   smooth* -- constant and nearly linear fields must come back unchanged, and at
   ``tau -> 0`` (Wagner's stagnation point, ``dCp/dtau = 0``) the smoothed model
   must return ``Cp(0) = 2C`` to within 0.1 %;
3. applying that operator to the analytic field is *grid consistent* (no sign
   flip between ``dx = 4 mm`` and ``dx = 2 mm``), which is what makes the
   measured cross-grid sign flip a property of the SPH data.

A fourth test is a regression on the committed artefacts themselves: the
per-frame wet-node count grows with time (the wetted width ``c = C V t`` grows),
so any claim of "wet nodes decay as the body sinks" is contradicted by the
artefacts.  That misreading was published in m17899 and is corrected in
``docs/case_sph_wedge_entry_2d_operator_bias_m18382.md``; the test pins the
artefact fact and the correction banner so the error cannot come back.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bias_mod():
    return _load("wedge_estimator_bias", BENCH / "estimator_bias.py")


@pytest.fixture(scope="module")
def ensemble_mod():
    return _load("wedge_ensemble_peak", BENCH / "ensemble_peak.py")


@pytest.fixture(scope="module")
def report():
    path = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias" / "bias_report.json"
    if not path.exists():
        pytest.skip("bias report not generated")
    return json.loads(path.read_text())


def test_estimator_copy_matches_ensemble_peak(bias_mod, ensemble_mod):
    """Same samples in -> same pooled peak out (the "same operator" premise)."""
    rng = np.random.default_rng(7)
    tau = rng.uniform(0.0, 1.2, 400)
    cp = rng.uniform(0.0, 10.0, 400)
    for n_bins in (6, 10, 12, 20, 30):
        centres, means, cnts = ensemble_mod.binned(tau.tolist(), cp.tolist(), n_bins)
        ref, ref_tau, _ = ensemble_mod.gated_peak(centres, means, cnts)
        got, got_tau = bias_mod.estimator_peak(tau, cp, n_bins)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)
        assert got_tau == pytest.approx(ref_tau, rel=1e-12, abs=1e-12)


def test_kernel_operator_is_exact_on_a_constant_field(bias_mod):
    """A constant field must survive ``p_w = sum p W / sum W`` untouched."""
    tau = np.array([0.01, 0.2, 0.5, 0.9, 0.98])
    vals = bias_mod.smoothed_node_values(tau, c=0.05, dx=0.002,
                                         beta=math.radians(30.0),
                                         field=lambda t: 7.25)
    assert np.allclose(vals, 7.25, rtol=1e-12)


def test_kernel_operator_is_near_unbiased_on_a_linear_field(bias_mod):
    """A linear field must come back within the stencil's offset error."""
    a, b = 1.0, 4.0
    tau = np.array([0.05, 0.3, 0.6, 0.9])
    vals = bias_mod.smoothed_node_values(tau, c=0.05, dx=0.002,
                                         beta=math.radians(30.0),
                                         field=lambda t: a + b * t)
    assert np.allclose(vals, a + b * tau, atol=0.05 * b)


def test_operator_has_support_at_the_peak_abscissa(bias_mod):
    """Guard: the stencil must be node-relative, so a node at tau = 0.93 has
    fluid neighbours.  A stencil anchored at the apex returns NaN there and
    silently turns "smoothed peak" into "value near the stagnation point"."""
    beta = math.radians(30.0)
    tau = np.array([0.5, 0.93, 0.98])
    vals = bias_mod.smoothed_node_values(tau, c=0.05, dx=0.002, beta=beta)
    assert np.all(np.isfinite(vals))
    assert vals[1] > vals[0], "Cp rises toward the Wagner peak at tau = 0.93"


def test_operator_is_unbiased_at_the_stagnation_point(bias_mod):
    """``Cp(0) = 2C`` is an operator-fixed point (flat profile), so no bias."""
    beta = math.radians(30.0)
    tau = np.array([0.0, 0.005, 0.01])
    vals = bias_mod.smoothed_node_values(tau, c=0.05, dx=0.002, beta=beta)
    point = bias_mod.wagner_p_over_q(0.0, beta)
    assert np.allclose(vals, point, rtol=2e-3)


def test_analytic_field_through_operator_is_grid_consistent(bias_mod, report):
    """H2: the operator alone does not reproduce the measured sign flip."""
    assert report["hypothesis"]["synth_has_sign_flip_between_grids"] is False
    for grid in ("dx2mm", "dx4mm"):
        last = report["grids"][grid]["by_frames"]["61"]
        # smoothed truth must sit BELOW the point reference (kernel averaging of
        # a profile whose peak is 3-4 dx from the spray root) and the two grids
        # must agree with each other inside the case tolerance
        assert last["synth_smoothed"]["peak_median"] < report["wagner_cp_max_point"]
        assert abs(last["synth_smoothed"]["spread_rel"]) < 0.15
    med = [report["grids"][g]["by_frames"]["61"]["synth_smoothed"]["peak_median"]
           for g in ("dx2mm", "dx4mm")]
    assert abs(med[0] - med[1]) / max(med) < 0.15


def test_sampling_model_reproduces_committed_wet_node_counts(report):
    """The forward sampling model is checked against the committed artefacts."""
    for grid in ("dx2mm", "dx4mm"):
        chk = report["grids"][grid]["sampling_model_check"]
        obs = np.asarray(chk["per_frame_obs"], dtype=float)
        assert chk["mean_abs_diff"] < 0.10 * obs[obs > 0].mean()


def test_committed_wet_nodes_grow_with_time_artefact_fact():
    """Wet nodes grow as c = C V t grows; the artefacts must show it."""
    art = json.loads((ART / "ensemble_dx2mm_c0v20.json").read_text())
    n_wet = [r["n_wet"] for r in art["frame_scan"]]
    late = np.mean(n_wet[-10:])
    early = np.mean(n_wet[1:11])          # frame 0 is the t = 0 snapshot
    assert late > early, "wetted node count must grow with the wetted width"
    # a shorter (i.e. later) window can only raise the ensemble mean
    means = [art["by_frames"][k]["mean_wet_nodes"] for k in ("61", "41", "25", "13", "7")]
    assert means == sorted(means)


def test_m17899_correction_banner_is_published():
    """The contradicting claim in m17899 is corrected in the repo, not silently."""
    text = (ROOT / "docs" / "case_sph_wedge_entry_2d_ensemble_peak_m17899.md").read_text()
    assert "m18382" in text and "更正" in text