"""Guards on the Young-Laplace resolution ladder.

The ladder (``benchmarks/laplace_law/resolution_ladder.json`` plus the per-rung
artifacts in ``convergence/``) is produced by

    PYTHONPATH=src python benchmarks/laplace_law/run_ladder.py

and rendered into ``docs/verification_laplace_resolution_ladder.md`` by
``make_ladder_doc.py``.  What is guarded here:

1. **the protocol** -- every rung must be read at the same *capillary time*
   (``t_over_t_sigma``), not at a fixed step count.  The artifacts this ladder
   replaced used ``steps = 380`` for all three resolutions, which put the
   windows at 0.33 / 0.44 / 0.56 t_sigma: the ladder compared three different
   phases of a capillary oscillation and the shortest one even violated
   ``tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent``
   (``t > 0.5 t_sigma``), which was red on HEAD before this fix;
2. **both read-outs** are present and self-consistent (``bandA`` and the
   absolute ``p(r<0.5R) - 0`` jump), so no verdict can quote a pressure
   difference without stating its reference;
3. **the verdicts are honest** -- a rung may only claim ``within_15pct`` if its
   own numbers say so, and the aggregate summary must agree with the artifacts;
4. **the document is a rendering**, not hand-copied prose: re-rendering it from
   the artifacts reproduces the committed file byte for byte.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_law"
LADDER = BENCH / "resolution_ladder.json"
CONVERGENCE = BENCH / "convergence"
DOC = ROOT / "docs" / "verification_laplace_resolution_ladder.md"
FIGURE = BENCH / "laplace_resolution_ladder.png"

EXPECTED_DXS = [0.0025, 0.002, 0.0015]


def load(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def ladder():
    return load(LADDER)


@pytest.fixture(scope="module")
def artifacts(ladder):
    return {r["artifact"].split("/")[-1]:
            load(BENCH / r["artifact"]) for r in ladder["rungs"]}


def test_ladder_covers_the_expected_resolutions(ladder):
    dxs = sorted(r["dx_m"] for r in ladder["rungs"])
    assert dxs == sorted(EXPECTED_DXS)
    for rung in ladder["rungs"]:
        assert (BENCH / rung["artifact"]).exists()


def test_every_rung_is_read_at_the_same_physical_time(ladder):
    """The whole point of the fix: fixed duration, not fixed step count."""
    duration = ladder["protocol"]["duration_t_over_t_sigma"]
    assert duration >= 0.5, "the ladder must not sit under the repository minimum"
    for rung in ladder["rungs"]:
        assert rung["t_over_t_sigma"] == pytest.approx(duration, rel=2e-3)
    # different resolutions cannot share a step count if they share a duration
    steps = {rung["steps"] for rung in ladder["rungs"]}
    assert len(steps) == len(ladder["rungs"]), steps


def test_artifacts_declare_both_readouts(artifacts):
    for name, rep in artifacts.items():
        lap = rep["laplace"]
        assert "measurement" in lap, name
        assert "absolute = p(r<0.5R) - 0" in lap["measurement"], name
        assert "bandA" in lap["measurement"], name
        for key in ("p_center_absolute_Pa", "rel_error_absolute_pct",
                    "rel_error_absolute_tailavg_pct",
                    "compression_ratio_measured_over_required",
                    "rho_core_rel_std",
                    "rel_error_from_mean_core_density_pct"):
            assert key in lap, (name, key)


def test_readout_arithmetic_is_self_consistent(artifacts):
    """Every derived number must recompute from the raw ones."""
    for name, rep in artifacts.items():
        lap = rep["laplace"]
        theory = lap["delta_p_theory_Pa"]
        band_err = ((lap["p_center_Pa"] - lap["p_interface_band_Pa"]) - theory) \
            / theory * 100.0
        assert band_err == pytest.approx(lap["rel_error_pct"], abs=1e-9)
        abs_err = (lap["p_center_absolute_Pa"] - theory) / theory * 100.0
        assert abs_err == pytest.approx(lap["rel_error_absolute_pct"], abs=1e-9)
        assert lap["within_15pct"] is (abs(band_err) < 15.0)
        assert lap["within_15pct_absolute"] is (abs(abs_err) < 15.0)
        assert lap["within_3pct_absolute"] is (abs(abs_err) < 3.0)
        # the historical bandA definition mixes in the surface density deficit,
        # so on this configuration it must be strictly worse than the absolute
        # one -- if that ever flips, the definition note in the document is wrong
        assert abs(band_err) > abs(abs_err), name


def test_compression_ratio_agrees_with_the_core_density(artifacts):
    for name, rep in artifacts.items():
        lap = rep["laplace"]
        measured = lap["density_compression_at_center"]
        required = lap["compressibility_theory"]
        assert lap["compression_ratio_measured_over_required"] == pytest.approx(
            measured / required, rel=1e-9)
        assert required > 0.0


def test_summary_matches_the_artifacts(ladder, artifacts):
    for rung in ladder["rungs"]:
        rep = artifacts[rung["artifact"].split("/")[-1]]
        lap = rep["laplace"]
        assert rung["dx_m"] == rep["setup"]["dx_m"]
        assert rung["n_particles"] == rep["setup"]["n_particles"]
        assert rung["steps"] == rep["setup"]["steps"]
        assert rung["absolute"]["rel_error_pct"] == pytest.approx(
            lap["rel_error_absolute_pct"], abs=1e-12)
        assert rung["absolute"]["rel_error_tailavg_pct"] == pytest.approx(
            lap["rel_error_absolute_tailavg_pct"], abs=1e-12)
        assert rung["bandA"]["rel_error_pct"] == pytest.approx(
            lap["rel_error_pct"], abs=1e-12)
        assert rung["absolute"]["within_15pct"] is lap["within_15pct_absolute"]


def test_verdicts_are_explicitly_fail_closed(ladder):
    """No rung may pass the 15 % or 3 % bar on a single-frame read-out alone.

    The ladder is the current statement of the Young-Laplace debt: if a future
    change makes a rung pass, this test fails so that the document, the
    platform capability audit and the WITHHELD verdict are all revisited
    together instead of drifting out of sync.
    """
    passing = [r for r in ladder["rungs"] if r["absolute"]["within_15pct"]]
    assert passing == [], (
        "a rung now meets 15 %: re-render the document, retract the WITHHELD "
        "verdict and re-baseline the capability audit")
    for r in ladder["rungs"]:
        assert math.isfinite(r["absolute"]["rel_error_pct"])


def test_figures_and_document_are_rendered_from_the_artifacts(ladder, artifacts):
    spec = importlib.util.spec_from_file_location(
        "make_ladder_doc", BENCH / "make_ladder_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.render(ladder, artifacts)
    assert DOC.exists(), "docs/verification_laplace_resolution_ladder.md missing"
    assert DOC.read_text() == rendered, (
        "docs/verification_laplace_resolution_ladder.md is stale; rerun "
        "python benchmarks/laplace_law/make_ladder_doc.py")
    assert FIGURE.exists()
    assert FIGURE.stat().st_size > 20_000


def test_field_snapshot_is_big_enough_for_the_cloud_map(artifacts):
    for name, rep in artifacts.items():
        field = rep["field"]
        n = rep["setup"]["n_particles"]
        assert len(field["x_m"]) == n == len(field["pressure_Pa"]), name
        assert len(field["rho_over_rho0"]) == n, name
        assert min(field["pressure_Pa"]) >= 0.0, (
            "the Tait pressure is clamped at zero; a negative field means the "
            "snapshot was taken before the clamp")
