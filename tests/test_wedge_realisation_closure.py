"""Guards for the committed realization-closure evidence (CASE-SPH-WEDGE-ENTRY-2D).

The artefacts under ``docs/artifacts/case_sph_wedge_entry_2d_realisations*/`` are
the only numeric support of
``docs/case_sph_wedge_entry_2d_realisation_closure_m22929.md``.  These tests do
not re-run SPH; they assert that the committed numbers *are* the numbers the
document claims, i.e. that the evidence layer cannot silently drift:

* the ``scale = 0`` realization is the determinism negative control: it must
  reproduce the committed single-trajectory reading **bit for bit** (the
  discrete-phase switch is default-off);
* the ensemble summary must follow the predeclared rule exactly
  (``U_real = max(U_ci, U_band)``, ``judgeable <=> U_real <= gate``), and its
  verdict must equal an independent re-evaluation of ``ladder_closure.verdict``;
* the committed markdown must quote the committed verdict (no hand-edited doc).

When the artefacts are absent (e.g. a source-only checkout) the tests skip
rather than pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
ART = ROOT / "docs" / "artifacts"
ENS_DIR = ART / "case_sph_wedge_entry_2d_realisations"
ENS = ENS_DIR / "seed_ensemble.json"
DOC = ROOT / "docs" / "case_sph_wedge_entry_2d_realisation_closure_m22929.md"
GATE = 0.15

sys.path.insert(0, str(BENCH))
import ladder_closure as lc  # noqa: E402


def _ensemble() -> dict:
    if not ENS.exists():
        pytest.skip(f"realization ensemble artefact not committed: {ENS}")
    return json.loads(ENS.read_text())


def _committed_single_trajectory(dx: float) -> dict:
    tag = f"dx{dx * 1e3:g}mm"
    path = ART / "case_sph_wedge_entry_2d_ensemble" / f"ensemble_{tag}_c0v20.json"
    if not path.exists():
        pytest.skip(f"ladder artefact missing: {path}")
    return json.loads(path.read_text())


def test_scale_zero_is_bitwise_negative_control():
    """``scale = 0`` must reproduce the committed reading exactly (default-off)."""
    ens = _ensemble()
    checked = 0
    for grid in ens["grids"]:
        zero = [r for r in grid["realisations"] if r["scale"] == 0.0]
        if not zero:
            continue
        ref = _committed_single_trajectory(grid["dx"])["by_frames"]["61"][
            "resolution_band"]["peak_median"]
        for r in zero:
            assert r["obs_peak"] == ref, (
                f"{grid['tag']} scale=0 realization {r['obs_peak']!r} != "
                f"committed {ref!r}: the discrete-phase switch is no longer "
                f"default-off, so the historical artefacts are not reproducible")
            checked += 1
    assert checked, "no scale=0 realization committed: negative control missing"


def test_uncertainty_definition_and_gate():
    """``U_real = max(U_ci, U_band)`` and ``judgeable <=> U_real <= gate``."""
    ens = _ensemble()
    assert ens["grids"], "empty ensemble"
    for g in ens["grids"]:
        u = g["uncertainty"]
        parts = [v for v in (g["U_realisation_ci"], g["U_band"]) if v == v]
        if parts:
            assert u == pytest.approx(max(parts), rel=0, abs=1e-12), g["tag"]
        if g["judgeable"]:
            assert u == u and u <= GATE, f"{g['tag']}: judgeable with U={u}"
        else:
            assert not (u == u and u <= GATE), (
                f"{g['tag']}: U={u} inside the gate but marked not judgeable")


def test_residual_sign_matches_residual_median():
    """The verdict's rows must be the ensemble medians, not a cherry-picked member."""
    ens = _ensemble()
    by_tag = {g["tag"]: g for g in ens["grids"]}
    for row in ens["rule_inputs"]:
        g = by_tag[row["tag"]]
        assert row["residual"] == pytest.approx(g["residual_median"], rel=0, abs=1e-12)
        assert row["obs_peak"] == pytest.approx(g["obs_median"], rel=0, abs=1e-12)
        assert row["uncertainty"] == pytest.approx(g["uncertainty"], rel=0, abs=1e-12)


def test_verdict_is_the_predeclared_rule():
    """Re-evaluating ``ladder_closure.verdict`` on the committed rows is idempotent."""
    ens = _ensemble()
    rows = sorted(ens["rule_inputs"], key=lambda r: -r["dx"])
    again = lc.verdict(rows, gate=GATE)
    assert again["verdict"] == ens["verdict"]["verdict"], (
        f"committed verdict {ens['verdict']['verdict']} != recomputed "
        f"{again['verdict']}")


def test_document_quotes_committed_verdict():
    """The markdown is generated; it must not claim a verdict the data lacks."""
    ens = _ensemble()
    if not DOC.exists():
        pytest.skip(f"generated document missing: {DOC}")
    text = DOC.read_text()
    assert ens["verdict"]["verdict"] in text, (
        "document does not quote the committed verdict")
    assert "人工不写数" in text, "document lost its generated-from-artefacts marker"
    # every graded grid must appear in the document, so a grid cannot be dropped
    for g in ens["grids"]:
        assert f"`{g['tag']}`" in text, f"{g['tag']} missing from the document"