"""Offline gates for ``complete_solenoid_ladder.py`` (no marching, no solve).

Why these exist: the shipped one-shot record
``benchmarks/laplace_sph/results_dynamic_ladder/diagnose_solenoid.json`` holds
only ``R/dx`` = 8 and 12, yet its page quotes a *fitted decay order* of
``-1.311``.  A two-point slope is not a fit, so the negative conclusion ("the
solenoidal part of the CSF traction does not refinish away") is under-evidenced
until levels 16 and 24 exist.  The completion tool must therefore

* run only the **missing** levels and never re-measure a level it already has,
* keep one record **per level** so a killed run leaves earlier levels usable,
* merge through ``diagnose_csf_solenoid.summarise`` (the authoritative gate
  logic) instead of any number computed in the tool,
* leave the merge as a no-op-with-recompute when no new level is requested.

Everything here runs on synthetic records, so it is fast and independent of the
particle solve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "laplace_sph"
sys.path.insert(0, str(BENCH))

import complete_solenoid_ladder as tool  # noqa: E402
import diagnose_csf_solenoid as D  # noqa: E402


def _fake_level(level: int, frac: float, abs_sol: float) -> dict:
    return {
        "R_over_dx": float(level), "dx_m": 0.02 / level, "h_m": 1.3 * 0.02 / level,
        "n_particles": int(780 * (level / 8.0) ** 2), "n_pairs": 1,
        "n_liquid": 1, "a_norm": abs_sol / frac, "solenoidal_fraction": frac,
        "solenoidal_abs": abs_sol, "irrotational_abs": abs_sol / frac - abs_sol,
        "irrotationality": frac, "adjointness_defect": 1e-16,
        "reconstruction_defect": 1e-18, "orthogonality": 1e-13,
        "cg_iterations": 100, "v_ref_m_per_s": 0.06, "dt_s": 1e-3,
        "t_ref_s": 1.0 / 3.0, "umax_over_vref_predicted": 1.0, "wall_s": 1.0,
    }


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Redirect the tool's two record locations into a temp tree."""
    leveldir = tmp_path / "solenoid_levels"
    leveldir.mkdir()
    out = tmp_path / "diagnose_solenoid.json"
    out.write_text(json.dumps({"levels": [_fake_level(8, 0.2, 1.0),
                                          _fake_level(12, 0.3, 3.0)]}))
    monkeypatch.setattr(tool, "LEVELDIR", leveldir)
    monkeypatch.setattr(D, "JSON_OUT", out)
    monkeypatch.setattr(D, "OUTDIR", tmp_path)
    return leveldir, out


def test_runs_only_the_missing_level_and_keeps_per_level_records(
        sandbox, monkeypatch, capsys):
    leveldir, out = sandbox
    (leveldir / "Rdx16.json").write_text(json.dumps(_fake_level(16, 0.4, 5.0)))
    called = []

    def _boom(level, *a, **k):          # a measured level must never be re-run
        called.append(level)
        raise AssertionError("re-measured level %r" % level)

    monkeypatch.setattr(D, "measure_level", _boom)
    monkeypatch.setattr(sys, "argv", ["tool", "--levels", "16", "--no-doc"])

    tool.main()

    assert called == [], "an already recorded level was measured again"
    s = json.loads(out.read_text())
    assert s["setup"]["levels_R_over_dx"] == [8, 12, 16]
    got = {r["R_over_dx"]: r for r in s["levels"]}
    assert got[16.0]["solenoidal_fraction"] == 0.4
    assert s["levels_not_measured"] == [24]


def test_new_level_is_persisted_per_level_then_merged(sandbox, monkeypatch):
    leveldir, out = sandbox

    def _fake_measure(level, *a, **k):
        rec = _fake_level(level, 0.5, 9.0)
        print("measured", level)
        return rec

    monkeypatch.setattr(D, "measure_level", _fake_measure)
    monkeypatch.setattr(sys, "argv", ["tool", "--levels", "24", "--no-doc"])

    tool.main()

    assert (leveldir / "Rdx24.json").exists(), "per-level record not written"
    s = json.loads(out.read_text())
    assert s["setup"]["levels_R_over_dx"] == [8, 12, 24]
    assert s["levels_not_measured"] == [16]
    # the fit is *derived* from the recorded levels, never hand-typed here
    assert s["orders"]["solenoidal_fraction_decay_order"] == pytest.approx(
        D.summarise(list(s["levels"]))["orders"]["solenoidal_fraction_decay_order"])


def test_empty_level_list_only_merges(sandbox, monkeypatch):
    leveldir, out = sandbox
    monkeypatch.setattr(D, "measure_level",
                        lambda *a, **k: pytest.fail("nothing should be measured"))
    monkeypatch.setattr(sys, "argv", ["tool", "--levels", "", "--no-doc"])

    tool.main()

    s = json.loads(out.read_text())
    assert s["setup"]["levels_R_over_dx"] == [8, 12]
    assert s["levels_not_measured"] == [16, 24]
    assert s["orders"]["H1_artifact_hypothesis_supported"] is False