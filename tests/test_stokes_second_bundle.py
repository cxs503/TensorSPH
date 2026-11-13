"""The published Stokes-second bundle must be checkable *as shipped*.

Audits of this case were done by hand: re-hashing artefacts one by one and
recomputing the graded metrics from the raw rows in a scratch script
(``docs/m16353_audit_verdict.md`` §②).  That is exactly the kind of check that
should be executable, so it cannot silently rot.  ``verify_stokes_second_bundle``
implements it fail-closed; these tests pin both directions:

* the bundle in the repository passes every check, including the two figures
  that ``results_ladder.json`` declares (they were *missing* while the manifest
  certified the bundle — a dangling reference no test noticed), and
* each defect class is actually caught: tampered artefact, unlisted file,
  missing figure, a metric the rows do not support, a stale manifest header.

The last two tests cover the producer side: ``_write_bundle`` must keep a
per-results-file record so a later ``--quick`` run cannot make the manifest
describe the ladder with the wrong mode/command, and regenerating the figures
must not touch a single graded number.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.stokes_second.run import _write_bundle
from scripts.verify_stokes_second_bundle import (
    DEFAULT_OUT,
    BundleError,
    sha256,
    verify_bundle,
)

REPO = Path(__file__).resolve().parents[1]


def _copy_bundle(tmp_path: Path) -> Path:
    dst = tmp_path / "out"
    shutil.copytree(DEFAULT_OUT, dst)
    return dst


def _rehash(manifest_path: Path, name: str, out_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    path = out_dir / name
    manifest["artifacts"][name] = {"sha256": sha256(path),
                                   "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=1))


# --------------------------------------------------------------------------- #
# The shipped bundle
# --------------------------------------------------------------------------- #

def test_published_bundle_passes_every_check():
    report = verify_bundle(DEFAULT_OUT)
    assert report["problems"] == []
    assert report["ok"] is True
    assert report["resolutions"] == [8, 12, 18]
    # the primary number the case is graded on is the finest rung's L2
    assert report["primary_value_pct"] < 5.0


def test_published_figures_exist_and_are_hashed():
    bundle = DEFAULT_OUT
    ladder = json.loads((bundle / "results_ladder.json").read_text())
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    assert ladder["figures"], "the ladder must declare its figures"
    for entry in ladder["figures"]:
        path = REPO / entry
        assert path.is_file(), f"declared figure missing: {entry}"
        assert manifest["artifacts"][path.name]["sha256"] == sha256(path)
    prov = json.loads((bundle / "FIGURES_PROVENANCE.json").read_text())
    assert prov["source"]["sha256"] == sha256(bundle / "results_ladder.json")


# --------------------------------------------------------------------------- #
# Defect classes the verifier has to catch
# --------------------------------------------------------------------------- #

def test_tampered_artefact_is_detected(tmp_path):
    out = _copy_bundle(tmp_path)
    path = out / "benchmark.json"
    data = json.loads(path.read_text())
    data["primary_value_pct"] = 0.5     # 50x better than measured
    path.write_text(json.dumps(data, indent=1))
    problems = verify_bundle(out)["problems"]
    assert any(p.startswith("V1 benchmark.json") for p in problems), problems


def test_unlisted_artefact_is_detected(tmp_path):
    out = _copy_bundle(tmp_path)
    (out / "notes.txt").write_text("scratch\n")
    assert any(p.startswith("V2 notes.txt") for p in verify_bundle(out)["problems"])


def test_missing_declared_figure_is_detected(tmp_path):
    """This is the defect the published bundle actually had: the ladder
    declared two PNGs that were not in the repository."""
    out = _copy_bundle(tmp_path)
    (out / "amplitude_decay.png").unlink()
    # still hashed -> V1 fires; the dangling *declaration* is V3
    assert any(p.startswith("V1 amplitude_decay.png")
               for p in verify_bundle(out)["problems"])

    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["artifacts"]["amplitude_decay.png"]
    manifest_path.write_text(json.dumps(manifest, indent=1))
    problems = verify_bundle(out)["problems"]
    assert any(p.startswith("V3") and "amplitude_decay.png" in p
               for p in problems), problems


def test_metric_the_rows_do_not_support_is_detected(tmp_path):
    out = _copy_bundle(tmp_path)
    path = out / "results_ladder.json"
    ladder = json.loads(path.read_text())
    run = max(ladder["runs"], key=lambda r: r["resolution_delta_over_dx"])
    run["metrics"]["amplitude_l2_half_space_pct"] *= 1.5
    path.write_text(json.dumps(ladder, indent=1))
    _rehash(out / "MANIFEST.json", "results_ladder.json", out)
    problems = verify_bundle(out)["problems"]
    assert any(p.startswith("V5") and "amplitude_l2_half_space_pct" in p
               for p in problems), problems


def test_stale_manifest_header_is_detected(tmp_path):
    out = _copy_bundle(tmp_path)
    path = out / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["mode"] = "quick"
    manifest["runs"]["results_ladder.json"]["command"] = "python run.py --quick"
    path.write_text(json.dumps(manifest, indent=1))
    problems = verify_bundle(out)["problems"]
    assert any(p.startswith("V8") for p in problems), problems


def test_incomplete_ladder_is_detected(tmp_path):
    out = _copy_bundle(tmp_path)
    path = out / "results_ladder.json"
    ladder = json.loads(path.read_text())
    ladder["runs"] = [max(ladder["runs"],
                          key=lambda r: r["resolution_delta_over_dx"])]
    ladder["convergence"]["resolutions"] = [18]
    path.write_text(json.dumps(ladder, indent=1))
    _rehash(out / "MANIFEST.json", "results_ladder.json", out)
    problems = verify_bundle(out)["problems"]
    assert any(p.startswith("V4") for p in problems), problems


# --------------------------------------------------------------------------- #
# Producer side
# --------------------------------------------------------------------------- #

def _payload(mode: str, finished: str) -> dict:
    return {
        "mode": mode,
        "finished": finished,
        "provenance": {"gates": {"G1_amplitude_l2_pct": 5.0}, "case": {}},
        "runs": [],
    }


def test_manifest_keeps_a_record_per_results_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # a ladder manifest is on disk; a later quick run must not erase its record
    (out / "results_ladder.json").write_text(json.dumps({"mode": "ladder",
                                                         "repro_cmd": "--ladder"}))
    (out / "MANIFEST.json").write_text(json.dumps({
        "mode": "ladder",
        "runs": {"results_ladder.json": {"mode": "ladder",
                                         "command": "--ladder",
                                         "finished": "2026-11-12T00:00:00Z"}}}))

    _write_bundle(out, _payload("quick", "2026-11-13T00:00:00Z"), "--quick")

    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["runs"]["results_ladder.json"]["command"] == "--ladder"
    assert manifest["runs"]["results_quick.json"]["mode"] == "quick"
    assert manifest["runs"]["results_quick.json"]["command"] == "--quick"
    assert "results_ladder.json" in manifest["artifacts"]


def test_regenerating_figures_does_not_move_a_graded_number(tmp_path, monkeypatch):
    out = _copy_bundle(tmp_path)
    before = sha256(out / "results_ladder.json")
    bench_before = (out / "benchmark.json").read_text()

    from scripts import regen_stokes_second_figures as regen

    monkeypatch.setattr(regen, "REPO", tmp_path)
    assert regen.main(["--out", str(out)]) == 0

    assert sha256(out / "results_ladder.json") == before
    assert (out / "benchmark.json").read_text() == bench_before
    assert verify_bundle(out)["ok"] is True


def test_regeneration_refuses_a_bundle_that_declares_other_figures(tmp_path, monkeypatch):
    out = _copy_bundle(tmp_path)
    path = out / "results_ladder.json"
    ladder = json.loads(path.read_text())
    ladder["figures"] = ["benchmarks/stokes_second/out/other.png"]
    path.write_text(json.dumps(ladder, indent=1))

    from scripts import regen_stokes_second_figures as regen

    monkeypatch.setattr(regen, "REPO", tmp_path)
    assert regen.main(["--out", str(out)]) == 2


def test_verifier_refuses_a_missing_directory(tmp_path):
    with pytest.raises(BundleError):
        verify_bundle(tmp_path / "nope")
