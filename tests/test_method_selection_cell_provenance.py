"""Guard for the decision-cell provenance manifest (mission 20968).

The manifest is the only place that binds a graded matrix record to the compute
job that produced it.  These tests pin the properties that make it usable as
evidence:

1. digests are recomputed from bytes (a changed file changes the manifest),
2. a missing record is reported as missing, never dropped or faked,
3. a claim that a record came back from a channel while the file is absent is
   refused (this is exactly the m20569 failure mode),
4. job specs are validated against the real cell catalogue, so a typo cannot
   silently shrink the manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO / "benchmarks" / "xval_method_selection"))

import record_provenance as prov  # noqa: E402
import run_cell  # noqa: E402

CELLS = {0: ("D20_h025_A", []), 1: ("D13_h025_A", [])}


def test_digest_matches_bytes_and_reports_missing(tmp_path: Path):
    blob = b'{"config": {"depth_cells": 20}}'
    path = tmp_path / "cell_0.json"
    path.write_bytes(blob)

    info = prov.digest(path)
    assert info["present"] is True
    assert info["bytes"] == len(blob)
    assert info["sha256"] == hashlib.sha256(blob).hexdigest()

    absent = prov.digest(tmp_path / "nope.json")
    assert absent == {"present": False, "sha256": None, "bytes": None}


def test_build_marks_missing_records_instead_of_dropping_them(tmp_path: Path,
                                                             monkeypatch):
    (tmp_path / "cell_0.json").write_text('{"a": 1}')
    manifest = prov.build(tmp_path, {0: "cj1", 1: "cj2"}, cells=CELLS, mission=20968)

    assert manifest["cells_total"] == 2
    assert manifest["cells_present"] == 1
    rows = {row["index"]: row for row in manifest["records"]}
    assert rows[0]["present"] is True and rows[0]["recovery"] == "fetch_tar"
    assert rows[1]["present"] is False and rows[1]["recovery"] == "missing"
    assert rows[1]["label"] == "D13_h025_A"
    assert rows[0]["job"] == "cj1" and rows[1]["job"] == "cj2"


def test_changed_record_changes_digest(tmp_path: Path):
    path = tmp_path / "cell_0.json"
    path.write_text('{"a": 1}')
    first = prov.build(tmp_path, {0: "cj1"}, cells=CELLS)["records"][0]["sha256"]
    path.write_text('{"a": 2}')
    second = prov.build(tmp_path, {0: "cj1"}, cells=CELLS)["records"][0]["sha256"]
    assert first != second


def test_recovery_channel_cannot_contradict_the_working_tree(tmp_path: Path):
    # m20569 failure mode: "the job succeeded" while no artefact exists.
    with pytest.raises(SystemExit):
        prov.build(tmp_path, {0: "cj1"}, recovery={0: "log_base64"}, cells=CELLS)


def test_unknown_recovery_mode_is_refused(tmp_path: Path):
    (tmp_path / "cell_0.json").write_text("{}")
    with pytest.raises(SystemExit):
        prov.build(tmp_path, {0: "cj1"}, recovery={0: "smoke_signal"}, cells=CELLS)


@pytest.mark.parametrize("specs", [["9=cj1"], ["zero=cj1"], ["0"], ["0=job-1"]])
def test_invalid_job_specs_are_refused(specs):
    with pytest.raises(SystemExit):
        prov.parse_job_specs(specs, run_cell.CELLS)


def test_real_cell_catalogue_is_accepted():
    jobs = prov.parse_job_specs([f"{i}=cj{100 + i}" for i in run_cell.CELLS],
                                run_cell.CELLS)
    assert sorted(jobs) == sorted(run_cell.CELLS)
    assert jobs[0] == "cj100"


def test_cli_writes_json_manifest(tmp_path: Path, capsys):
    (tmp_path / "cell_1.json").write_text('{"config": {}}')
    out = tmp_path / "manifest.json"
    rc = prov.main(["--record-dir", str(tmp_path), "--out", str(out),
                    "--job", "1=cj3095", "--mission", "20968"])
    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["anchor"] == prov.ANCHOR
    assert manifest["cells_total"] == len(run_cell.CELLS)
    present = [r for r in manifest["records"] if r["present"]]
    assert [r["label"] for r in present] == ["D13_h025_A"]
    assert "records present" in capsys.readouterr().out