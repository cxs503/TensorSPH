"""The base64 record channel used to bring matrix cells back from the cluster.

Mission 20569 dispatched cells 0-3, all four returned ``rc=0``, and yet no
record could be retrieved: ``run_compute fetch`` guarantees the job log, not the
node's scratch files.  The fix shipped here is a log-carried, digest-checked
copy of the record (``run_cell_dump.py``) plus its reader
(``rebuild_records.py``).  These tests pin the contract without running SPH.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BENCH = REPO / "benchmarks" / "xval_method_selection"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dump(label: str, blob: bytes) -> str:
    digest = hashlib.sha256(blob).hexdigest()
    body = base64.b64encode(blob).decode("ascii")
    return (f"some solver chatter\n"
            f">>>BEGIN_RECORD_B64 {label} {digest} {len(blob)}\n"
            f"{body}\n"
            f">>>END_RECORD_B64 {label}\n"
            f"trailing line\n")


def test_roundtrip_recovers_bytes_exactly(tmp_path):
    rebuild = _load("rebuild_records")
    blob = b'{"ok": true, "times": [0.0, 0.004], "n": "e-acute"}'
    log = tmp_path / "log.txt"
    log.write_text(_dump("D20_h025_A", blob))

    blobs, problems = rebuild.parse_log(log.read_text())
    assert problems == []
    assert blobs["D20_h025_A"] == blob


def test_multiple_dumps_in_one_log(tmp_path):
    rebuild = _load("rebuild_records")
    a, b = b"cell-a", b"cell-b" * 50
    text = _dump("A", a) + _dump("B", b)
    blobs, problems = rebuild.parse_log(text)
    assert problems == []
    assert blobs == {"A": a, "B": b}


def test_truncated_dump_is_rejected_not_accepted():
    rebuild = _load("rebuild_records")
    blob = b"x" * 100
    text = _dump("C", blob).split(">>>END_RECORD_B64")[0]
    blobs, problems = rebuild.parse_log(text)
    assert blobs == {}
    assert problems and problems[0]["error"] == "unterminated dump"


def test_corrupted_payload_is_rejected():
    rebuild = _load("rebuild_records")
    text = _dump("D", b"payload-bytes")
    # flip one base64 character inside the payload line
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(">>>BEGIN_RECORD_B64"))
    idx = start + 1  # the payload line that follows the header
    lines[idx] = ("A" if lines[idx][0] != "A" else "B") + lines[idx][1:]
    blobs, problems = rebuild.parse_log("\n".join(lines))
    assert blobs == {}
    assert problems, "corrupted payload must be reported"


def test_run_cell_dump_matrix_matches_grade_matrix_rows():
    """The dump writer's labels are the labels ``grade_matrix`` grades."""
    run_cell = _load("run_cell")
    grade = _load("grade_matrix")
    dump_labels = {label for label, _ in run_cell.CELLS.values()}
    matrix_labels = {row[0] for row in grade.MATRIX if row[2] != "archive"}
    assert dump_labels == matrix_labels


def test_rebuild_writes_files_with_documented_names(tmp_path):
    rebuild = _load("rebuild_records")
    log = tmp_path / "log.txt"
    log.write_text(_dump("D13_h025_A", b'{"k": 1}'))
    out_dir = tmp_path / "runs"

    rc = rebuild.main([str(log), "--out-dir", str(out_dir),
                       "--names", '{"D13_h025_A": "cell_1.json"}'])
    assert rc == 0
    assert (out_dir / "cell_1.json").read_bytes() == b'{"k": 1}'