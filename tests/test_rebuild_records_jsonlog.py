"""Job logs from the compute gate arrive JSON-escaped: the record channel must survive it.

Mission 20968 recovered cell 0 of the method-selection matrix from
``/root/tp-agent/cluster/jobs/cj3094.log``.  The file is a JSON object whose
payload has its newlines escaped as ``\\n``; the parser saw one line, found no
``>>>BEGIN_RECORD_B64`` marker and reported "nothing written" - indistinguishable
from the m20569/m20783 failure mode where the artefact really was lost.  These
tests pin the unescaping so that "the log is JSON" can never again be mistaken
for "the job produced no record".
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO / "benchmarks" / "xval_method_selection"))

import rebuild_records as rr  # noqa: E402


def _dump(label: str, blob: bytes) -> str:
    import hashlib

    digest = hashlib.sha256(blob).hexdigest()
    return "\n".join([
        f">>>BEGIN_RECORD_B64 {label} {digest} {len(blob)}",
        base64.b64encode(blob).decode("ascii"),
        f">>>END_RECORD_B64 {label}",
    ])


BLOB = b'{"case": "regular", "failed": true, "error": "non-finite acceleration"}'


def test_plain_log_recovers_record():
    blobs, problems = rr.parse_log(_dump("cell_0", BLOB))
    assert problems == []
    assert blobs["cell_0"] == BLOB


def test_json_escaped_log_recovers_record():
    # Shape of /root/tp-agent/cluster/jobs/cj3094.log: a JSON object holding the
    # job's stdout with escaped newlines.
    log = json.dumps({"job_id": "cj3094", "node": "n330", "rc": 0,
                      "output_tail": _dump("cell_0", BLOB)})
    assert "\\n" in log and "\n" not in log.strip()
    blobs, problems = rr.parse_log(log)
    assert problems == []
    assert blobs["cell_0"] == BLOB


def test_escaped_log_with_wrong_digest_is_refused():
    dump = _dump("cell_0", BLOB)
    head, payload, tail = dump.split("\n")
    # Flip one base64 character: the printed sha256 no longer describes the bytes.
    flipped = ("B" if payload[0] != "B" else "C") + payload[1:]
    log = json.dumps({"output": "\n".join([head, flipped, tail])})
    blobs, problems = rr.parse_log(log)
    assert "cell_0" not in blobs
    assert problems, "a damaged dump must be reported, never silently accepted"