"""Rebuild matrix records from a job log that carries the base64 dump.

``run_cell_dump.py`` prints

    >>>BEGIN_RECORD_B64 <label> <sha256> <bytes>
    <base64 blob>
    >>>END_RECORD_B64 <label>

into the job log.  This reader turns such a log back into files, verifying the
printed sha256 so a truncated or interleaved log is rejected instead of being
silently accepted as a record.

Usage::

    python benchmarks/xval_method_selection/rebuild_records.py log.txt \
        --out-dir runs/xval
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

BEGIN = re.compile(r"^>>>BEGIN_RECORD_B64 (\S+) ([0-9a-f]{64}) (\d+)\s*$")
END = ">>>END_RECORD_B64 "


def normalize_log(text: str) -> str:
    """Undo one level of JSON escaping in a job log.

    The compute gate stores a finished job as a JSON object whose ``output_tail``
    (or the whole body) has newlines escaped as ``\\n``.  Feeding that file
    straight to :func:`parse_log` would see a single line and find no dump, which
    looks like "the job wrote nothing" - the exact confusion this module exists to
    prevent.  So unescape newlines when the text clearly is JSON.
    """
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return text.replace("\\n", "\n")
    parts = []
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, str):
                parts.append(value)
        parts.append(stripped)
    return "\n".join(parts).replace("\\n", "\n")


def parse_log(text: str) -> tuple[dict[str, bytes], list[dict]]:
    """Return ``({label: blob}, [problems])`` for every dump found in ``text``.

    A dump whose digest or length disagrees with its header is reported as a
    problem and *not* returned, so callers cannot accidentally use a damaged
    record.
    """
    blobs: dict[str, bytes] = {}
    problems: list[dict] = []
    lines = normalize_log(text).splitlines()
    i = 0
    while i < len(lines):
        head = BEGIN.match(lines[i].strip())
        if not head:
            i += 1
            continue
        label, digest, nbytes = head.group(1), head.group(2), int(head.group(3))
        payload: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(END):
            payload.append(lines[i].strip())
            i += 1
        if i >= len(lines):
            problems.append({"label": label, "error": "unterminated dump"})
            break
        try:
            blob = base64.b64decode("".join(payload), validate=True)
        except (binascii.Error, ValueError) as exc:
            problems.append({"label": label, "error": f"bad base64: {exc}"})
            i += 1
            continue
        if len(blob) != nbytes:
            problems.append({"label": label,
                             "error": f"length {len(blob)} != header {nbytes}"})
        elif hashlib.sha256(blob).hexdigest() != digest:
            problems.append({"label": label, "error": "sha256 mismatch"})
        else:
            blobs[label] = blob
        i += 1
    return blobs, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="+", help="job log file(s) to read")
    parser.add_argument("--out-dir", default="runs/xval")
    parser.add_argument("--names", default="{}",
                        help="JSON map label -> record filename "
                             "(default: cell_<n>.json by matrix order)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = json.loads(args.names)

    written: list[dict] = []
    problems: list[dict] = []
    for log in args.log:
        blobs, bad = parse_log(Path(log).read_text(errors="replace"))
        problems.extend(bad)
        for label, blob in blobs.items():
            name = names.get(label, f"{label}.json")
            path = out_dir / name
            path.write_bytes(blob)
            written.append({"label": label, "path": str(path),
                            "bytes": len(blob),
                            "sha256": hashlib.sha256(blob).hexdigest()})

    print(json.dumps({"written": written, "problems": problems}, indent=1))
    return 0 if written and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())