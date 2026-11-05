"""Semantic-union resolution of the two conflicted files (no mechanical overlay).

Each conflict block is resolved by explicit reasoning, never by picking one side:
 * both sides' additions are kept; where the two sides disagree on a *default*
   the newer main value wins, and the PR side's new knobs keep their historical
   default so the union stays behaviour-preserving for existing callers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_blocks(text: str):
    """Return list of (start_line_idx, end_line_idx, ours, theirs)."""
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            start = i
            j = i + 1
            ours = []
            while not lines[j].startswith("======="):
                ours.append(lines[j])
                j += 1
            j += 1
            theirs = []
            while not lines[j].startswith(">>>>>>> "):
                theirs.append(lines[j])
                j += 1
            blocks.append((start, j, ours, theirs))
            i = j + 1
        else:
            i += 1
    return blocks


def strip_docstring_close(lines):
    """Drop a trailing `    \"\"\"` line (the docstring terminator) from a block."""
    out = list(lines)
    while out and out[-1].strip() == "":
        out.pop()
    if out and out[-1].strip() == '"""':
        out.pop()
        while out and out[-1].strip() == "":
            out.pop()
    return out


def fix_csf_consistent(text: str) -> str:
    lines = text.split("\n")
    blocks = parse_blocks(text)
    assert len(blocks) == 2, f"csf_consistent.py: expected 2 conflicts, got {len(blocks)}"
    out = list(lines)
    for start, end, ours, theirs in reversed(blocks):
        if len(ours) == 1 and len(theirs) == 1 and ":" in ours[0] and ":" in theirs[0]:
            # signature: both sides appended one keyword argument -> union, ours first
            union = ours + theirs
        else:
            # docstring paragraph + body guard: keep ours' paragraph, then theirs
            # (its paragraph + close + validation + `n = pos.shape[0]`)
            union = strip_docstring_close(ours) + [""] + theirs
        out[start:end + 1] = union
    return "\n".join(out)


def fix_droplet_impact(text: str) -> str:
    lines = text.split("\n")
    blocks = parse_blocks(text)
    assert len(blocks) == 1, f"droplet_impact.py: expected 1 conflict, got {len(blocks)}"
    start, end, ours, theirs = blocks[0]
    idx = [k for k, l in enumerate(theirs) if re.match(r"\s+csf_\w+:", l)]
    assert len(idx) == 3, idx
    extra = theirs[idx[1]:]                      # csf_delta_s + csf_force_density
    union = list(ours) + extra                   # main's csf_curvature block (default kept)
    out = list(lines)
    out[start:end + 1] = union
    return "\n".join(out)


def main() -> int:
    for name, fn in (("src/tensorsph/csf_consistent.py", fix_csf_consistent),
                     ("src/tensorsph/droplet_impact.py", fix_droplet_impact)):
        p = ROOT / name
        new = fn(p.read_text())
        assert "<<<<<<<" not in new and ">>>>>>>" not in new, name
        assert "\n=======\n" not in new, name
        p.write_text(new)
        print(f"resolved {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())