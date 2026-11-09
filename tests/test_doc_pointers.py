"""Documentation-pointer gate: every ``docs/*.md`` reference must resolve.

A docstring that points at ``docs/verification_<x>.md`` which does not exist is
an unverifiable claim with a link: the reader cannot tell "not written yet"
from "written and moved".  This repo already had five such pointers (including
``src/tensorsph/wave_flume.py`` -> ``docs/verification_wave_flume.md``, the
entry point of a whole module's evidence), which is exactly the kind of debt a
grep finds in a second and nobody owns.

Two ways to satisfy the gate:

* make the file exist (the preferred fix: the pointer is then evidence), or
* list the reference in :data:`WAIVERS` with a reason -- for files that live
  outside this repository (ecosystem-level documents) or that are templates.

The scan is deliberately textual: it covers ``src/``, ``tests/``,
``benchmarks/`` and ``README.md``, i.e. everything a reader of the code can be
sent to.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "tests", "benchmarks")
SCAN_FILES = ("README.md",)
PATTERN = re.compile(r"docs/[\w./+-]+\.md")

#: References that are allowed to stay dangling, each with a reason.
WAIVERS = {
    "docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md":
        "ecosystem-wide vision document; it lives with the platform, not in this "
        "solver repository (referenced from README.md as context only)",
    "docs/verification_threed.md":
        "generator benchmarks/threed_summary/make_doc.py currently raises "
        "KeyError('momentum_scale_sum_m_abs_v') against the committed "
        "benchmarks/threed_*/report.json, so the page cannot be produced from "
        "the evidence on disk; the 3-D claims are measured (README quotes them) "
        "but the page needs the generator repaired -- registered as debt instead "
        "of shipping a hand-written page with undocumented numbers",
    "docs/verification_contact_angle_m6453.md": "debt E3: the m6453 contact-angle doc was never committed; the instrument was repaired and documented in docs/verification_contact_angle_m6503.md instead (referenced by make_overcredit_doc.py and test_contact_angle_instrument_m6503.py)",
}


def _sources():
    for name in SCAN_DIRS:
        root = REPO / name
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.suffix in {".py", ".md"} and "__pycache__" not in path.parts:
                    yield path
    for name in SCAN_FILES:
        path = REPO / name
        if path.exists():
            yield path


def dangling_references() -> dict[str, list[str]]:
    """Map every missing ``docs/*.md`` target to the files that reference it."""
    missing: dict[str, list[str]] = {}
    for path in _sources():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in PATTERN.finditer(text):
            target = match.group(0).rstrip(".,;:)`")
            if (REPO / target).exists():
                continue
            missing.setdefault(target, []).append(str(path.relative_to(REPO)))
    return {target: sorted(set(sources))
            for target, sources in missing.items()}


def test_every_docs_pointer_resolves_or_is_waived():
    missing = dangling_references()
    unexplained = {target: sources for target, sources in missing.items()
                   if target not in WAIVERS}
    assert not unexplained, (
        "dangling docs pointers (either create the file or add a waiver with a "
        "reason):\n" + "\n".join(f"  {target} <- {', '.join(sources)}"
                                 for target, sources in unexplained.items()))


def test_waivers_are_still_needed():
    """A waiver for a file that now exists is stale bookkeeping, not a waiver."""
    missing = dangling_references()
    stale = [target for target in WAIVERS if target not in missing]
    assert not stale, f"waivers no longer needed: {stale}"


def test_wave_flume_docstring_points_at_existing_evidence():
    """The module whose evidence pointer was missing gets its own pin."""
    module = (REPO / "src" / "tensorsph" / "wave_flume.py").read_text()
    for target in PATTERN.findall(module):
        assert (REPO / target).exists(), f"wave_flume.py points at missing {target}"
