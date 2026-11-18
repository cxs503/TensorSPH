#!/usr/bin/env python
"""Honesty audit for the TensorSPH capability registry (m25063, CAP-REG-SPH).

Mechanical re-derivation of every flag the decorators claim:

* ``wired``    -> a non-test, non-registry module under ``src/tensorsph/``
                  references the capability's module base name (word match).
* ``tested``   -> ``test_path`` exists and defines at least one ``test*``
                  function (AST).
* ``verified`` -> ``evidence.grade`` in {ANALYTIC, XVALUE} and
                  ``evidence.max_error_pct`` present; the evidence note must
                  cite at least one file that exists in the repository.
* ``endpoint`` -> scheme in {cli:, route:, api:} and the capability is wired.

Output: ``docs/m25063/registry_audit.{md,json}``.  Exit 1 when any
error-severity finding is open.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tensorsph"
TESTS = ROOT / "tests"
REGISTRY_CLI = ROOT / "src" / "tensorsph" / "registry.py"
OUT_DIR = ROOT / "docs" / "m25063"

VERIFYING_GRADES = ("ANALYTIC", "XVALUE")
ENDPOINT_SCHEMES = ("cli:", "route:", "api:")


def static_dump() -> dict:
    out = subprocess.run([sys.executable, str(REGISTRY_CLI), "--static"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def src_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def module_base(module_path: str) -> str:
    return module_path.rsplit(".", 1)[-1]


def wired_evidence(cap: dict) -> list[str]:
    """Other src modules that reference this capability's module."""
    base = module_base(cap["module_path"])
    own = (SRC / (cap["module_path"].split(".", 1)[-1].replace(".", "/") + ".py"))
    hits = []
    for p in src_files():
        if p.name == "registry.py" or p == own:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(rf"\b{re.escape(base)}\b", text):
            hits.append(str(p.relative_to(ROOT)))
    return hits


def test_functions(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    return [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test")]


def cited_files(note: str) -> list[str]:
    return re.findall(r"[\w./-]+\.(?:md|json|py)", note or "")


def audit(caps: list[dict]) -> list[dict]:
    findings: list[dict] = []

    def add(severity, code, cid, detail, disposition="open"):
        findings.append({"severity": severity, "code": code,
                         "capability_id": cid, "detail": detail,
                         "disposition": disposition})

    for cap in caps:
        cid = cap["id"]
        # --- wired -------------------------------------------------------
        if cap["wired"]:
            hits = wired_evidence(cap)
            if not hits:
                add("error", "WIRED_UNSUBSTANTIATED", cid,
                    "wired=True but no non-test src module references "
                    f"{module_base(cap['module_path'])!r}")
        # --- tested ------------------------------------------------------
        if cap["tested"]:
            tp = cap.get("test_path")
            if not tp or not (ROOT / tp).is_file():
                add("error", "TESTED_NO_TEST_FILE", cid,
                    f"tested=True but test_path={tp!r} missing")
            elif not test_functions(ROOT / tp):
                add("error", "TEST_FILE_EMPTY", cid,
                    f"test_path={tp!r} defines no test* function")
        # --- verified ----------------------------------------------------
        if cap["verified"]:
            ev = cap.get("evidence") or {}
            grade = ev.get("grade")
            if grade not in VERIFYING_GRADES:
                add("error", "VERIFIED_ON_WEAK_EVIDENCE", cid,
                    f"verified=True with grade={grade!r}")
            if "max_error_pct" not in ev:
                add("error", "VERIFIED_NO_ERROR_BAND", cid,
                    "verified=True without evidence.max_error_pct")
            cited = cited_files(str(ev.get("note", "")))
            missing = [c for c in cited if not (ROOT / c).exists()]
            if cited and missing:
                add("error", "EVIDENCE_CITATION_MISSING", cid,
                    f"evidence note cites missing file(s): {missing}")
            if not cited:
                add("warning", "EVIDENCE_NO_CITATION", cid,
                    "verified=True but evidence note cites no repository file")
        # --- endpoint ----------------------------------------------------
        if cap.get("endpoint"):
            if not cap["endpoint"].startswith(ENDPOINT_SCHEMES):
                add("error", "ENDPOINT_BAD_SCHEME", cid,
                    f"endpoint={cap['endpoint']!r}")
            if not cap["wired"]:
                add("error", "ENDPOINT_WITHOUT_WIRING", cid,
                    "endpoint declared for a non-wired capability")
    return findings


def main() -> int:
    payload = static_dump()
    caps = payload["capabilities"]
    findings = audit(caps)
    open_findings = [f for f in findings if f["disposition"] == "open"]

    md = ["# m25063 — TensorSPH capability registry audit", "",
          f"source_commit: `{payload['source_commit']}`", "",
          f"capabilities: {len(caps)}  findings: {len(findings)}  "
          f"open: {len(open_findings)}", "",
          "| severity | code | capability | detail | disposition |",
          "|---|---|---|---|---|"]
    for f in findings:
        md.append(f"| {f['severity']} | {f['code']} | {f['capability_id']} "
                  f"| {f['detail']} | {f['disposition']} |")
    md += ["", "## Wired evidence (mechanical)", ""]
    for cap in caps:
        if cap["wired"]:
            md.append(f"- `{cap['id']}` <- {', '.join(wired_evidence(cap))}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "registry_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT_DIR / "registry_audit.json").write_text(
        json.dumps({"source_commit": payload["source_commit"],
                    "count": len(caps), "findings": findings,
                    "open_findings": len(open_findings)},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"findings={len(findings)} open={len(open_findings)}")
    return 1 if open_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
