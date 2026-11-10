"""JSON schema for the verification-ladder document.

The ladder report is the machine-readable contract of the campaign node: every
anchor states its reference, the measured quantity, the error and the gate that
was applied, and the document carries the provenance needed to reproduce it.
:func:`validate_ladder_json` is a small dependency-free validator so a consumer
does not need ``jsonschema`` to reject an incomplete artefact.
"""
from __future__ import annotations

import math

LADDER_SCHEMA_VERSION = "1.0"

ANCHOR_FIELDS = ("id", "title", "physics", "reference", "measured", "error_pct",
                 "gate", "verdict", "config", "series", "figures")
NEGCTL_FIELDS = ("id", "title", "mechanism", "metric", "expected", "measured",
                 "verdict")
TOP_FIELDS = ("schema_version", "node", "generated", "provenance", "anchors",
              "negative_controls", "summary", "campaign_criteria", "limitations")


def _require(container: dict, fields, where: str, errors: list[str]) -> None:
    for name in fields:
        if name not in container:
            errors.append(f"{where}: missing required field {name!r}")


def validate_ladder_json(doc: dict) -> list[str]:
    """Return a list of schema violations (empty means the document is valid)."""
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    _require(doc, TOP_FIELDS, "ladder", errors)
    if doc.get("schema_version") != LADDER_SCHEMA_VERSION:
        errors.append(f"schema_version must be {LADDER_SCHEMA_VERSION!r}")
    for key in ("anchors", "negative_controls"):
        if not isinstance(doc.get(key), list):
            errors.append(f"{key} must be a list")

    anchors = doc.get("anchors") or []
    expected_ids = {"C1_hydrostatic", "C2_tgv", "C3_dambreak"}
    ids = set()
    for index, anchor in enumerate(anchors):
        where = f"anchors[{index}]"
        if not isinstance(anchor, dict):
            errors.append(f"{where} must be an object")
            continue
        _require(anchor, ANCHOR_FIELDS, where, errors)
        ids.add(anchor.get("id"))
        error = anchor.get("error_pct")
        if not isinstance(error, (int, float)) or not math.isfinite(float(error)):
            errors.append(f"{where}.error_pct must be a finite number")
        gate = anchor.get("gate")
        if not isinstance(gate, dict):
            errors.append(f"{where}.gate must be an object")
        else:
            _require(gate, ("quantity", "limit_pct", "pass"), f"{where}.gate", errors)
            if gate.get("pass") is not (float(error) <= float(gate.get("limit_pct", 0.0))):
                errors.append(f"{where}: verdict inconsistent with error vs limit")
            if anchor.get("verdict") != ("PASS" if gate.get("pass") else "FAIL"):
                errors.append(f"{where}: verdict must be PASS/FAIL from gate.pass")
    missing = expected_ids - ids
    if missing:
        errors.append(f"anchors missing ids: {sorted(missing)}")

    for index, control in enumerate(doc.get("negative_controls") or []):
        where = f"negative_controls[{index}]"
        if not isinstance(control, dict):
            errors.append(f"{where} must be an object")
            continue
        _require(control, NEGCTL_FIELDS, where, errors)

    provenance = doc.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        _require(provenance, ("git_sha", "python", "torch", "package_version",
                              "command"), "provenance", errors)

    summary = doc.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        _require(summary, ("anchors_total", "anchors_passed", "all_passed",
                           "honest_failures"), "summary", errors)
    return errors