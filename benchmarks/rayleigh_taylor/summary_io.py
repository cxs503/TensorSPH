"""Campaign summary assembly for the Rayleigh–Taylor benchmark.

Why this module exists
---------------------
The board audit of ``CASE-SPH-RAYLEIGH-TAYLOR-MP`` (``audit_verdict_m16318.md``,
findings 5 and 6) recorded two artifact-integrity defects in the runner:

* **Finding 5 – aggregation loses shards.**  Every invocation rewrote
  ``runs/summary.json`` from *its own* case list only.  A campaign executed in
  several invocations (or on several shards, one ``--grid`` value per job) left
  the summary holding just the last invocation's rows: in the repository state
  the directory carried 8 case files and ``summary.json`` carried **1** row.  A
  downstream consumer that reads the summary — the natural thing to do — saw
  1/8 of the evidence and could not tell that the other 7 rows existed.
* **Finding 6 – ``NaN`` in JSON.**  The coarsest case has no usable fit window
  (``n_fit = 0``), so ``fit.sigma`` and ``sigma_over_ideal`` came out ``nan``.
  ``NaN`` is *not* valid JSON (RFC 8259 §6); ``json.dumps`` emits the bare
  token ``NaN`` and strict parsers (``json.loads(..., parse_constant=...)``,
  ``jq`` in strict mode, PostgreSQL ``json`` casts) reject the file.  The value
  was also indistinguishable from "field not applicable".

The fix is deliberately small and keeps one rule: **the per-case file stays
the single source of truth**, and ``summary.json`` becomes a *view* rebuilt
from every ``runs/<tag>.json``.  Consequences:

* a re-run of one tag replaces exactly that row (merge by ``tag``), never the
  whole campaign;
* a case with no usable fit is stored as ``status = "no_fit"`` with ``null``
  numbers, and every non-finite float anywhere in a row is converted to
  ``null`` before writing, so the file is always strict-JSON;
* ``--aggregate-only`` rebuilds the summary from artifacts already on disk,
  without recomputing anything.

Files matching ``NON_CASE_STEMS`` are never treated as case records: they are
themselves aggregates or reference tables, and including them would recurse.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

__all__ = [
    "CASE_KEYS",
    "NON_CASE_STEMS",
    "is_case_record",
    "load_case_rows",
    "load_summary_rows",
    "sanitize_row",
    "dumps_strict",
    "write_summary",
]

#: Fields copied from a full case record into the summary row.
CASE_KEYS = (
    "tag",
    "status",
    "fit",
    "fit_velocity_gauge",
    "reference",
    "sigma_over_ideal",
    "sigma_over_finite_depth",
    "sigma_over_viscous",
    "sigma_velocity_over_ideal",
    "max_amplitude",
    "derived",
    "wall_seconds",
)

#: Aggregate / reference tables in ``runs/`` that are not case records.
NON_CASE_STEMS = frozenset(
    {"summary", "table", "reference", "damping", "spurious", "aggregate"}
)

STATUS_OK = "ok"
STATUS_NO_FIT = "no_fit"


def _nonfinite(value) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _clean(value):
    """Recursively replace non-finite floats by ``None`` (JSON ``null``)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _fit_usable(fit) -> bool:
    if not isinstance(fit, dict):
        return False
    sigma = fit.get("sigma")
    if not isinstance(sigma, (int, float)) or _nonfinite(float(sigma)):
        return False
    n_fit = fit.get("n_fit")
    if isinstance(n_fit, (int, float)) and n_fit <= 0:
        return False
    return True


def is_case_record(payload) -> bool:
    """True when ``payload`` is a full per-case record (not an aggregate)."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("tag"), str)
        and "fit" in payload
    )


def sanitize_row(record: dict) -> dict:
    """Reduce a full case record to a strict-JSON summary row.

    Sets ``status`` explicitly: ``"no_fit"`` when the growth-rate fit holds no
    usable window, ``"ok"`` otherwise.  The growth-rate numbers are then set to
    ``null`` on a ``no_fit`` row so that no consumer can average ``NaN`` into a
    verdict, and every remaining non-finite float is nulled as well.
    """
    row = {key: record.get(key) for key in CASE_KEYS if key in record}
    row["tag"] = record["tag"]
    usable = _fit_usable(record.get("fit"))
    row["status"] = STATUS_OK if usable else STATUS_NO_FIT
    if not usable:
        for key in ("sigma_over_ideal", "sigma_over_finite_depth",
                    "sigma_over_viscous"):
            if key in row:
                row[key] = None
    return _clean(row)


def load_case_rows(runs_dir: Path) -> dict[str, dict]:
    """Read every ``runs/<tag>.json`` case file, keyed by tag.

    A file whose name stem is in :data:`NON_CASE_STEMS` is skipped, as is any
    JSON that does not look like a case record.  When the same tag appears in
    more than one file the lexicographically last file wins (deterministic, and
    it is the same rule a re-run uses).
    """
    runs_dir = Path(runs_dir)
    rows: dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*.json")):
        if path.stem in NON_CASE_STEMS:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not is_case_record(payload):
            continue
        rows[payload["tag"]] = sanitize_row(payload)
    return rows


def load_summary_rows(path: Path) -> dict[str, dict]:
    """Read an existing ``summary.json`` (a list of rows), keyed by tag.

    Used as the *base* of a rebuild so that rows whose per-case file has been
    archived or moved are not silently dropped by the next invocation.  Case
    files always win over these rows (they are the source of truth), and the
    rows of the current invocation win over both.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, list):
        return {}
    rows: dict[str, dict] = {}
    for record in payload:
        if is_case_record(record):
            rows[record["tag"]] = sanitize_row(record)
    return rows


def dumps_strict(payload) -> str:
    """``json.dumps`` that *refuses* to emit ``NaN``/``Infinity``."""
    return json.dumps(payload, indent=1, allow_nan=False)


def write_summary(runs_dir: Path, new_rows=(),
                  filename: str = "summary.json") -> list[dict]:
    """Rebuild ``runs/<filename>`` from the case files plus ``new_rows``.

    Precedence on a tag clash, highest first: ``new_rows`` (this invocation),
    the per-case file ``runs/<tag>.json``, the previous ``summary.json``.  The
    last level keeps rows alive whose case file has been archived, so the
    summary can only grow through an invocation, never shrink to it.  Rows are
    sorted by ``tag`` so the file is byte-stable when the case set is unchanged.
    """
    runs_dir = Path(runs_dir)
    rows = load_summary_rows(runs_dir / filename)
    rows.update(load_case_rows(runs_dir))
    for record in new_rows:
        if is_case_record(record):
            rows[record["tag"]] = sanitize_row(record)
    ordered = [rows[tag] for tag in sorted(rows)]
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / filename).write_text(dumps_strict(ordered))
    return ordered