"""SQLite evidence store for the free-surface verification face.

Every number that reaches a report goes through this module, so that a report
can be regenerated from the database alone and every figure/table cell carries
the query that produced it ("no hand-typed numbers").  The store is deliberately
small: three tables and one query helper.

Design rules
------------
* A metric is written once, together with the run that produced it.  There is no
  ``update`` path: a re-run is a new row, so the history of a gate is auditable.
* A *reference* (experimental table, analytic constant) is registered with its
  file path and SHA-256 and its evidence grade.  A gate that uses a reference
  graded ``WITHHELD`` must say so in the metric's ``provenance`` field; the
  store cannot enforce that, but :func:`tensorsph.verification.report` reads the
  grade back out and writes it into the report.
* :meth:`EvidenceStore.digest` returns ``{"sql": ..., "value": ...}`` for every
  reported number, which is what the report's provenance block is built from.

The store is a plain ``.sqlite`` file; there is no server and no ORM.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = "1.0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    case_name   TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    host        TEXT NOT NULL,
    git_commit  TEXT,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id     INTEGER NOT NULL,
    name       TEXT NOT NULL,
    value      REAL,
    unit       TEXT NOT NULL DEFAULT '',
    gate_limit REAL,
    verdict    TEXT,
    provenance TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS series (
    run_id  INTEGER NOT NULL,
    name    TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS references_ (
    name     TEXT PRIMARY KEY,
    path     TEXT NOT NULL,
    sha256   TEXT NOT NULL,
    grade    TEXT NOT NULL,
    note     TEXT NOT NULL DEFAULT ''
);
"""


def sha256_of(path: str | Path) -> str:
    """SHA-256 of a file, so a reference table cannot be swapped silently."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Metric:
    """A reported number with the SQL that produced it."""

    name: str
    value: float | None
    unit: str
    gate_limit: float | None
    verdict: str | None
    provenance: str


class EvidenceStore:
    """Write-once evidence log for a verification campaign."""

    def __init__(self, path: str | Path = "artifacts/verification/evidence.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            ("schema_version", SCHEMA_VERSION))
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            ("created_utc", utc_now()))
        self._conn.commit()

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes ----------------------------------------------------------
    def add_run(self, case_name: str, config: dict, notes: str = "",
                git_commit: str | None = None) -> int:
        """Register one execution.  Returns the ``run_id`` used by everything else."""
        if git_commit is None:
            git_commit = os.environ.get("GIT_COMMIT") or _git_head()
        cur = self._conn.execute(
            "INSERT INTO runs(case_name, config_json, created_utc, host, git_commit, notes)"
            " VALUES(?, ?, ?, ?, ?, ?)",
            (case_name, json.dumps(config, sort_keys=True, default=str), utc_now(),
             os.uname().nodename, git_commit, notes))
        self._conn.commit()
        return int(cur.lastrowid)

    def add_metric(self, run_id: int, name: str, value: float | None,
                   unit: str = "", gate_limit: float | None = None,
                   verdict: str | None = None, provenance: str = "") -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError(f"metric {name!r} must be a number, got {type(value)!r}")
        self._conn.execute(
            "INSERT INTO metrics(run_id, name, value, unit, gate_limit, verdict,"
            " provenance) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (run_id, name, None if value is None else float(value), unit,
             None if gate_limit is None else float(gate_limit), verdict,
             provenance))
        self._conn.commit()

    def add_metrics(self, run_id: int, metrics: dict[str, dict]) -> None:
        """Bulk form: ``{name: {value, unit, gate_limit, verdict, provenance}}``."""
        for name, spec in metrics.items():
            self.add_metric(run_id, name, spec.get("value"),
                            spec.get("unit", ""), spec.get("gate_limit"),
                            spec.get("verdict"), spec.get("provenance", ""))

    def add_series(self, run_id: int, name: str, payload) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO series(run_id, name, payload) VALUES(?, ?, ?)",
            (run_id, name, json.dumps(payload, sort_keys=True)))
        self._conn.commit()

    def add_reference(self, name: str, path: str | Path, grade: str,
                      note: str = "") -> str:
        digest = sha256_of(path)
        self._conn.execute(
            "INSERT OR REPLACE INTO references_(name, path, sha256, grade, note)"
            " VALUES(?, ?, ?, ?, ?)",
            (name, str(path), digest, grade, note))
        self._conn.commit()
        return digest

    # -- reads -----------------------------------------------------------
    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params).fetchall())

    def metric(self, run_id: int, name: str) -> Metric:
        rows = self.query(
            "SELECT name, value, unit, gate_limit, verdict, provenance FROM metrics"
            " WHERE run_id=? AND name=?", (run_id, name))
        if not rows:
            raise KeyError(f"no metric {name!r} for run {run_id}")
        row = rows[0]
        return Metric(row["name"], row["value"], row["unit"], row["gate_limit"],
                      row["verdict"], row["provenance"])

    def digest(self, run_id: int, name: str) -> dict:
        """The report-facing form of a number: value plus the query that made it.

        This is the mechanism behind the "no hand-typed numbers" gate: a report
        cell is built from this dict, never from a literal.
        """
        sql = ("SELECT value, unit, gate_limit, verdict FROM metrics "
               "WHERE run_id=? AND name=?")
        rows = self.query(sql, (run_id, name))
        if not rows:
            raise KeyError(f"no metric {name!r} for run {run_id}")
        row = rows[0]
        return {"sql": sql, "params": [run_id, name], "value": row["value"],
                "unit": row["unit"], "gate_limit": row["gate_limit"],
                "verdict": row["verdict"]}

    def scalar(self, sql: str, params: tuple = ()) -> float | None:
        """Run an aggregate query and return its first column.

        Used where the reported number is itself a SQL aggregate over the metric
        table (``MAX``, ``AVG`` ...), so the aggregation is auditable too.
        """
        rows = self.query(sql, params)
        return None if not rows else rows[0][0]

    def run_count(self, case_name: str | None = None) -> int:
        if case_name is None:
            return int(self.scalar("SELECT COUNT(*) FROM runs") or 0)
        return int(self.scalar("SELECT COUNT(*) FROM runs WHERE case_name=?", (case_name,)) or 0)

    def metric_names(self, run_id: int) -> list[str]:
        return [r["name"] for r in self.query(
            "SELECT name FROM metrics WHERE run_id=? ORDER BY name", (run_id,))]

    def gate_table(self, case_name: str) -> list[dict]:
        """Every gated metric of a case, newest run first."""
        rows = self.query(
            "SELECT r.run_id, r.created_utc, m.name AS metric, m.value, m.unit,"
            " m.gate_limit,"
            " m.verdict FROM runs r JOIN metrics m USING(run_id) WHERE r.case_name=?"
            " AND m.gate_limit IS NOT NULL ORDER BY r.run_id DESC, m.name", (case_name,))
        return [dict(r) for r in rows]


def _git_head() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent in some sandboxes
        return None
