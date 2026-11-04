"""Claim-to-evidence audit for TensorSPH.

TensorSPH publishes headline numbers in ``README.md`` and in ~14 verification
documents.  Nothing in the test suite ties those numbers back to the JSON that
was supposed to produce them, so the prose drifts from the data without any
signal (the surface-tension limitation bullet in ``README.md`` was still saying
"no surface-tension model" two model revisions after the CSF modules merged).

This tool reads a declarative ledger (``docs/claims/ledger.json``) in which every
published claim is bound to machine-checkable evidence, evaluates each binding,
and reports an evidence grade plus any contradiction:

* ``REAL``     - the bound evidence resolves and matches the published value.
* ``WITHHELD`` - the claim is published but the bound evidence is missing from
                 the repository, records a *failure*, or records a different
                 quantity.  These are listed as debt, not silently dropped.

Contradictions are fatal (exit 1): an ``absence_claim`` ("README says the
capability does not exist") whose capability is demonstrably present, or an
``artifact_metric`` whose artifact disagrees with the pinned value (someone
changed the data or the number and did not update the other side).

Usage::

    python tools/claim_audit.py                     # report to stdout
    python tools/claim_audit.py --check             # exit 1 on contradiction
    python tools/claim_audit.py --json out.json --md out.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

LEDGER_REL = Path("docs/claims/ledger.json")
README_REL = Path("README.md")

_DEF_RE = r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+{name}\b"


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer. ``""`` is the whole document."""
    if pointer in ("", "/"):
        return document
    node = document
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            if token not in node:
                raise KeyError(f"pointer {pointer!r}: no key {token!r}")
            node = node[token]
        else:
            raise KeyError(f"pointer {pointer!r}: cannot descend into {type(node).__name__}")
    return node


def _matches(relation: str, actual: Any, expected: Any, tol: float) -> bool:
    if relation == "eq":
        return actual == expected
    if relation == "close_to":
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return abs(float(actual) - float(expected)) <= tol
    raise ValueError(f"unknown relation {relation!r}")


def _declares_symbol(path: Path, symbol: str) -> bool:
    if not path.is_file():
        return False
    pattern = re.compile(_DEF_RE.format(name=re.escape(symbol)), re.MULTILINE)
    return bool(pattern.search(path.read_text(encoding="utf-8", errors="replace")))


def _declares_test(path: Path, name: str) -> bool:
    return _declares_symbol(path, name)


def _check_code_present(root: Path, evidence: dict) -> dict:
    wanted = [(evidence["file"], list(evidence["symbols"]))]
    for extra in evidence.get("also_require", []) or []:
        wanted.append((extra["file"], list(extra["symbols"])))
    missing = []
    for rel, symbols in wanted:
        path = root / rel
        if not path.is_file():
            missing.append(f"{rel} (file absent)")
            continue
        missing.extend(f"{rel}::{s}" for s in symbols if not _declares_symbol(path, s))
    return {"targets": [rel for rel, _ in wanted], "missing": missing}


def _check_test_present(root: Path, evidence: dict) -> dict:
    path = root / evidence["file"]
    missing = []
    if not path.is_file():
        missing.append(f"{evidence['file']} (file absent)")
    else:
        missing.extend(
            f"{evidence['file']}::{t}" for t in evidence["tests"] if not _declares_test(path, t)
        )
    return {"targets": [evidence["file"]], "missing": missing}


def _check_artifact_metric(root: Path, evidence: dict) -> dict:
    artifact = root / evidence["artifact"]
    detail: dict[str, Any] = {"artifact": evidence["artifact"], "checks": []}
    if not artifact.is_file():
        detail["reason"] = "artifact_missing"
        return detail
    try:
        document = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        detail["reason"] = f"artifact_unparsable: {exc}"
        return detail

    specs = [
        {
            "pointer": evidence["pointer"],
            "relation": evidence["relation"],
            "expected": evidence["expected"],
            "abs_tol": evidence.get("abs_tol", 0.0),
        }
    ] + list(evidence.get("also_pin", []) or [])

    for spec in specs:
        abs_tol = spec.get("abs_tol", 0.0)
        row = {"pointer": spec["pointer"], "relation": spec["relation"],
               "expected": spec["expected"], "abs_tol": abs_tol}
        try:
            actual = resolve_pointer(document, spec["pointer"])
        except (KeyError, ValueError, IndexError) as exc:
            row.update(actual=None, ok=False, error=str(exc))
        else:
            row.update(actual=actual,
                       ok=_matches(spec["relation"], actual, spec["expected"], row["abs_tol"]))
        detail["checks"].append(row)
    return detail


def _grade(claim: dict, detail: dict) -> tuple[str, str]:
    kind = claim["kind"]
    if kind == "prose_only":
        return "WITHHELD", claim["evidence"]["why_absent"]
    if detail.get("reason", "").startswith("artifact_"):
        return "WITHHELD", detail["reason"]
    if kind in ("code_present", "test_present"):
        if detail["missing"]:
            return "WITHHELD", "evidence missing: " + ", ".join(detail["missing"])
        return "REAL", "all declarations present"
    mismatched = [c for c in detail.get("checks", []) if not c["ok"]]
    if mismatched:
        return "DRIFT", "; ".join(
            f"{c['pointer']} = {c['actual']!r} != {c['expected']!r}"
            + (f" ({c['error']})" if c.get("error") else "")
            for c in mismatched
        )
    return "REAL", "artifact matches every pinned value"


def audit(root: Path) -> dict:
    """Evaluate the claim ledger rooted at ``root``. Pure read, no writes."""
    root = Path(root)
    ledger_path = root / LEDGER_REL
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    readme_path = root / README_REL
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""

    results = []
    for claim in ledger["claims"]:
        kind = claim["kind"]
        if kind == "prose_only":
            detail: dict[str, Any] = {"would_need": claim["evidence"]["would_need"]}
        elif kind == "code_present":
            detail = _check_code_present(root, claim["evidence"])
        elif kind == "test_present":
            detail = _check_test_present(root, claim["evidence"])
        elif kind == "artifact_metric":
            detail = _check_artifact_metric(root, claim["evidence"])
        else:
            raise ValueError(f"claim {claim['id']!r}: unknown kind {kind!r}")
        grade, reason = _grade(claim, detail)
        results.append({
            "id": claim["id"],
            "kind": kind,
            "statement": claim["statement"],
            "grade": grade,
            "reason": reason,
            "verdict": claim.get("verdict"),
            "public_in": claim.get("public_in", []),
            "detail": detail,
        })

    by_id = {r["id"]: r for r in results}
    findings = []
    for claim in results:
        if claim["grade"] == "DRIFT":
            findings.append({
                "severity": "fatal",
                "type": "artifact_drift",
                "claim": claim["id"],
                "message": f"{claim['statement']} -> {claim['reason']}",
                "fix": "update the published number or the artifact, then re-run the audit",
            })

    for absence in ledger.get("absence_claims", []):
        capability = by_id.get(absence["capability"])
        if capability is None:
            findings.append({
                "severity": "fatal", "type": "ledger_error", "claim": absence["id"],
                "message": ("absence claim points at unknown claim "
                            f"{absence['capability']!r}"),
            })
            continue
        if not re.search(absence["pattern"], readme):
            continue
        if capability["grade"] == "REAL":
            findings.append({
                "severity": "fatal",
                "type": "stale_absence_claim",
                "claim": absence["id"],
                "message": (f"README still asserts {absence['pattern']!r} while "
                            f"{capability['id']} is present ({capability['reason']})"),
                "fix": absence["note"],
            })
        else:
            findings.append({
                "severity": "debt",
                "type": "absence_claim_unverified",
                "claim": absence["id"],
                "message": (f"README asserts {absence['pattern']!r}; the capability is "
                            f"{capability['grade']}, so the sentence is not proven wrong"),
            })

    debt = [r for r in results if r["grade"] == "WITHHELD"]
    summary = {
        "claims": len(results),
        "real": sum(r["grade"] == "REAL" for r in results),
        "withheld": len(debt),
        "drift": sum(r["grade"] == "DRIFT" for r in results),
        "fatal_findings": sum(f["severity"] == "fatal" for f in findings),
        "debt_findings": sum(f["severity"] == "debt" for f in findings),
        "pass": not any(f["severity"] == "fatal" for f in findings),
    }
    return {"summary": summary, "claims": results, "findings": findings}


def to_markdown(report: dict, ledger_path: str = str(LEDGER_REL)) -> str:
    s = report["summary"]
    lines = [
        "# TensorSPH claim audit（声明↔证据台账）",
        "",
        "> 由 `tools/claim_audit.py` 从 "
        f"`{ledger_path}` 生成，请勿手改。重跑：`python tools/claim_audit.py "
        "--json docs/claims/report.json --md docs/claim_audit.md`",
        "",
        "台账解决的问题：README 与 14 份验证文档里的数字此前**没有任何机器可裁的绑定**，",
        "所以正文可以漂移而 CI 全绿（实例：CSF 模块合入两个版本后，README 仍写着",
        "“No surface-tension model”）。现在每条公开声明都绑定到文件 / JSON pointer /",
        "测试 id，一条不一致就让 `--check` 退出码非零。",
        "",
        "## 汇总",
        "",
        f"- 声明条数 **{s['claims']}**：REAL **{s['real']}**"
        f" / WITHHELD **{s['withheld']}** / DRIFT **{s['drift']}**",
        f"- 致命发现 **{s['fatal_findings']}**，债务提示 **{s['debt_findings']}**",
        f"- 判定：**{'PASS' if s['pass'] else 'FAIL'}**",
        "",
        "![claim audit](claims/claim_audit.png)",
        "",
        "图：左 = 每条声明的证据等级（蓝 REAL / 橙 WITHHELD）；右 = B2 dam-break 各项闸门按",
        "`measured/limit` 归一，虚线为验收线 1.0——实验对照是唯一越线项（0.573/0.15 = 3.8×）。",
        "由 `tools/make_claim_audit_figure.py` 从 `docs/claims/report.json` 生成。",
        "",
        "证据等级定义：REAL = 绑定证据存在且数值/符号对得上；WITHHELD = 声明已发布但绑定证据",
        "缺失、记录的是失败、或记录的是另一个量（列入债务，不静默丢弃）。",
        "",
        "## 逐条台账",
        "",
        "| 声明 | 类型 | 证据等级 | 依据 |",
        "|---|---|---|---|",
    ]
    for claim in report["claims"]:
        reason = claim["reason"].replace("|", "\\|")
        if len(reason) > 150:
            reason = reason[:147] + "…"
        lines.append(f"| `{claim['id']}` | {claim['kind']} | **{claim['grade']}** | {reason} |")

    lines += ["", "## 债务清单（WITHHELD：公开了数字，但仓库里没有可复读的证据）", ""]
    if not report["claims"] or not any(c["grade"] == "WITHHELD" for c in report["claims"]):
        lines.append("（无）")
    for claim in report["claims"]:
        if claim["grade"] != "WITHHELD":
            continue
        lines.append(f"- **`{claim['id']}`** — {claim['statement']}")
        lines.append(f"  - 缺什么：{claim['reason']}")
        if claim.get("verdict"):
            lines.append(f"  - 等级说明：{claim['verdict']}")

    lines += ["", "## 发现", ""]
    if not report["findings"]:
        lines.append("（无致命发现）")
    for f in report["findings"]:
        lines.append(f"- **[{f['severity']}] {f['type']}** `{f.get('claim')}`：{f['message']}")
        if f.get("fix"):
            lines.append(f"  - 处置：{f['fix']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--json", help="write the machine report here")
    parser.add_argument("--md", help="write the generated markdown report here")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any fatal finding is reported")
    args = parser.parse_args(argv)

    report = audit(Path(args.root))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(to_markdown(report), encoding="utf-8")
    if not args.json and not args.md:
        print(to_markdown(report))
    s = report["summary"]
    print(f"claim_audit: {s['real']} REAL / {s['withheld']} WITHHELD / {s['drift']} DRIFT, "
          f"{s['fatal_findings']} fatal -> {'PASS' if s['pass'] else 'FAIL'}",
          file=sys.stderr)
    return 0 if (report["summary"]["pass"] or not args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
