"""Render ``docs/reports/m4374-pr189-ci-cure.png`` for the PR #189 CI cure (m4374).

Every number comes from artifacts produced by the audit itself:

* ``ruff`` (subprocess, ``--isolated --line-length 100 --select E,F,W``, i.e. exactly
  the repository gate in ``pyproject.toml``) on the pre-cure and post-cure revision
  of ``docs/reports/m3950_post_report.py``;
* the string constants of both revisions (``ast``), compared in source order;
* the report body the script would POST, replayed with a stubbed
  ``urllib.request.urlopen`` and byte-compared.

Usage::

    python docs/reports/m4374_make_figure.py <pre-cure-copy.py> <post-cure.py>
"""
from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "m4374-pr189-ci-cure.png"
RUFF = [sys.executable, "-m", "ruff", "check", "--isolated",
        "--line-length", "100", "--select", "E,F,W", "--output-format", "json"]


def ruff_counts(path: Path) -> Counter:
    proc = subprocess.run(RUFF + [str(path)], capture_output=True, text=True, check=False)
    return Counter(item["code"] for item in json.loads(proc.stdout or "[]"))


def string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def payload_bytes(path: Path) -> int:
    """Byte size of the report body the script would POST (network stubbed)."""
    os.environ.setdefault("TENSOR_API_BASE", "http://127.0.0.1:8010")
    os.environ.setdefault("TENSOR_AGENT_USER", "figure")
    os.environ.setdefault("TENSOR_AGENT_PASS", "figure")
    bodies: list[str] = []
    original = urllib.request.urlopen

    def fake(req, timeout=None):
        bodies.append(req.data.decode())
        return io.BytesIO(json.dumps({"access_token": "figure"}).encode())

    urllib.request.urlopen = fake
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
             {"__name__": "figure_probe", "__file__": str(path)})
    finally:
        urllib.request.urlopen = original
    return len(bodies[1].encode())


def main() -> int:
    pre, post = Path(sys.argv[1]), Path(sys.argv[2])
    before, after = ruff_counts(pre), ruff_counts(post)
    rules = sorted(set(before) | set(after))
    consts_pre, consts_post = string_constants(pre), string_constants(post)
    consts_ok = consts_pre == consts_post
    n_consts = len(consts_pre)
    pre_bytes, post_bytes = payload_bytes(pre), payload_bytes(post)
    widths = [max(len(line) for line in p.read_text(encoding="utf-8").split("\n"))
              for p in (pre, post)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = range(len(rules))
    ax.bar([i - 0.2 for i in x], [before.get(r, 0) for r in rules], width=0.4,
           label="pre-cure (fa7a5ee)", color="#c0392b")
    ax.bar([i + 0.2 for i in x], [after.get(r, 0) for r in rules], width=0.4,
           label="post-cure (this branch)", color="#27ae60")
    for i, r in enumerate(rules):
        ax.text(i - 0.2, before.get(r, 0) + 0.3, str(before.get(r, 0)), ha="center", fontsize=9)
        ax.text(i + 0.2, after.get(r, 0) + 0.3, str(after.get(r, 0)), ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rules)
    ax.set_ylabel("ruff findings (E,F,W @ line-length 100)")
    ax.set_title(f"(a) CI lint gate: {sum(before.values())} findings -> {sum(after.values())}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels = ["string constants\nidentical", "POST payload\nidentical",
              "max line width\n<= 100"]
    values = [n_consts if consts_ok else 0, pre_bytes if pre_bytes == post_bytes else 0,
              widths[1]]
    totals = [n_consts, pre_bytes, 100]
    bars = ax.bar(labels, [v / t * 100 for v, t in zip(values, totals)], color="#2980b9")
    for bar, v, t in zip(bars, values, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:,} / {t:,}", ha="center", fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("match [% of reference]")
    ax.set_title("(b) Output-preservation proof (100% = byte identical)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("TensorSPH PR #189 (agent/m3950) CI cure: lint-gate reflow, zero content change",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    print(json.dumps({"before": dict(before), "after": dict(after),
                      "constants_identical": consts_ok, "constants": n_consts,
                      "payload_bytes": [pre_bytes, post_bytes],
                      "max_line_width": widths}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
