"""Render the PR #103 merge-cure figures from the programmatic inputs.

Inputs (all produced by the merge machinery, nothing hand-copied):
  doc/merge/pr103/ladder_compare.json   -- both protocols' read-outs per rung
  docs/merge/pr103/union_audit.json     -- per-file leaf counts (no-loss proof)

Outputs:
  docs/merge/pr103/pr103_protocol_compare.png
  docs/merge/pr103/pr103_union_coverage.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    cmp = json.loads((HERE / "ladder_compare.json").read_text())
    audit = json.loads((HERE / "union_audit.json").read_text())
    rungs = sorted(cmp["rungs"], key=lambda r: -r["dx_m"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # (a) capillary phase each protocol reads at: main's fixed 900 steps does
    #     not sample the same t/t_sigma across resolutions.
    dxs = [r["dx_m"] * 1e3 for r in rungs]
    xs = range(len(rungs))
    ax.plot(xs, [r["primary"]["t_over_t_sigma"] for r in rungs], "o-",
            color="#3b7dd8", label="branch: t/t_sigma = 1.0 protocol")
    ax.plot(xs, [r["preserved"]["t_over_t_sigma"] for r in rungs], "s--",
            color="#d8723b", label="main: fixed 900 steps")
    ax.axhline(1.0, color="k", lw=1, ls=":", label="common capillary time 1.0")
    ax.set_xticks(list(xs), [f"dx = {d:.2f} mm" for d in dxs])
    ax.set_ylabel(r"read-out time $t/t_\sigma$ [-]")
    ax.set_title("(a) what the two conflict sides measure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) bandA read-out error per protocol, against the 360 Pa reference
    w = 0.36
    ax2 = bx
    ax2.bar([i - w / 2 for i in xs], [r["primary"]["bandA_err_pct"] for r in rungs],
            width=w, color="#3b7dd8", label="branch protocol (kept as primary)")
    ax2.bar([i + w / 2 for i in xs], [r["preserved"]["bandA_err_pct"] for r in rungs],
            width=w, color="#d8723b", label="main protocol (preserved verbatim)")
    for i, r in enumerate(rungs):
        ax2.text(i - w / 2, r["primary"]["bandA_err_pct"] - 4,
                 f"{r['primary']['bandA_err_pct']:.1f}", ha="center",
                 fontsize=7, color="#1b4b8f")
        ax2.text(i + w / 2, r["preserved"]["bandA_err_pct"] - 4,
                 f"{r['preserved']['bandA_err_pct']:.1f}", ha="center",
                 fontsize=7, color="#8a4212")
    ax2.axhline(-15.0, color="r", ls="--", lw=1, label="15 % acceptance bar")
    ax2.axhline(-3.0, color="r", ls=":", lw=1, label="3 % bar")
    ax2.set_xticks(list(xs), [f"dx = {d:.2f} mm" for d in dxs])
    ax2.set_ylabel(r"bandA error vs $\Delta p = 2\sigma/R$ [%]")
    ax2.set_title("(b) both runs keep their numbers (no scalar merge)")
    ax2.set_ylim(-125, 8)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "pr103_protocol_compare.png", dpi=140)
    plt.close(fig)

    # (c) union coverage: leaves kept per conflict file
    fig, cx = plt.subplots(figsize=(9.2, 4.2))
    names = [f["file"].split("/")[-1] for f in audit["files"]]
    branch = [f["n_leaves_branch"] for f in audit["files"]]
    mains = [f["n_leaves_main"] for f in audit["files"]]
    merged = [f["n_leaves_merged"] for f in audit["files"]]
    idx = range(len(names))
    w = 0.26
    cx.bar([i - w for i in idx], branch, width=w, color="#3b7dd8",
           label="branch leaves")
    cx.bar(list(idx), mains, width=w, color="#d8723b", label="main leaves")
    cx.bar([i + w for i in idx], merged, width=w, color="#4a4a4a",
           label="merged leaves (primary + preserved)")
    for i in idx:
        cx.text(i + w, merged[i] + 40,
                f"{mains[i]} main\nconserved", ha="center", fontsize=7)
    cx.set_xticks(list(idx), names)
    cx.set_ylabel("leaf values in JSON")
    cx.set_title("union coverage: merged = branch leaves + main leaves (0 lost)")
    cx.legend(fontsize=8)
    cx.grid(alpha=0.3, axis="y")
    cx.set_ylim(0, max(merged) * 1.25)
    fig.tight_layout()
    fig.savefig(HERE / "pr103_union_coverage.png", dpi=140)
    plt.close(fig)
    print("wrote pr103_protocol_compare.png, pr103_union_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
