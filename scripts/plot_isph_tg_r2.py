#!/usr/bin/env python3
"""Figure for docs/xval_isph_projection_taylor_green_r2.md (programmatic, no hand numbers)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
doc = json.loads((REPO / "outputs/isph_taylor_green/error_budget.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for row in doc["rows"]:
    tr = row["trace"]
    t = [r["t"] for r in tr]
    label = f"dx={row['dx']:g}, Co={row['courant']:g}"
    axes[0].semilogy(t, [max(r["rel_l2_pct"], 1e-3) for r in tr], "o-", ms=3, label=label)
    tr2 = [r for r in tr if r["min_pair_distance_over_dx"] is not None]
    axes[1].plot([r["t"] for r in tr2], [r["min_pair_distance_over_dx"] for r in tr2],
                 "s-", ms=3, label=label)
axes[0].axhline(15.0, color="k", ls="--", lw=1)
axes[0].text(0.02, 17, "owner standard 15 %", fontsize=8)
axes[0].set_xlabel("t"); axes[0].set_ylabel("velocity rel. L2 error [%]")
axes[0].set_title("Taylor-Green ISPH: error growth (log scale)")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].axhline(1.0, color="k", ls=":", lw=1)
axes[1].set_xlabel("t"); axes[1].set_ylabel("min pair distance / dx")
axes[1].set_title("cloud decoherence precedes the blow-up")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
out = REPO / "docs/reports/isph_taylor_green_r2.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print("wrote", out)
