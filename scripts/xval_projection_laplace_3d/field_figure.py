"""Field figure for the 3-D projection-SPH Young-Laplace case.

Panel (a): pressure field on a meridional slice (``y = 0``) -- the jump across
the interface is the measurement.  Panel (b): measured curvature around the
interface as a function of polar angle, against the analytic ``2/R``, which is
what the delta-weighted ``kappa R`` in the report averages over.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tensorsph.laplace_isph3d import laplace_isph3d_case  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
DX = 0.0025
torch.set_num_threads(8)


def main():
    results = {}
    for arm in ("mls", "color_quadric"):
        rep = laplace_isph3d_case(radius=0.02, dx=DX, sigma=0.072,
                                  curvature_method=arm, curvature_stencil=2.0,
                                  return_fields=True)
        f = rep.pop("fields")
        results[arm] = {"summary": rep,
                        "p": f["p"].tolist(), "pos": f["pos"].tolist(),
                        "kappa": f["kappa"].tolist(),
                        "delta_s": f["delta_s"].tolist(),
                        "normal": f["normal"].tolist(),
                        "active": f["active"].tolist()}
        print(arm, rep["dp_error_pct"], rep["kappaR_delta_weighted"], flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    rep = results["mls"]
    pos = torch.tensor(rep["pos"])
    p = torch.tensor(rep["p"])
    slice_mask = pos[:, 1].abs() < 0.5 * DX
    ax = axes[0]
    sc = ax.scatter(pos[slice_mask, 0] * 1e3, pos[slice_mask, 2] * 1e3, c=p[slice_mask],
                    s=18, cmap="coolwarm")
    th = [i * 2 * math.pi / 100 for i in range(101)]
    ax.plot(20 * torch.cos(torch.tensor(th)), 20 * torch.sin(torch.tensor(th)),
            "k--", lw=0.8, label="interface (r=R)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(f"(a) projected pressure, {rep['summary']['curvature_method']} arm\n"
                 f"$\\Delta p={rep['summary']['dp_measured_Pa']:.3f}$ Pa "
                 f"({rep['summary']['dp_error_pct']:+.2f} %)")
    fig.colorbar(sc, ax=ax, label="p [Pa]")
    ax.legend(fontsize=8)

    ax = axes[1]
    for arm, style in (("mls", "o"), ("color_quadric", "s")):
        rep = results[arm]
        pos = torch.tensor(rep["pos"])
        kap = torch.tensor(rep["kappa"])
        dsv = torch.tensor(rep["delta_s"])
        act = torch.tensor(rep["active"])
        r = torch.linalg.vector_norm(pos, dim=1)
        theta = torch.rad2deg(torch.acos((pos[:, 2] / r.clamp_min(1e-30)).clamp(-1, 1)))
        sel = act & (theta < 90.0)
        # binomial (0.5 deg) binning of kappa*R over the interface
        bins = torch.arange(0, 91, 3.0)
        idx = torch.bucketize(theta[sel], bins) - 1
        vals, means = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if int(m.sum()) > 5:
                vals.append(float(bins[b] + 1.5))
                means.append(float(kap[sel][m].mean()) * 0.02)
        ax.plot(vals, means, style + "-", ms=3, lw=0.9,
                label=f"{arm} (band mean)")
    ax.axhline(2.0, color="k", ls="--", lw=1, label=r"analytic $\kappa R=2$")
    ax.axhspan(1.9, 2.1, color="grey", alpha=0.15)
    ax.set_xlabel(r"polar angle $\theta$ [deg]")
    ax.set_ylabel(r"$\kappa R$")
    ax.set_title(f"(b) curvature around the interface ($R/\\Delta x=8$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "xval_projection_laplace_3d_field.png", dpi=140)
    (OUT / "xval_projection_laplace_3d_field.json").write_text(json.dumps(
        {k: v["summary"] for k, v in results.items()}, indent=1))
    print("figure written")


if __name__ == "__main__":
    main()
