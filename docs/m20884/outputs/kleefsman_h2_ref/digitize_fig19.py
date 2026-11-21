"""Digitise the experimental water-height trace at probe H2 from Kleefsman et al. (2005) Fig. 19.

Source PDF: Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) J. Comput. Phys. 206:363-393,
doi:10.1016/j.jcp.2004.12.007 -- openly reachable copy at
https://pure.rug.nl/ws/files/2941055/2005JCompPhysKleefsman.pdf (the research.rug.nl mirror of the
same file is behind a Cloudflare challenge).  Fig. 19 is a raster JPEG (1182x450 px) embedded in
page 22 of the PDF, so the traces must be digitised from pixels; the axis tick labels are part of the
same raster and were read pixel-by-pixel (glyphs dumped as ASCII and identified); see
docs/kleefsman-h2-experimental-anchor.md for the audit trail.

Two curves share the panel:
  * thick solid line, legend "COMFLOW"   -> the authors' VOF computation (NOT the reference),
  * dashed line,     legend "experiment" -> the measured water height (the reference we want).
The legend identifies blue as ComFLOW and red as experiment. Use those colour
channels directly, including where the curves cross."""
import json
import sys

import numpy as np

# --- axis calibration (pixel <-> data), read from the raster tick labels ------------------
# right panel (H2): frame x 661..1179, y 5..413; x labels 0..6 s (7 ticks), y labels 0.4/0.2/0.0
X0_PX, X1_PX = 661.0, 1179.0
T0_S, T1_S = 0.0, 6.0
Y_TOP_PX, Y_BOT_PX = 6.5, 413.5      # centres of the "0.4" and "0.0" tick labels
H_TOP_M, H_BOT_M = 0.4, 0.0


def to_t(x_px):
    return T0_S + (np.asarray(x_px, float) - X0_PX) * (T1_S - T0_S) / (X1_PX - X0_PX)


def to_h(y_px):
    return H_TOP_M + (np.asarray(y_px, float) - Y_TOP_PX) * (H_BOT_M - H_TOP_M) / (Y_BOT_PX - Y_TOP_PX)


def main(jpeg, out_prefix):
    from PIL import Image
    rgb = np.asarray(Image.open(jpeg).convert("RGB"), dtype=np.int16)
    if rgb.shape != (450, 1182, 3):
        raise ValueError("expected the original Figure 19 embedded image")
    red, green, blue = rgb.transpose((2, 0, 1))
    solid, dashed = {}, {}
    for mask, trace in (((blue-red > 45) & (blue-green > 45), solid),
                        ((red-blue > 45) & (red-green > 45), dashed)):
        for x in range(int(X0_PX)+1, int(X1_PX)):
            lo = 65 if x > 1040 else 7  # exclude the legend in the upper right
            ys = np.flatnonzero(mask[lo:414, x])
            if len(ys):
                trace[x] = float(np.median(ys)+lo)
    res = {}
    for name, d in (("solid_comflow", solid), ("dashed_experiment", dashed)):
        xs = np.array(sorted(d), float)
        ys = np.array([d[k] for k in sorted(d)], float)
        res[name] = {"n_columns": int(xs.size), "columns": list(xs), "y_px": list(ys)}
        t = to_t(xs)
        h = to_h(ys)
        # peak = smallest y_px (highest water); arrival = first t with h > 0.02 m
        ip = int(np.argmin(ys))
        ihit = np.where(h > 0.02)[0]
        i_arr = int(ihit[0]) if ihit.size else None
        res[name].update(
            peak_height_m=float(h[ip]), peak_time_s=float(t[ip]),
            arrival_time_s=float(t[i_arr]) if i_arr is not None else None,
            t_min=float(t.min()), t_max=float(t.max()))
        print(f"{name:18s} cols={xs.size:4d}  peak={h[ip]:.3f} m @ t={t[ip]:.2f} s  "
              f"arrival(>2 cm)={res[name]['arrival_time_s']}")
    with open(out_prefix + ".json", "w") as f:
        json.dump(res, f, indent=1)
    measured = res["dashed_experiment"]
    np.savetxt(out_prefix + "_experiment.csv",
               np.column_stack((to_t(measured["columns"]), to_h(measured["y_px"]))),
               delimiter=",", fmt="%.7f", header="t_s,h_m", comments="")
    # overlay figure for the audit trail
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=140)
    for name, style in (("solid_comflow", dict(ls="-", lw=1.6, color="tab:orange")),
                        ("dashed_experiment", dict(ls="", marker=".", ms=1.5, color="k"))):
        r = res[name]
        ax.plot(to_t(np.array(r["columns"])), to_h(np.array(r["y_px"])), label=name, **style)
    r = res["dashed_experiment"]
    ax.plot([r["peak_time_s"]], [r["peak_height_m"]], "r*", ms=12,
            label=f"exp peak {r['peak_height_m']:.3f} m @ {r['peak_time_s']:.2f} s")
    ax.set_xlabel("t [s]"); ax.set_ylabel("water height at H2 [m]")
    ax.set_title("digitised from Kleefsman et al. (2005) Fig. 19, right panel")
    ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out_prefix + ".png")
    print("wrote", out_prefix + ".json/.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
