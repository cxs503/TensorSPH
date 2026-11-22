"""Compare co-located point and fixed circular measurements in the first MARIN impact."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

from compare import PEAK_END, PEAK_START, TAPS, XLS_SHA, experiment, read_series, rel, tap_event

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kleefsman_obstacle"))
from convergence_analysis import observed_order  # noqa: E402


def refinement(runs):
    """Screen a constant-Courant, three-grid ladder for diminishing signed changes."""
    levels = sorted(runs, key=lambda r: -r["dx_m"])
    dx = [r["dx_m"] for r in levels]
    if len(set(dx)) != 3 or len(levels) != 3:
        return None
    ratio = dx[0] / dx[1]
    if not math.isclose(ratio, dx[1] / dx[2]) or any(
        r["sound_speed_m_per_s"] != levels[0]["sound_speed_m_per_s"]
        or not math.isclose(r["dt_s"] / r["dx_m"], levels[0]["dt_s"] / dx[0])
        for r in levels
    ):
        return None
    trends = {}
    for i in range(1, 5):
        tap = f"P{i}"
        trends[tap] = {}
        for key in ("peak_Pa", "impulse_Pa_s"):
            values = [r["taps"][tap]["disk"][key] for r in levels]
            d1, d2 = np.diff(values)
            order = observed_order(*values, ratio)
            if d1 == 0 or d2 == 0:
                trend = "stationary difference"
            elif d1 * d2 < 0:
                trend = "oscillating"
            elif abs(d2) >= abs(d1):
                trend = "nonshrinking"
            else:
                trend = "monotone shrinking"
            trends[tap][key] = dict(
                changes=[d1, d2],
                trend=trend,
                apparent_order=order if np.isfinite(order) and order > 0 else None,
            )
    return dict(dx_m=dx, refinement_ratio=ratio, taps=trends)


def check_impact_clock(t, meta):
    """Require every fixed-step sample, including after a checkpoint restart."""
    if meta["output_every"] != 1 or len(t) != meta["steps"] + 1:
        raise ValueError("impact peaks require pressure output at every solver step")
    # The production CSV serializes timestamps to six decimal places.
    tolerance = 0.5001e-6
    if not np.allclose(t, np.arange(len(t)) * meta["dt"], rtol=0, atol=tolerance) or not (
        math.isclose(t[-1], meta["simulated_time_s"], rel_tol=0, abs_tol=tolerance)
        and t[-1] >= PEAK_END
    ):
        raise ValueError("pressure trace is inconsistent with its fixed-step event clock")


def evaluate(paths, observed, out):
    measured = {key: tap_event(observed["t_s"], observed[key]) for key in TAPS}
    runs = []
    protocol = None
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for panel, key in zip(ax[0], TAPS[:2]):
        panel.plot(
            observed["t_s"],
            (observed[key] - measured[key]["baseline_Pa"]) / 1000,
            color="black",
            label="MARIN",
            linewidth=1.5,
        )
        panel.set(
            xlim=(PEAK_START, PEAK_END),
            xlabel="Time (s)",
            ylabel="Gauge pressure (kPa)",
            title=key.split("_")[0],
        )
    for path in paths:
        tag = path.name.split("_time_series")[0]
        meta = json.loads((path.parent / f"{tag}_meta.json").read_text())
        if not meta["complete"] or meta["pressure_disk_diameter_m"] <= 0:
            raise ValueError(f"expected a complete finite-footprint run: {tag}")
        settings = {
            key: meta[key]
            for key in (
                "pressure_disk_diameter_m",
                "viscosity",
                "obstacle_x",
                "obstacle_z",
                "wall_layers",
                "wall_operator",
            )
        }
        if protocol is not None and settings != protocol:
            raise ValueError("incompatible sensor footprint, fluid or wall geometry")
        protocol = settings
        status = json.loads((path.parent / f"{tag}.json").read_text())["run"]
        data = read_series(path)
        t = data["t_s"]
        check_impact_clock(t, meta)
        taps = {}
        for key in TAPS:
            ident = key.split("_")[0]
            point = tap_event(t, data[key])
            disk = tap_event(t, data[ident + "_disk_Pa"])
            idx = int(np.argmin(np.abs(t - disk["peak_t_s"])))
            reference = measured[key]
            errors = {
                name: rel(disk[name], reference[name])
                for name in ("peak_Pa", "impulse_Pa_s")
                if key in TAPS[:4]
            }
            taps[ident] = dict(
                point=point,
                disk=disk,
                relative_error_pct=errors,
                minimum_node_support_at_peak=int(data[ident + "_disk_support_min"][idx]),
            )
        impulse = {}
        for kind in ("sph", "clamp"):
            j = data[f"J_{kind}_x_Ns"]
            impulse[kind] = float(np.interp(PEAK_END, t, j) - np.interp(PEAK_START, t, j))
        runs.append(
            dict(
                tag=tag,
                dx_m=meta["dx"],
                diameter_m=meta["pressure_disk_diameter_m"],
                sound_speed_m_per_s=meta["sound_speed"],
                body_impulse_x_Ns=impulse,
                max_compression=status["max_density_compression"],
                dt_s=meta["dt"],
                samples=len(t),
                fluid_particles=status["fluid_particles"],
                initial_cell_volume_m3=status["fluid_particles"] * meta["dx"] ** 3,
                obstacle_lattice_width_m=status["obstacle_width_on_lattice_m"],
                taps=taps,
            )
        )
        label = f"{meta['dx'] * 1000:g} mm, c={meta['sound_speed']:g} m/s"
        for panel, key in zip(ax[0], TAPS[:2]):
            ident = key.split("_")[0]
            panel.plot(
                t,
                (data[ident + "_disk_Pa"] - taps[ident]["disk"]["baseline_Pa"]) / 1000,
                label=label,
                lw=0.9,
            )
        for panel, name in zip(ax[1], ("peak_Pa", "impulse_Pa_s")):
            panel.plot(
                range(1, 5),
                [taps[f"P{i}"]["relative_error_pct"][name] for i in range(1, 5)],
                "o-",
                label=label,
            )
    for panel, title in zip(ax[1], ("Peak pressure", "Pressure impulse")):
        panel.axhline(0, c="k", lw=0.7)
        panel.set(
            xticks=[1, 2, 3, 4], xlabel="Front-face pressure tap", ylabel="Error (%)", title=title
        )
    for panel in ax.flat:
        panel.legend(fontsize=8)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pressure_footprint.png", dpi=145)
    plt.close(fig)
    report = dict(
        workbook_sha256=XLS_SHA,
        protocol=protocol,
        window_s=[PEAK_START, PEAK_END],
        experimental=measured,
        refinement=refinement(runs),
        runs=runs,
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xls", type=Path, required=True)
    ap.add_argument("--series", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    evaluate(args.series, experiment(args.xls), args.out)
