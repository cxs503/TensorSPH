"""Run the 3-D cold-plate drop and its independent planar Stefan control."""
import argparse
import csv
import json
from pathlib import Path

import torch

from tensorsph.ice_thermal import properties as p
from tensorsph.ice_thermal.enthalpy import neumann_lambda, stefan_front, stefan_number_fresh
from tensorsph.ice_thermal.sessile_drop import simulate


def experiment_record(result):
    record = dict(result.metrics)
    end = record["last_particle_frozen_s"]
    record.update(experimental_time_s=18., source_doi="10.1063/1.4747185",
                  relative_time_error=end/18.-1 if end is not None else None)
    return record


def verify_slab(cells):
    t = 2.
    result = simulate(cells=cells, shape="slab", height=.002, end_s=t, glass_depth=0.)
    # Equal fixed mass and equal volume give the area-mean solid thickness.
    depth = (result.positions[:, 2].max().item()+.5*result.metrics["dx_m"])
    computed = depth*result.metrics["final_frozen_mass_fraction"]
    ste = stefan_number_fresh(p.T_MELT, p.T_MELT-20.)
    ref = stefan_front(t, neumann_lambda(ste), p.K_ICE/(p.RHO_FRESH*p.C_ICE))
    return dict(cells_per_radius=cells, t_s=t, front_m=computed, reference_m=ref,
                relative_error=computed/ref-1)


def figure(result, out):
    import matplotlib.pyplot as plt

    # Actual particle values in the first plane next to y=0.
    pos = result.positions[result.section_mask]*1000
    frames = result.sections
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.6), layout="constrained")
    for ax, (time, solid) in zip(axes.flat, frames):
        scatter = ax.scatter(pos[:, 0], pos[:, 2], c=solid, cmap="viridis", vmin=0, vmax=1,
                             marker="s", s=(100/result.metrics["cells_per_radius"])**2,
                             edgecolors="none")
        ax.axhline(0, color="#664c39", lw=2)
        ax.set(xlim=(-2.1, 2.1), ylim=(-.05, 1.15), aspect="equal", title=f"{time:.2f} s")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([0, 1])
    for ax in list(axes.flat)[len(frames):-1]:
        ax.set_visible(False)
    ax = axes.flat[-1]
    ax.plot([r["t_s"] for r in result.history],
            [r["frozen_mass_fraction"] for r in result.history], color="#185783", label="SPH")
    ax.scatter([18.], [1.], marker="x", color="#ae3b29", label="Photo, ≈18 s")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set(xlabel="Time (s)", ylabel="Ice mass fraction", ylim=(0, 1.08), xlim=(0, 19))
    fig.colorbar(scatter, ax=list(axes.flat)[:len(frames)], label="Local ice mass fraction",
                 shrink=.7)
    fig.suptitle("Cold-plate freezing · 3-D sessile drop · central slice (mm)")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def run(output, cells, safety=.8, plot=True, fem_reference=False):
    if sorted(set(cells)) != cells or not cells:
        raise ValueError("provide distinct resolutions in increasing order")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    finest = None
    for n in cells:
        for glass in (0., .001):
            result = simulate(cells=n, glass_depth=glass, safety=safety)
            records.append(experiment_record(result))
            if n == cells[-1] and glass > 0:
                finest = result
    control = simulate(cells=cells[0], safety=safety/2)
    anchors = [verify_slab(n) for n in cells]
    data = dict(cold_plate_runs=records, half_step=experiment_record(control),
                planar_stefan=anchors, experimental_scope="post-nucleation duration only")
    if fem_reference:
        from tensorfem.freezing_layer import freeze_layer
        for row in anchors:
            front = freeze_layer(cells=row["cells_per_radius"], depth=.002,
                                 end_s=row["t_s"]).history[-1]["front_m"]
            row.update(fem_front_m=front, fem_relative_error=front/row["reference_m"]-1,
                       sph_minus_fem_m=row["front_m"]-front)
    (output/"results.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    with (output/"history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=finest.history[0], lineterminator="\n")
        w.writeheader()
        w.writerows(finest.history)
    torch.save(dict(position_m=finest.positions, solid_mass_fraction=finest.solid_fraction,
                    specific_enthalpy_J_per_kg=finest.specific_enthalpy), output/"particles.pt")
    if plot:
        figure(finest, output/"freezing.png")
    for row in records:
        freeze = row["last_particle_frozen_s"]
        value = f"{freeze:.3f} s" if freeze is not None else "incomplete"
        print(f"R/dx={row['cells_per_radius']} slide={1000*row['glass_depth_m']:.2f} mm: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--fem-reference", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output, args.cells, plot=not args.no_plot, fem_reference=args.fem_reference)
