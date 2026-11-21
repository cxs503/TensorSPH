"""Independent temperature and contact-resistance sweeps with no fit to freeze time."""

import argparse
import json
from pathlib import Path

import torch

from tensorsph.ice_thermal.sessile_drop import simulate


def calculate(cells):
    rows = []
    for n in cells:
        for temp, resistance in (
            (-20.0, 0.0),
            (-15.0, 0.0),
            (-25.0, 0.0),
            (-20.0, 0.0001),
            (-20.0, 0.0003),
            (-20.0, 0.001),
        ):
            row = simulate(cells=n, plate_c=temp, contact_resistance=resistance, end_s=40.0).metrics
            rows.append(row)
    return {
        "experimental_post_nucleation_s": 18.0,
        "source_doi": "10.1063/1.4747185",
        "glass_depth_m": 0.001,
        "runs": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 12])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calculate(args.cells), indent=2, allow_nan=False) + "\n")
