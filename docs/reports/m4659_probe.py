"""Re-measure the three quantities mission m4659 wanted to re-pin.

The closed PR #309 (m4659) relaxed three assertions after the db0c3265 tree went
red.  This probe measures the same three quantities on the current main tree so
the re-opened mission can decide, per assertion, whether the old relaxation is
still needed, is already in place, or would now be a *red* test.

Run from the repository root:

    PYTHONPATH=$PWD/src python docs/reports/m4659_probe.py

Writes ``docs/reports/m4659_probe.json`` (raw evidence; the summary document and
the figure are generated from it, so no number is copied by hand).
"""
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "init_consistency"))
sys.path.insert(0, str(ROOT / "tests"))

import run as R  # noqa: E402  (benchmark harness, imported as a module)
from tensorsph.droplet_impact import Config, make_particles, pool_lattice, rhs  # noqa: E402
from tensorsph.initial_density import calibration_factor  # noqa: E402
from tensorsph.lattice_consistency import interior_mask, kernel_sum_factor  # noqa: E402
from test_wall_pressure import SMALL, closed_form, two_particle_state  # noqa: E402

H_FACTORS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8)
MASS_SCALES = (0.8, 3.0)


def shipped_square_bias() -> dict:
    """C - 1 for the lattice ``make_particles`` actually ships (h = 1.3 dx)."""
    cfg = Config(dx=0.01, h_factor=1.3, density_calibration=False)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    h = cfg.h_factor * cfg.dx
    c = kernel_sum_factor(pos, mass / cfg.rho0, h)
    n_pool = int(pool_lattice(cfg, torch.device("cpu")).shape[0])
    pool = interior_mask(pos[:n_pool], h)
    return {
        "dx": cfg.dx,
        "h_over_dx": cfg.h_factor,
        "n_pool": n_pool,
        "n_pool_interior": int(pool.sum()),
        "bias_pool_interior": float(c[:n_pool][pool].mean()) - 1.0,
        "bias_whole_box": float(c[interior_mask(pos, h)].mean()) - 1.0,
    }


def lattice_sweep() -> list:
    """Raw and calibrated kernel-sum bias over the h/dx ladder, both lattices."""
    rows = []
    for kind in ("square", "hex"):
        if kind == "square":
            lat = R.square_lattice(0.20, 0.01)
        else:
            lat = R.hex_lattice(0.20, 0.01)
        for hf in H_FACTORS:
            row = R.audit_case(kind, 0.01, hf, lat)
            rows.append({
                "lattice": kind,
                "h_over_dx": hf,
                "bias_raw": row["density_bias_raw"],
                "bias_calibrated": row["density_bias_calibrated"],
                "criterion_rel_pressure": row["criterion_rel_pressure"],
            })
    return rows


def wall_pressure() -> list:
    """Two-particle Adami eq. (11) state: solver value against the closed form."""
    rows = []
    for ms in MASS_SCALES:
        cfg = Config(**SMALL)
        pos, wall, mass = two_particle_state(cfg, ms)
        _, fluid_pressure, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
        wall_p, expected = closed_form(cfg, ms)
        a = float(acc[0, 1])
        e = float(expected[1])
        rows.append({
            "mass_scale": ms,
            "fluid_pressure_Pa": float(fluid_pressure[0]),
            "wall_pressure_Pa": float(wall_p),
            "acc_rhs": a,
            "acc_closed_form": e,
            "relerr": abs(a - e) / abs(e),
            "rho_g_dx_Pa": cfg.rho0 * cfg.gravity * cfg.dx,
        })
    return rows


def initial_density() -> dict:
    """Uncalibrated mass, calibration factor, and the flag's default."""
    cfg = Config(dx=0.002, h_factor=1.3, rho0=1000.0, pool_width=0.1, pool_depth=0.05,
                 drop_radius=0.005, drop_center_y=0.1, gravity=9.81,
                 density_calibration=False)
    _, m_off = make_particles(cfg, torch.device("cpu"))
    cfg_on = Config(**{**cfg.__dict__, "calibrate_initial_mass": True})
    _, m_on = make_particles(cfg_on, torch.device("cpu"))
    factor = calibration_factor(cfg.dx, cfg.h_factor * cfg.dx, kind="square")
    return {
        "density_calibration_default": bool(Config().density_calibration),
        "n_particles": int(m_off.numel()),
        "m_off_over_rho0_dx2": float((m_off / (cfg.rho0 * cfg.dx**2))[0]),
        "m_on_over_m_off": float((m_on / m_off)[0]),
        "calibration_factor": float(factor),
        "factor_relerr": abs(float((m_on / m_off)[0]) - float(factor)) / float(factor),
    }


def main() -> int:
    out = {
        "arch": platform.machine(),
        "torch": torch.__version__,
        "dtype": str(torch.get_default_dtype()),
        "shipped_square": shipped_square_bias(),
        "sweep": lattice_sweep(),
        "wall": wall_pressure(),
        "density": initial_density(),
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
