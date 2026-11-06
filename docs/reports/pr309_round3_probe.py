"""Round-3 probe: are PR #309's three relaxations still *justified*, not just green?

Mission m4659 (merged as #322) settled *whether* the three assertions of the
closed PR #309 hold on the then-current tree.  It left one question open, and
that question is what this probe measures: for each relaxation, is the loose
tolerance still **necessary** (a strict one would be red) and is the loose one
still **tight enough to mean something** (it is not hiding a regression)?
An audit that only shows green tests cannot answer that; it needs the failing
side of each gate as well.

Three controls, one per PR #309 file:

``A`` kernel sum (``tests/test_kernel_sum_consistency.py``)
    The merged test replaces a digit pin (``-5.32e-5``) by ``|bias| < 1e-4``.
    What backs that bound is the claim that the *fifth digit* is a property of
    the summation order, not of the lattice.  Measured here: the same lattice
    summed in float32, in float64, and over a permuted particle order -- if the
    spread is far below the gate, the gate is a lattice claim; if it were not,
    the gate would be an artefact of the reduction tree.

``B`` wall pressure (``tests/test_wall_pressure.py``)
    The merged test runs ``rtol=2e-4, atol=2e-4``.  Measured here: the same two
    solver states against the *old* ``rtol=1e-5, atol=1e-6``.  Only if the
    two-term state is red at the old tolerance and the body-force-only state is
    green is the relaxation a correction rather than a weakening.

``C`` initial density (``tests/test_initial_density_calibration.py``)
    The merged test asserts the flag defaults to True.  Measured here: the
    spurious pressure the *uncalibrated* pool would carry, so the default is
    load bearing rather than cosmetic.

Run from the repository root (the ``PYTHONPATH`` is not optional: this machine
has editable installs of other agent worktrees on ``sys.path``):

    PYTHONPATH=$PWD/src python docs/reports/pr309_round3_probe.py

Writes ``docs/reports/pr309_round3_probe.json``; the mission document and the
figure read that file, so no number in them is typed by hand.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TESTS))

from tensorsph.droplet_impact import (  # noqa: E402
    Config,
    make_particles,
    pool_lattice,
    rhs,
)
from tensorsph.initial_density import calibration_factor, lattice_bias  # noqa: E402
from tensorsph.lattice_consistency import (  # noqa: E402
    interior_mask,
    kernel_sum_factor,
)
from test_wall_pressure import SMALL, closed_form, two_particle_state  # noqa: E402

GATE = 1.0e-4          # m2971 acceptance gate on |p_false| / (rho0 c0^2)
OLD_PIN = -1.19e-7     # the value PR #309 tried to pin (db0c3265 tree)
STRICT = dict(rel_tol=1.0e-5, abs_tol=1.0e-6)    # pre-#309 wall-pressure gate
SHIPPED = dict(rel_tol=2.0e-4, abs_tol=2.0e-4)   # gate m4635 left on main
MASS_SCALES = (0.8, 3.0)


def _shipped_pool(dtype=None):
    cfg = Config(dx=0.01, h_factor=1.3, density_calibration=False)
    pos, mass = make_particles(cfg, torch.device("cpu"))
    n_pool = int(pool_lattice(cfg, torch.device("cpu")).shape[0])
    h = cfg.h_factor * cfg.dx
    if dtype is not None:
        pos, mass = pos.to(dtype), mass.to(dtype)
    return cfg, pos, mass, n_pool, h


def control_a_kernel_sum() -> dict:
    """Same lattice, four reductions: how much of the 5th digit is the sum?"""
    cfg, pos, mass, n_pool, h = _shipped_pool()
    pool_pos, pool_mass = pos[:n_pool], mass[:n_pool]
    # Permute *within the pool*: the drop particles must stay out of the prefix,
    # otherwise the interior mask is applied to the free-surface contamination
    # (the -1.09e-2 "masked trap" of docs/mission_m4659_fix_summary.md section 1.1).
    perm = torch.randperm(n_pool, generator=torch.Generator().manual_seed(4659))

    def bias(p, m, dtype, order=None):
        p, m = p.to(dtype), m.to(dtype)
        c = kernel_sum_factor(p, m / cfg.rho0, h)
        if order is not None:
            c = c[order]
            p = p[order]
        return float(c[interior_mask(p, h)].mean()) - 1.0

    variants = {
        "fp32_as_shipped": bias(pool_pos, pool_mass, torch.float32),
        "fp32_permuted": bias(pool_pos, pool_mass, torch.float32, perm),
        "fp64_cast": bias(pool_pos, pool_mass, torch.float64),
        "fp64_cast_permuted": bias(pool_pos, pool_mass, torch.float64, perm),
    }
    vals = list(variants.values())
    spread = max(vals) - min(vals)
    return {
        "dx": cfg.dx,
        "h_over_dx": cfg.h_factor,
        "n_pool": n_pool,
        "n_interior": int(interior_mask(pool_pos, h).sum()),
        "bias_by_variant": variants,
        "spread_max_minus_min": spread,
        "gate": GATE,
        "spread_over_gate": spread / GATE,
        "old_pin": OLD_PIN,
        "measured_over_old_pin": abs(variants["fp32_as_shipped"]) / abs(OLD_PIN),
        "all_same_sign": len({v < 0.0 for v in vals}) == 1,
        "all_inside_gate": all(abs(v) < GATE for v in vals),
    }


def control_b_wall_pressure() -> dict:
    """Two solver states graded by the old gate and by the shipped one."""
    rows = []
    for ms in MASS_SCALES:
        cfg = Config(**SMALL)
        pos, wall, mass = two_particle_state(cfg, ms)
        _, _, acc = rhs(pos, torch.zeros_like(pos), mass, cfg, wall=wall)
        _, expected = closed_form(cfg, ms)
        a, e = float(acc[0, 1]), float(expected[1])
        rows.append({
            "mass_scale": ms,
            "acc_rhs": a,
            "acc_closed_form": e,
            "relerr": abs(a - e) / abs(e),
            "passes_old_gate": math.isclose(a, e, **STRICT),
            "passes_shipped_gate": math.isclose(a, e, **SHIPPED),
            "head_rho_g_dx_Pa": cfg.rho0 * cfg.gravity * cfg.dx,
        })
    return {
        "old_gate": STRICT,
        "shipped_gate": SHIPPED,
        "rows": rows,
        "old_gate_red_on_two_term": not rows[1]["passes_old_gate"],
        "old_gate_green_on_body_only": rows[0]["passes_old_gate"],
        "worst_relerr": max(r["relerr"] for r in rows),
        "shipped_gate_margin": None,
    }


def control_c_initial_density() -> dict:
    """How much spurious pressure the default (calibrated) flag removes."""
    cfg, pos, mass, n_pool, h = _shipped_pool()
    mask = interior_mask(pos[:n_pool], h)
    raw = float(kernel_sum_factor(pos, mass / cfg.rho0, h)[:n_pool][mask].mean()) - 1.0
    factor = calibration_factor(cfg.dx, h, kind="square")
    calibrated = (1.0 + raw) * factor - 1.0
    gamma = cfg.gamma
    c0 = 20.0

    def p_ratio(b):
        return abs((1.0 + b) ** gamma - 1.0) / gamma

    hex_bias = lattice_bias(cfg.dx, h, kind="hex")
    return {
        "density_calibration_default": bool(Config().density_calibration),
        "gamma": gamma,
        "sound_speed": c0,
        "square": {
            "bias_raw": raw,
            "bias_calibrated": calibrated,
            "p_ratio_raw": p_ratio(raw),
            "p_ratio_calibrated": p_ratio(calibrated),
            "clamped_to_zero": raw < 0.0,
        },
        "hex_same_h": {
            "bias_raw": hex_bias,
            "p_ratio_raw": p_ratio(hex_bias),
            "clamped_to_zero": hex_bias < 0.0,
            "abs_p_false_Pa": p_ratio(hex_bias) * cfg.rho0 * c0 * c0,
        },
        "gate": GATE,
        # The shipped square pool is saved by the tension clamp (raw bias is
        # negative -> p = 0 Pa), so on *this* lattice the default costs nothing
        # measurable; the flag is load bearing for lattices whose bias is
        # positive, where the clamp cannot hide it.  Both verdicts are recorded
        # so the document cannot claim more than was measured.
        "square_gate_met_without_calibration": p_ratio(raw) < GATE,
        "square_saved_by_clamp": raw < 0.0,
        "hex_gate_met_without_calibration": p_ratio(hex_bias) < GATE,
        "default_flag_load_bearing_on_hex": p_ratio(hex_bias) >= GATE,
    }


def main() -> int:
    out = {
        "mission": "m4817",
        "lineage": "closed PR #309 -> m4659/m4793 -> this round-3 control set",
        "main": _git_head(),
        "arch": platform.machine(),
        "torch": torch.__version__,
        "default_dtype": str(torch.get_default_dtype()),
        "control_a_kernel_sum": control_a_kernel_sum(),
        "control_b_wall_pressure": control_b_wall_pressure(),
        "control_c_initial_density": control_c_initial_density(),
    }
    b = out["control_b_wall_pressure"]
    worst = b["worst_relerr"]
    b["shipped_gate_margin"] = SHIPPED["rel_tol"] / worst if worst else None
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {path}")
    return 0


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
