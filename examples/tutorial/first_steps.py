"""TensorSPH tutorial: six measured steps from a kernel to a running case.

Every step below *measures* something and asserts it, so the tutorial cannot
drift away from the code: run it and it either prints the numbers it claims or
it goes red.

    python examples/tutorial/first_steps.py                # ~2 s, CPU
    python examples/tutorial/first_steps.py --json out.json
    python examples/tutorial/first_steps.py --figure kernel.png

Each ``step_*`` function returns a ``dict`` of float measurements. The console
prints them as ``MEASURED <name>=<value>`` lines, which is where the numbers
quoted in ``docs/tutorials/01-first-steps.md`` come from (the doc keeps the
captured log next to it). ``tests/test_tutorial_first_steps.py`` runs the same
functions and pins what they return.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import torch

import tensorsph as sph

FLOAT = torch.float64


# --------------------------------------------------------------------------- #
# Step 1 -- is the kernel a normalised 2-D kernel at all?
# --------------------------------------------------------------------------- #

def step1_kernel_normalisation(dx: float = 0.02, h_factor: float = 1.3,
                               samples: int = 8001) -> dict:
    """Integrate the cubic spline over the plane in polar coordinates.

    The 2-D normalisation is ``10/(7 pi h^2)``. The only way to know it is the
    2-D one (and not, say, the 3-D one) without trusting a comment is to
    integrate: ``2 pi * int_0^{2h} W(r) r dr`` must be 1. Trapezoid on a fine
    radial grid is enough because ``W`` is piecewise cubic and vanishes smoothly
    at ``r = 2h``.
    """
    h = h_factor * dx
    r = torch.linspace(0.0, 2.0 * h, samples, dtype=FLOAT)
    kernel = sph.cubic_spline_kernel(r, h)
    # 2-D polar quadrature. trapz is (trapezoid) O(dr^2); 8001 points over 2h
    # puts dr at h/2000, so the truncation error is far below the 1e-4 gate.
    integral = float(2.0 * math.pi * torch.trapz(kernel * r, r))
    at_support = float(sph.cubic_spline_kernel(torch.tensor(2.0 * h, dtype=FLOAT), h))
    peak = float(kernel[0])
    assert abs(integral - 1.0) < 1.0e-4, f"kernel integral {integral} != 1"
    assert at_support == 0.0, f"kernel does not vanish at r=2h ({at_support})"
    # A kernel must be positive and decreasing: check the peak is the max.
    assert bool((kernel[:-1] >= kernel[1:] - 1e-14).all()), "kernel is not monotone"
    return {"kernel_integral": integral, "kernel_peak": peak,
            "kernel_at_2h": at_support, "h": h}


# --------------------------------------------------------------------------- #
# Step 2 -- density summation: the lattice bias and the calibrated mass
# --------------------------------------------------------------------------- #

def step2_density_calibration(dx: float = 0.02, h_factor: float = 1.3,
                              rows: int = 40, rho0: float = 1000.0) -> dict:
    """Build a square lattice and measure its summation density.

    ``rho_i = sum_j m_j W_ij`` on a complete lattice with ``m = rho0 dx^2``
    returns ``rho0 dx^2 S``, where ``S`` is the discrete kernel sum. ``S`` is not
    ``1/dx^2``, so the interior starts at ``rho0 (1 + b)`` with a *uniform* bias
    ``b``. ``sph.lattice_kernel_sum(dx, h)`` gives ``S``; ``m = rho0 / S`` makes
    the interior start at ``rho0`` to machine precision.

    The bias depends on the packing, not only on ``h/dx``: at ``h = 1.3 dx`` a
    square lattice is *under*-dense (``b ~ -5e-5``) while a hex lattice is
    over-dense by three orders of magnitude more (``b ~ +5.9e-3``). Calibrating
    with the other packing's factor is therefore as wrong as not calibrating,
    which is what the last measurement here shows. Note the argument order:
    ``lattice_kernel_sum(dx, h)``.
    """
    h = h_factor * dx
    coords = (torch.arange(rows, dtype=FLOAT) + 0.5) * dx
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    pairs = sph.cell_linked_pairs(pos, 2.0 * h)

    # Interior = full kernel support stays inside the lattice (support = 2h).
    low, high = coords[0], coords[-1]
    inside = ((pos[:, 0] - 2.0 * h >= low) & (pos[:, 0] + 2.0 * h <= high)
              & (pos[:, 1] - 2.0 * h >= low) & (pos[:, 1] + 2.0 * h <= high))
    assert int(inside.sum()) > 100, "lattice too small: fewer than 100 interior particles"

    lattice_sum = sph.lattice_kernel_sum(dx, h)
    plain = torch.full((pos.shape[0],), rho0 * dx * dx, dtype=FLOAT)
    calibrated = torch.full((pos.shape[0],), rho0 / lattice_sum, dtype=FLOAT)
    hex_factor = 1.0 + sph.lattice_bias(dx, h, kind="hex")
    wrong = torch.full((pos.shape[0],), rho0 * dx * dx / hex_factor, dtype=FLOAT)

    ratio_plain = sph.density_from_neighbor_pairs(pos, plain, h, pairs) / rho0
    ratio_calibrated = sph.density_from_neighbor_pairs(pos, calibrated, h, pairs) / rho0
    ratio_wrong = sph.density_from_neighbor_pairs(pos, wrong, h, pairs) / rho0

    bias = float(ratio_plain[inside].mean()) - 1.0
    spread = float((ratio_plain[inside] - ratio_plain[inside].mean()).abs().max())
    calibrated_error = float((ratio_calibrated[inside] - 1.0).abs().max())
    wrong_packing_error = float((ratio_wrong[inside] - 1.0).abs().max())
    hex_bias = sph.lattice_bias(dx, h, kind="hex")
    geometry = dx * dx * lattice_sum - 1.0
    assert spread < 1.0e-12, "the lattice bias must be uniform, not noisy"
    assert abs(bias - geometry) < 1.0e-12, "measured bias disagrees with dx^2 * S"
    # The bias is uniform; the calibrated mass removes it, the wrong packing does not.
    assert abs(bias) < 1.0e-3, f"unexpected square-lattice bias {bias}"
    assert calibrated_error < 1.0e-12, f"calibrated mass left {calibrated_error}"
    assert wrong_packing_error > 1.0e-3, (
        f"a hex calibration factor is harmless on a square lattice ({wrong_packing_error})")
    return {"lattice_kernel_sum": lattice_sum, "lattice_bias_square": bias,
            "lattice_bias_hex": hex_bias, "lattice_bias_spread": spread,
            "calibrated_error": calibrated_error,
            "wrong_packing_error": wrong_packing_error,
            "interior_particles": float(inside.sum())}


# --------------------------------------------------------------------------- #
# Step 3 -- the equation of state and the sound speed it actually carries
# --------------------------------------------------------------------------- #

def step3_equation_of_state(dx: float = 0.02, h_factor: float = 1.3,
                            rho0: float = 1000.0, sound_speed: float = 20.0,
                            gamma: float = 7.0) -> dict:
    """Recover ``c0`` from ``p(rho)`` by finite differences, and price the bias.

    ``p = B((rho/rho0)^gamma - 1)`` with ``B = rho0 c0^2 / gamma``. The nominal
    ``sound_speed`` is a *parameter*; the check is that ``sqrt(dp/drho)`` at
    ``rho0`` is the ``c0`` the solver is configured with, because that is what
    decides the Courant limit. Then the step-2 biases are turned into pressure: a
    uniform ``rho/rho0 = 1 + b`` is a uniform spurious pressure at the first
    step, and the clamp turns the square lattice's negative bias into exactly
    zero -- which is why the hex number is the one that matters.
    """
    cfg = sph.Config(rho0=rho0, sound_speed=sound_speed, gamma=gamma)
    p_at_rho0 = float(sph.tait_pressure(torch.tensor([rho0], dtype=FLOAT), cfg)[0])
    eps = 1.0e-5
    # Careful: a *central* difference around rho0 straddles the clamp. The
    # negative side is not -p but exactly 0, so the naive slope comes out at
    # c0/sqrt(2) -- half the square. The one-sided difference from rho0 is the
    # honest measurement, first order with error (gamma-1) eps / 2 ~ 3e-5.
    p_up = float(sph.tait_pressure(torch.tensor([rho0 * (1.0 + eps)], dtype=FLOAT), cfg)[0])
    slope = (p_up - p_at_rho0) / (eps * rho0)
    recovered = math.sqrt(slope)
    # Away from rho0 both sides are compressive, so a central difference works
    # there: dp/drho = c0^2 (rho/rho0)^(gamma-1), i.e. c = c0 (rho/rho0)^3.
    ratio = 1.01
    p_plus = float(sph.tait_pressure(
        torch.tensor([rho0 * ratio * (1.0 + eps)], dtype=FLOAT), cfg)[0])
    p_minus = float(sph.tait_pressure(
        torch.tensor([rho0 * ratio * (1.0 - eps)], dtype=FLOAT), cfg)[0])
    central = math.sqrt((p_plus - p_minus) / (2.0 * eps * rho0 * ratio))
    expected = sound_speed * ratio ** 3
    # The free-surface clamp: rho < rho0 must never produce tensile pressure.
    tensile = float(sph.tait_pressure(torch.tensor([0.9 * rho0, 0.5 * rho0], dtype=FLOAT),
                                     cfg).max())
    h = h_factor * dx
    b_square = sph.lattice_bias(dx, h)
    b_hex = sph.lattice_bias(dx, h, kind="hex")
    pressures = {}
    for label, bias in (("square", b_square), ("hex", b_hex)):
        rho = torch.tensor([rho0 * (1.0 + bias)], dtype=FLOAT)
        pressures[label] = float(sph.tait_pressure(rho, cfg)[0])
    bulk = rho0 * sound_speed ** 2 / gamma
    assert p_at_rho0 == 0.0, f"p(rho0) = {p_at_rho0}, expected 0"
    assert abs(recovered / sound_speed - 1.0) < 1.0e-4, (
        f"EOS carries c = {recovered}, configured {sound_speed}")
    assert abs(central / expected - 1.0) < 1.0e-4, (
        f"compressed EOS carries {central}, expected {expected}")
    assert tensile == 0.0, f"clamp failed: compressive-only EOS returned {tensile} Pa"
    assert pressures["hex"] > 1.0e3, (
        f"hex bias pressure {pressures['hex']} Pa is small enough to ignore")
    assert pressures["square"] == 0.0, "the negative square bias should clamp to zero"
    return {"bulk_modulus": bulk, "p_at_rho0": p_at_rho0,
            "recovered_sound_speed": recovered,
            "sound_speed_error": recovered / sound_speed - 1.0,
            "compressed_sound_speed": central,
            "compressed_sound_speed_expected": expected,
            "bias_pressure_square": pressures["square"],
            "bias_pressure_hex": pressures["hex"],
            "tensile_pressure": tensile}


# --------------------------------------------------------------------------- #
# Step 4 -- neighbour search: the fast path against the reference
# --------------------------------------------------------------------------- #

def step4_neighbour_search(dx: float = 0.02, h_factor: float = 1.3,
                           count: int = 2000, seed: int = 20261105) -> dict:
    """``cell_linked_pairs`` must return exactly what brute force returns.

    An approximate neighbour search silently changes the physics, so the check
    is equality of the pair lists, not "close enough". The Verlet skin is the
    same claim one level up: pairs found at ``cutoff + skin`` are exact at
    ``cutoff`` as long as nobody moved more than ``skin/2``.
    """
    h = h_factor * dx
    cutoff = 2.0 * h
    generator = torch.Generator().manual_seed(seed)
    pos = torch.rand((count, 2), generator=generator, dtype=FLOAT)
    reference = sph.brute_force_pairs(pos, cutoff)
    fast = sph.cell_linked_pairs(pos, cutoff)
    identical = torch.equal(fast, reference)

    cache = sph.PairListCache(cutoff, 0.25 * h, fluid_count=count)
    mass = torch.full((count,), 1.0 / count, dtype=FLOAT)
    pairs0, _ = cache.get(pos)
    rho_fresh = sph.density_from_neighbor_pairs(pos, mass, h, pairs0)
    # Displace every particle by less than half the skin (0.125 h), then reuse.
    shifted = pos.clone()
    shifted[:, 0] += 0.1 * h
    pairs1, rebuilt = cache.get(shifted)
    rho_reused = sph.density_from_neighbor_pairs(shifted, mass, h, pairs1)
    rho_reference = sph.density_from_neighbor_pairs(shifted, mass, h)
    reuse_error = float((rho_reused - rho_reference).abs().max())

    assert identical, "cell-linked search disagrees with brute force"
    assert not rebuilt, "the cache rebuilt even though nobody moved skin/2"
    assert cache.reuses == 1, f"expected one reuse, got {cache.reuses}"
    assert reuse_error < 1.0e-15, f"skin reuse changed the density by {reuse_error}"
    return {"pair_count_brute_force": float(reference.shape[0]),
            "pair_count_cell_linked": float(fast.shape[0]),
            "pairs_identical": 1.0 if identical else 0.0,
            "skin_reuse_density_error": reuse_error,
            "rho_first_step_max": float(rho_fresh.max())}


# --------------------------------------------------------------------------- #
# Step 5 -- which limit actually binds the step?
# --------------------------------------------------------------------------- #

def step5_time_step_policy(dx: float = 0.02, h_factor: float = 1.3,
                           sound_speed: float = 20.0, viscosity: float = 0.01,
                           gravity: float = 9.81) -> dict:
    """Four limits, one ``min``, and a configuration that is refused.

    ``stable_dt`` is ``min(acoustic, viscous, gravity, diffusion, capillary)``.
    At ``dx = 0.02`` the acoustic limit binds by an order of magnitude, which is
    why WCSPH is expensive. A too-large ``dt`` must raise, and the message has
    to name the binding limit and the largest allowed step -- a silent clamp
    would let a user believe the requested step was honoured.
    """
    cfg = sph.Config(dx=dx, h_factor=h_factor, sound_speed=sound_speed,
                     viscosity=viscosity, gravity=gravity)
    h = cfg.h_factor * cfg.dx
    acoustic = sph.acoustic_dt_limit(cfg)
    viscous = sph.viscous_dt_limit(cfg)
    gravity_limit = sph.gravity_dt_limit(cfg)
    stable = sph.stable_dt(cfg)
    assert abs(stable - min(acoustic, viscous, gravity_limit)) < 1e-18, (
        "stable_dt is not the min of the limits it reports")

    # A moving flow: the acoustic signal speed is c0 + |v|.
    fast = 10.0
    moving = sph.stable_dt(cfg, max_speed=fast)
    expected_ratio = sound_speed / (sound_speed + fast)

    refused = ""
    too_big = sph.Config(dx=dx, h_factor=h_factor, sound_speed=sound_speed,
                         viscosity=viscosity, gravity=gravity,
                         dt=2.0 * acoustic)
    try:
        sph.validate_config(too_big)
    except ValueError as error:
        refused = str(error)
    assert refused, "validate_config accepted an unstable dt"
    assert "acoustic Courant" in refused, f"message does not name the limit: {refused}"
    assert "use dt <=" in refused, f"message does not offer a legal dt: {refused}"

    assert stable < viscous and stable < gravity_limit, (
        "the acoustic limit no longer binds at this resolution")
    assert abs(moving / stable - expected_ratio) < 1e-12, "c0+|v| not applied"
    return {"h": h, "acoustic_dt_limit": acoustic, "viscous_dt_limit": viscous,
            "gravity_dt_limit": gravity_limit, "stable_dt": stable,
            "moving_dt_limit": moving, "moving_to_stable_ratio": moving / stable,
            "refused_message": refused}


# --------------------------------------------------------------------------- #
# Step 6 -- run the case and read what it recorded
# --------------------------------------------------------------------------- #

def step6_run_case(dx: float = 0.05, steps: int = 40, output_every: int = 10,
                   out_dir: str | Path | None = None) -> dict:
    """Run 40 steps of the impact case and audit the manifest and the CSV.

    This is deliberately short: it is not a physics result. What it *does*
    establish is that the run end to end works (artifacts, manifest, acceptance
    criteria), that ``mass_total`` is constant (summation density is a function
    of position, mass is a constant of the motion here), and that the reported
    acceptance numbers come from the diagnostics rather than from the code
    quoting itself.
    """
    cfg = sph.Config(dx=dx, steps=steps, output_every=output_every, device="cpu")
    temporary = None
    if out_dir is None:
        temporary = tempfile.TemporaryDirectory()
        out_dir = temporary.name
    out = Path(out_dir)
    sph.run(cfg, out, plot=False)

    manifest = json.loads((out / "run_manifest.json").read_text())
    with (out / "diagnostics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    time_end = float(rows[-1]["time"]) * 1.0
    mass_first = float(rows[0]["mass_total"])
    mass_last = float(rows[-1]["mass_total"])
    mass_drift = abs(mass_last - mass_first) / mass_first
    speed_end = float(rows[-1]["max_speed"])
    free_fall = math.sqrt(2.0 * (cfg.drop_center_y - cfg.drop_radius
                                 - cfg.pool_depth) / cfg.gravity)
    criteria = {item["id"]: item for item in manifest["acceptance"]["criteria"]}

    assert manifest["statuses"]["software_execution"] == "success", manifest["failure"]
    assert manifest["statuses"]["numerical_verification"] == "pass", (
        manifest["acceptance"]["criteria"])
    assert len(rows) == steps // output_every + 1, f"{len(rows)} diagnostic rows"
    assert mass_drift < 1e-12, f"total mass drifted by {mass_drift}"
    # Free-surface particles under-count their neighbours, so rho_min < rho0 is
    # expected; rho <= 0 would mean a particle lost every neighbour.
    assert 0.0 < float(rows[-1]["rho_min"]) < 1000.0, "implausible rho_min"
    # The drop is in free fall at this point, so the plebs' speed is g t.
    assert abs(speed_end - cfg.gravity * time_end) / (cfg.gravity * time_end) < 0.05, (
        f"free-fall speed {speed_end} vs g t = {cfg.gravity * time_end}")
    assert criteria["acoustic_courant"]["result"] == "pass", criteria["acoustic_courant"]
    if temporary is not None:
        temporary.cleanup()
    return {"steps": steps, "time_end": time_end,
            "particle_count": float(manifest["initial_particle_count"]),
            "wall_particle_count": float(manifest["wall_particle_count"]),
            "mass_total": mass_first, "mass_drift_relative": mass_drift,
            "max_speed_end": speed_end, "free_fall_time": free_fall,
            "peak_density_compression": criteria["density_compression"]["measured"],
            "peak_courant": criteria["acoustic_courant"]["measured"],
            "peak_mach": criteria["mach_number"]["measured"],
            "numerical_verification": manifest["statuses"]["numerical_verification"]}


STEPS = (
    ("step1_kernel_normalisation", step1_kernel_normalisation),
    ("step2_density_calibration", step2_density_calibration),
    ("step3_equation_of_state", step3_equation_of_state),
    ("step4_neighbour_search", step4_neighbour_search),
    ("step5_time_step_policy", step5_time_step_policy),
    ("step6_run_case", step6_run_case),
)


def run_all() -> dict:
    """Run the six steps in order and return ``{step_name: measurements}``."""
    results = {}
    for name, function in STEPS:
        measurements = function()
        results[name] = measurements
        for key, value in measurements.items():
            shown = value if isinstance(value, str) else f"{value:.12g}"
            print(f"MEASURED {name}.{key}={shown}")
    return results


def _figure(path: str | Path) -> None:
    """Optional teaching figure: kernel shape, density bias, and the step limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx, h_factor = 0.02, 1.3
    h = h_factor * dx
    r = torch.linspace(0.0, 2.2 * h, 400, dtype=FLOAT)
    kernel = sph.cubic_spline_kernel(r, h)

    coords = (torch.arange(40, dtype=FLOAT) + 0.5) * dx
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    pairs = sph.cell_linked_pairs(pos, 2.0 * h)
    lattice_sum = sph.lattice_kernel_sum(dx, h)
    plain = sph.density_from_neighbor_pairs(
        pos, torch.full((pos.shape[0],), 1000.0 * dx * dx, dtype=FLOAT), h, pairs)
    calibrated = sph.density_from_neighbor_pairs(
        pos, torch.full((pos.shape[0],), 1000.0 / lattice_sum, dtype=FLOAT), h, pairs)
    x = pos[:, 0].reshape(40, 40)
    bias = (plain / 1000.0 - 1.0).reshape(40, 40)
    interior = ((pos[:, 0] - 2.0 * h >= coords[0]) & (pos[:, 0] + 2.0 * h <= coords[-1])
                & (pos[:, 1] - 2.0 * h >= coords[0]) & (pos[:, 1] + 2.0 * h <= coords[-1]))
    resid = (calibrated[interior] / 1000.0 - 1.0).abs()

    cfg = sph.Config(dx=dx, h_factor=h_factor)
    limits = {"acoustic": sph.acoustic_dt_limit(cfg), "viscous": sph.viscous_dt_limit(cfg),
              "gravity": sph.gravity_dt_limit(cfg)}

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].plot(r.numpy(), kernel.numpy(), lw=2)
    axes[0].axvline(2.0 * h, ls=":", c="k")
    axes[0].set_title("cubic spline W(r), 2-D")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel("W [1/m$^2$]")
    image = axes[1].pcolormesh(x.numpy(), x.numpy(), bias.numpy(), shading="auto")
    figure.colorbar(image, ax=axes[1], label="rho/rho0 - 1")
    axes[1].set_title(f"bias b = {float(bias[20, 20]):.2e} (interior)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[2].semilogy([0, 1], [1, 1], alpha=0.0)
    axes[2].bar(list(limits), list(limits.values()), color="#4c72b0")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("dt limit [s]")
    axes[2].set_title("step-size limits at dx = 0.02 m")
    for index, (label, value) in enumerate(limits.items()):
        axes[2].text(index, value * 1.3, f"{value:.2e}", ha="center", fontsize=8)
    assert float(resid.max()) < 1e-12, "calibrated figure disagrees with step 2"
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None,
                        help="write every measurement to this JSON file")
    parser.add_argument("--figure", type=Path, default=None,
                        help="write a teaching figure to this PNG path")
    arguments = parser.parse_args(argv)
    torch.manual_seed(0)
    results = run_all()
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"wrote {arguments.json}")
    if arguments.figure is not None:
        _figure(arguments.figure)
    print("all tutorial assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
