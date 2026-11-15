"""Lake at rest over a *non-flat* bed: does the flat-bottom certification transfer?

Case-pool anchor ``CASE-SPH-LAKE-AT-REST`` (still water, no forcing beyond
gravity, ``u = 0`` and ``eta = const`` as the analytic reference) is certified in
:mod:`tensorsph.lake_at_rest` on a **flat** bottom: laterally periodic tank,
``max|a|/g = 6.8e-13``.  The wall-bounded tank is the documented honest FAIL.

Both of those configurations share one property that is *not* part of the anchor's
statement: **the bottom is a plane**.  A well-balanced scheme is supposed to stay
at rest over *any* bathymetry -- that is the whole point of the hydrostatic
reconstruction of Xing & Shu (2005) / Audusse et al. (2004), and it is the reason
"lake at rest over a wavy bed" is the standard stress test for a
free-surface solver, not the flat-bottom one.  On a plane bottom the kernel
truncation is the same in every column, so a mass field that is *independent of
``x``* can absorb it; on a curved bed it cannot be, because the near-bed support
fraction, the bed-normal direction and the fluid-bed clearance all vary with
``x``.

This module answers the transferability question with measurements, and keeps the
oracle honest:

* the analytic reference is still exact (``p = rho0 g (eta - y)`` does not depend
  on the bed: in still water ``grad p`` is vertical, so a slope exerts no
  tangential driving force), hence the anchor tolerance (machine precision) is
  meaningful here too;
* the *same* shipped operator (:func:`tensorsph.droplet_impact.rhs`) is used, the
  *same* reconstruction (:func:`tensorsph.lake_at_rest.reconstruct`), the *same*
  analytic prior -- nothing is case-specific except the geometry;
* the unknown map of the reconstruction is exposed as a knob, because that is
  what the question is about:

  ``tie="row"``
      one mass per ``y`` row -- the translation-invariant statement that
      certified the flat periodic tank.  If the anchor's certification were a
      property of the operator, this map would keep the lake at rest on a curved
      bed as well.
  ``tie="particle"``
      one mass per particle -- the discrete equilibrium is *sought* rather than
      inherited.  Achieving roundoff here is not a proof of well-balancedness: it
      is a fit whose mass modulation is the measure of how much the flat-bottom
      mass field has to be deformed to buy the rest state.  Both numbers are
      reported, and neither is sold as the other.

Bed geometry
------------
Bottom elevation (``y`` axis up, still-water surface at ``y = H``)::

    b(x) = -(A/2) (1 - cos(2 pi n x / W))          b(x) = BED_AMPLITUDE * shape

zero at the periodic seams ``x = 0, W`` (so ``A = 0`` reduces the geometry to the
flat periodic tank *through this code path*, which is the regression control
``B1``), troughs of depth ``A`` at ``x = W (2k+1) / (2n)``, and maximum bed slope
``A pi n / W``.  Wall particles are three layers of the bed lattice offset along
the *bed normal* (not downwards), so the near-bed support fraction is the physical
one instead of an artefact of a staircase; the fluid lattice keeps the free
surface flat (rows counted down from ``y = H``) and drops the rows that come
within ``clearance * dx`` of the bed.  A lattice cannot be flat at the surface and
aligned at a curved bed at the same time: the free-surface identity ``eta =
const`` is the anchor's oracle, so the staircase is taken on the bed side.

Nothing below is random or fitted: every number is a function of ``W``, ``H``,
``A``, ``n``, ``dx``, ``c0`` and the library's kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .droplet_impact import lattice_mass_unit
from .lake_at_rest import (LakeConfig, analytic_profile, density_operator,
                           integrate_at_rest, periodic_pairs, reconstruct,
                           residual_report)

__all__ = [
    "BedConfig",
    "bed_height",
    "bed_slope",
    "max_bed_slope",
    "fluid_lattice",
    "bed_wall_particles",
    "bed_state",
    "tie_map",
    "run_case",
    "sweep",
    "GATES",
    "evaluate_gates",
]


@dataclass
class BedConfig(LakeConfig):
    """A lake-at-rest configuration whose bottom is a periodic cosine ripple.

    ``bed_amplitude = 0`` is the flat bottom (``B1`` control); ``bed_waves`` is
    the number of cosine periods over the tank width and must be an integer for
    the seam to close under ``wrap_width = W``.  ``clearance`` is the minimum
    fluid-bed gap in units of ``dx``; it is a discretisation choice, reported with
    the results because it sets how much of the support is cut by the bed.
    """

    bed_amplitude: float = 0.02
    bed_waves: int = 1
    clearance: float = 0.5
    periodic_x: bool = True      # the wavy-bed case is the periodic channel
    depth: float = 0.30
    width: float = 0.40
    dx: float = 0.02

    def wave_number(self) -> float:
        k = 2.0 * math.pi * self.bed_waves / self.width
        if abs(k * self.width / (2.0 * math.pi) - self.bed_waves) > 1e-12:
            raise ValueError("bed_waves must be an integer for a periodic bed")
        return k


def bed_height(x, cfg: BedConfig):
    """Bed elevation ``b(x)``; ``b = 0`` at the seams, ``b = -A`` at the troughs."""
    k = cfg.wave_number()
    return -0.5 * cfg.bed_amplitude * (1.0 - torch.cos(k * x))


def bed_slope(x, cfg: BedConfig):
    """``db/dx`` (analytic; the wall lattice follows the normal, not ``-y``)."""
    k = cfg.wave_number()
    return -0.5 * cfg.bed_amplitude * k * torch.sin(k * x)


def max_bed_slope(cfg: BedConfig) -> float:
    return 0.5 * cfg.bed_amplitude * cfg.wave_number()


def _columns(cfg: BedConfig):
    count = int(round(cfg.width / cfg.dx))
    if abs(count * cfg.dx - cfg.width) > 1e-9:
        raise ValueError("width / dx must be an integer")
    k = torch.arange(count, dtype=torch.get_default_dtype())
    return (k + 0.5) * cfg.dx


def fluid_lattice(cfg: BedConfig):
    """Flat free surface, bed-side staircase, ``clearance`` enforced below.

    Rows are anchored to the **crest** of the bed (``y = (l + 1/2) dx``) and a
    particle is kept when its centre is at least ``clearance * dx`` above the
    local bed.  Dropping rows happens at the *bottom*, so the free surface stays
    at the same height in every column -- the anchor's oracle is ``eta = const``
    and a staircase surface would be indistinguishable from a spurious wave.  At
    ``bed_amplitude = 0`` this reduces *exactly* to the shipped flat periodic
    lattice (``tensorsph.lake_at_rest.build_state``), which is what makes the
    ``B1`` regression control an identity instead of a comparison.
    """
    xs = _columns(cfg)
    b = bed_height(xs, cfg)
    levels = int(math.floor(cfg.depth / cfg.dx + 1.0e-9))
    keep_x, keep_y = [], []
    for level in range(levels):
        y = (level + 0.5) * cfg.dx
        mask = (y - b) >= cfg.clearance * cfg.dx
        if not bool(mask.any()):
            continue
        keep_x.append(xs[mask])
        keep_y.append(torch.full_like(xs[mask], y))
    if not keep_x:
        raise ValueError("no particle clears the bed: reduce bed_amplitude")
    return torch.stack((torch.cat(keep_x), torch.cat(keep_y)), dim=-1)


def bed_wall_particles(cfg: BedConfig, device=None):
    """``wall_layers`` lattice layers offset along the outward bed normal.

    The shipped flat-wall construction offsets the layers straight down
    (``y = 0.5 dx - k dx``).  On a sloped bed that is the wrong direction: the
    support would be cut anisotropically and the truncation would depend on the
    offset direction instead of on the geometry.  Here the offset is
    ``-(k - 1/2) dx * n_hat`` with ``n_hat = (b', -1) / sqrt(1 + b'^2)``, i.e. the
    lattice is the bed contour shifted into the solid at spacing ``dx``.
    """
    dtype = torch.get_default_dtype()
    xs = _columns(cfg)
    b = bed_height(xs, cfg)
    s = bed_slope(xs, cfg)
    norm = torch.sqrt(1.0 + s.square())
    parts = []
    for layer in range(1, cfg.wall_layers + 1):
        offset = (layer - 0.5) * cfg.dx
        wx = xs + offset * s / norm
        wy = b - offset / norm
        parts.append(torch.stack((wx, wy), dim=-1).to(dtype))
    wall = torch.cat(parts, dim=0)
    if device is not None:
        wall = wall.to(device)
    return wall


def tie_map(pos: torch.Tensor, cfg: BedConfig, tie: str) -> torch.Tensor:
    """Particle -> independent unknown index of the reconstruction.

    ``"row"`` ties every particle of a lattice row (the flat-bed statement:
    the equilibrium mass field does not depend on ``x``); ``"column"`` ties every
    particle of a column (duplicated masses down a column -- the deliberately
    wrong map, kept because it is the cheapest way to show the tie matters at
    all); ``"particle"`` leaves every particle free.
    """
    n = pos.shape[0]
    if tie == "particle":
        return torch.arange(n, dtype=torch.long, device=pos.device)
    coordinate = pos[:, 1] if tie == "row" else pos[:, 0]
    levels = torch.unique(coordinate.round(decimals=12))
    distance = (coordinate.round(decimals=12)[:, None] - levels[None, :]).abs()
    index = distance.argmin(dim=1)
    if float(distance.gather(1, index[:, None]).max()) > 1e-12:
        raise ValueError(f"particles do not sit on lattice {tie}s")
    return index


def bed_state(cfg: BedConfig, tie: str = "row"):
    """Lattice, bed wall and the frozen periodic pair list for one tie map."""
    shipped = cfg.shipped()
    device = torch.device(cfg.device)
    pos = fluid_lattice(cfg).to(cfg.torch_dtype).round(decimals=12).to(device)
    wall = (bed_wall_particles(cfg).to(cfg.torch_dtype).round(decimals=12)
            .to(device))
    pairs = periodic_pairs(pos, wall, cfg.h(), cfg.width)
    b = bed_height(pos[:, 0], cfg)
    gap = pos[:, 1] - b
    return {
        "config": shipped,
        "pos": pos,
        "wall": wall,
        "pairs": pairs,
        "m_unit": float(lattice_mass_unit(shipped)),
        "n_fluid": int(pos.shape[0]),
        "n_wall": int(wall.shape[0]),
        "columns": tie_map(pos, cfg, tie),
        "tie": tie,
        "wrap_width": cfg.width,
        "bed": b,
        "clearance": gap,
    }


def _mass_modulation(state, cfg: BedConfig, mass: torch.Tensor,
                     prior: torch.Tensor) -> dict:
    ratio = mass / prior
    return {
        "mass_over_prior_min": float(ratio.min()),
        "mass_over_prior_max": float(ratio.max()),
        "mass_modulation_max_abs": float((ratio - 1.0).abs().max()),
        "mass_over_unit_min": float((mass / state["m_unit"]).min()),
        "mass_over_unit_max": float((mass / state["m_unit"]).max()),
        # how far the equilibrium mass field is from being a function of y only:
        # the row-tie residual of the same field, in mass units
        "mass_row_scatter": float(
            (mass - torch.stack([mass[(state["pos"][:, 1]
                                       - y).abs() < 1e-12].mean()
                                 for y in torch.unique(
                                     state["pos"][:, 1].round(decimals=12))]
             )[tie_map(state["pos"], cfg, "row")]).abs().max()),
    }


def _rest(state, cfg: BedConfig, mass: torch.Tensor, t_end) -> dict:
    run = integrate_at_rest(state, cfg, mass, t_end=t_end)
    return {k: run[k] for k in
            ("max_v_over_c0", "max_dy_over_H", "max_drho_over_rho0",
             "dt", "steps", "t_end")}


def _summary(report: dict, cfg: BedConfig) -> dict:
    """The three residual norms of a state, kept apart on purpose.

    ``residual_report`` names its vertical maximum ``a_max_over_g``; on a curved
    bed the horizontal one is no longer negligible by symmetry, so a single
    number would hide the component that matters.
    """
    acc = report["acceleration"]
    g = cfg.gravity
    return {
        "a_y_over_g": report["a_max_over_g"],
        "a_x_over_g": report["a_max_over_g_horizontal"],
        "a_norm_over_g": float(acc.norm(dim=1).max()) / g,
        "p_max": report["p_max"],
        "rho_over_rho0_min": report["rho_over_rho0_min"],
        "rho_over_rho0_max": report["rho_over_rho0_max"],
    }


def _band_residual(report: dict, state, cfg: BedConfig) -> dict:
    """Residual split by height above the local bed (in ``dx``).

    The curved-bed truncation lives in the two layers that touch the bed, so the
    number that means something is the near-bed one; the interior must stay
    small, otherwise the case has a different problem.
    """
    acc = report["acceleration"]
    clearance = state["clearance"] / cfg.dx
    bands = {"near_bed": clearance < 2.0,
             "interior": clearance >= 2.0}
    out = {}
    for name, mask in bands.items():
        if bool(mask.any()):
            out[name] = {
                "particles": int(mask.sum()),
                "a_y_over_g": float(acc[mask, 1].abs().max()) / cfg.gravity,
                "a_x_over_g": float(acc[mask, 0].abs().max()) / cfg.gravity,
            }
    return out


def run_case(cfg: BedConfig, ties=("row", "particle"), t_end=None,
             verbose: bool = False) -> dict:
    """One bed amplitude: geometry, fitting-free controls, both tie maps, rest.

    Two families of numbers, never mixed:

    * **controls** -- the shipped operator driven by a mass field that comes from
      continuum hydrostatics, with *no* equilibrium fitting: the uniform lattice
      mass, the analytic prior (summation density exactly the EOS hydrostatic
      profile), and each one integrated forward for ``t_end`` to read the
      spurious current it generates.  These are the geometry-transferability
      measurements; nothing in them can be tuned.
    * **solutions** -- the mass field that :func:`tensorsph.lake_at_rest.reconstruct`
      finds for a chosen unknown map.  This is the flat-bottom certification
      machinery applied to a curved bed; whether it still reaches roundoff is the
      question, and the mass modulation it needs is the price.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(cfg.torch_dtype)
    try:
        states = {tie: bed_state(cfg, tie) for tie in ties}
        state = states[ties[0]]
        pos = state["pos"]
        prior = analytic_profile(state, cfg)
        uniform = torch.full((state["n_fluid"],), state["m_unit"],
                             dtype=pos.dtype, device=pos.device)

        controls = {}
        for name, mass in (("uniform_lattice_mass", uniform),
                           ("analytic_hydrostatic_profile", prior["mass"])):
            report = residual_report(state, cfg, mass)
            entry = _summary(report, cfg)
            entry["bands"] = _band_residual(report, state, cfg)
            entry["rest"] = _rest(state, cfg, mass, t_end)
            controls[name] = entry

        solutions, rests, fields = {}, {}, {}
        for tie in ties:
            sol = reconstruct(states[tie], cfg, prior=prior["mass"],
                              verbose=verbose and tie == ties[0])
            entry = {
                "tie": tie,
                "unknowns": int(sol["u_unique"].numel()),
                "particles": state["n_fluid"],
                "iterations": sol["iterations"],
                "converged": sol["converged"],
                "residual_history": sol["residual_history"],
            }
            entry.update(_summary(sol, cfg))
            entry["bands"] = _band_residual(sol, states[tie], cfg)
            entry.update(_mass_modulation(states[tie], cfg, sol["mass"],
                                          prior["mass"]))
            solutions[tie] = entry
            rests[tie] = _rest(states[tie], cfg, sol["mass"], t_end)
            fields["acceleration_" + tie] = sol["acceleration"].cpu()
            fields["mass_" + tie] = sol["mass"].cpu()

        geometry = {
            "width": cfg.width, "depth": cfg.depth, "dx": cfg.dx,
            "h": cfg.h(), "bed_amplitude": cfg.bed_amplitude,
            "bed_waves": cfg.bed_waves,
            "max_bed_slope": max_bed_slope(cfg),
            "clearance_over_dx": cfg.clearance,
            "clearance_min_over_dx": float(state["clearance"].min() / cfg.dx),
            "clearance_max_over_dx": float(state["clearance"].max() / cfg.dx),
            "particles": state["n_fluid"], "wall_particles": state["n_wall"],
            "pairs": int(state["pairs"].shape[0]),
            "water_volume": float(state["n_fluid"] * cfg.dx * cfg.dx),
            "periodic_x": True,
            "sound_speed": cfg.sound_speed, "gamma": cfg.gamma,
            "gravity": cfg.gravity,
            "acoustic_mach": cfg.sound_speed / math.sqrt(cfg.gravity * cfg.depth),
        }
        fields.update({"pos": pos.cpu(), "bed": state["bed"].cpu(),
                       "clearance_over_dx": (state["clearance"] / cfg.dx).cpu(),
                       "wall": state["wall"].cpu()})
        return {
            "case": "CASE-SPH-LAKE-AT-REST/bed",
            "reference": "analytic hydrostatic identity (u = 0, eta = const, "
                         "p = rho0 g (eta - y)); independent of the bed",
            "geometry": geometry,
            "controls": controls,
            "solutions": solutions,
            "rest": rests,
            "fields": fields,
        }
    finally:
        torch.set_default_dtype(previous)


def sweep(cases, t_end=None, verbose: bool = False) -> dict:
    """Run a list of :class:`BedConfig` and return the flattenable result."""
    results = []
    for cfg in cases:
        entry = run_case(cfg, t_end=t_end, verbose=verbose)
        results.append({"bed_amplitude": cfg.bed_amplitude, "dx": cfg.dx,
                        "result": entry})
    return {"case": "CASE-SPH-LAKE-AT-REST/bed", "runs": results}

# --------------------------------------------------------------------------
# Gate table
# --------------------------------------------------------------------------
# The thresholds are declared before the measurement and are graded by the
# runner, the docs generator and the tests from this single list.  ``B1`` and
# ``B4`` are the gates that keep the rest honest (flat identity, live operator);
# ``B2``/``B3`` fail the anchor's tolerance *by design* -- they are the claim
# that the flat-bottom certification does not transport to a curved bed, and a
# future well-balanced operator would turn them red.
GATES = (
    {"gate": "B1", "name": "flat identity (row tie, A=0)",
     "dx": 0.04, "amplitude": 0.0, "comparison": "lt", "threshold": 1.0e-9,
     "select": ("solutions", "row", "a_y_over_g"),
     "why": "A = 0 must reduce to the shipped flat periodic certification"},
    {"gate": "B2", "name": "transfer fails (row tie, A=0.01)",
     "dx": 0.04, "amplitude": 0.01, "comparison": "gt", "threshold": 1.0e-3,
     "select": ("solutions", "row", "a_y_over_g"),
     "why": "the translation-invariant mass field cannot balance a curved bed"},
    {"gate": "B3", "name": "price of the fit (a_x ratio, A=0.005)",
     "dx": 0.04, "amplitude": 0.005, "comparison": "gt", "threshold": 5.0,
     "select": ("solutions", "particle", "a_x_over_g"),
     "ratio_to": ("controls", "analytic_hydrostatic_profile", "a_x_over_g"),
     "why": "fitting a_y costs a horizontal residual worse than not fitting"},
    {"gate": "B4", "name": "operator is live (uniform mass)",
     "dx": 0.04, "amplitude": 0.01, "comparison": "gt", "threshold": 0.5,
     "select": ("controls", "uniform_lattice_mass", "a_y_over_g"),
     "why": "uniform mass in still water must free-fall at g, or B1 is dead"},
    {"gate": "B5", "name": "the fit is a fit (dM/M, A=0.005)",
     "dx": 0.04, "amplitude": 0.005, "comparison": "gt", "threshold": 0.05,
     "select": ("solutions", "particle", "mass_modulation_max_abs"),
     "why": "roundoff at A > 0 is bought with a mass field, and that is the price"},
    {"gate": "B6", "name": "defect is near-bed (A=0.01 vs flat)",
     "dx": 0.04, "amplitude": 0.01, "comparison": "gt", "threshold": 10.0,
     "select": ("controls", "analytic_hydrostatic_profile", "bands",
                "near_bed", "a_y_over_g"),
     "ratio_to_run": {"amplitude": 0.0,
                      "select": ("controls", "analytic_hydrostatic_profile",
                                 "bands", "near_bed", "a_y_over_g")},
     "why": "a curved bed must inject its residual into the near-bed bands, "
            "not into the free surface"},
)


def _dig(entry, path):
    for key in path:
        entry = entry[key]
    return entry


def evaluate_gates(payload: dict) -> dict:
    """Grade ``GATES`` against a ``run.py`` payload; missing runs -> ``not_run``."""
    runs = {}
    for run in payload.get("runs", []):
        runs[(round(run["dx"], 12), round(run["bed_amplitude"], 12))] = run["result"]
    graded = []
    for gate in GATES:
        key = (round(gate["dx"], 12), round(gate["amplitude"], 12))
        result = runs.get(key)
        if result is None:
            graded.append({**{k: gate[k] for k in
                              ("gate", "name", "comparison", "threshold", "why")},
                           "measured": None, "verdict": "not_run"})
            continue
        measured = _dig(result, gate["select"])
        if "ratio_to" in gate:
            measured = measured / _dig(result, gate["ratio_to"])
        if "ratio_to_run" in gate:
            reference = gate["ratio_to_run"]
            other = runs.get((key[0], round(reference["amplitude"], 12)))
            if other is None:
                graded.append({**{k: gate[k] for k in
                                  ("gate", "name", "comparison", "threshold",
                                   "why")},
                               "measured": None, "verdict": "not_run"})
                continue
            measured = measured / _dig(other, reference["select"])
        passed = (measured < gate["threshold"] if gate["comparison"] == "lt"
                  else measured > gate["threshold"])
        graded.append({**{k: gate[k] for k in
                          ("gate", "name", "comparison", "threshold", "why")},
                       "measured": float(measured),
                       "verdict": "pass" if passed else "FAIL"})
    return {"gates": graded,
            "passed": all(g["verdict"] == "pass" for g in graded)}
