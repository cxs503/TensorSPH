"""Reference cases for the cross-validation of the fluid->solid load transfer.

Every case is a *static* particle arrangement with an analytically known load,
so the operator can be scored without the solver error of a coupled run
(the design decision is recorded in ``docs/xval_fsi2/``).  Geometry and
parameters that come from the open-source reference are cited with file:line.

Analytic references
-------------------
* hydrostatic pressure on a vertical wall of depth ``H``::

      F_x = 1/2 rho g H^2                       (per unit thickness)

* hydrostatic pressure on a horizontal floor carrying a column ``H`` deep and
  ``L`` wide::

      F_y = -rho g H L   (the fluid column pushes the floor down)

* plane Couette shear on the lower wall with gap ``h`` and wall speed ``U``::

      tau = mu U / h,      F_x = tau * L

* a *closed* body in a linear pressure field ``p = p0 + c . x``::

      F = -oint p n dl = -c * A          (A = enclosed area, unit thickness)

  which is Archimedes' law written for an arbitrary linear field; it is the
  reference used on the fsi2 insert body (cylinder + beam), where the load the
  structure actually feels is a *pressure* field with no gravity component
  (``fsi2.h`` has no gravity), so the wall-pressure extrapolation of the OSS
  operator contributes nothing there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

# --------------------------------------------------------------------------- #
# OSS parameters (verbatim from the read blueprint, no hand-copied numbers)
# --------------------------------------------------------------------------- #
#: hydrostatic_fsi.cpp:14-19 -- tank 1.0 x 2.1, water block 1.0 x 2.0, gate 0.05
HYDROSTATIC_FSI = {"DL": 1.0, "DH": 2.1, "Dam_L": 1.0, "Dam_H": 2.0,
                   "Gate_width": 0.05,
                   "source": "tests/2d_examples/test_2d_hydrostatic_fsi/hydrostatic_fsi.cpp:14-19"}
#: fsi2.h:14-35 -- channel 11 x 4.1, cylinder r = 0.5 at (2, 2), beam 3.5 x 0.2
FSI2 = {"DL": 11.0, "DH": 4.1, "global_resolution": 0.1,
        "insert_circle_center": (2.0, 2.0), "insert_circle_radius": 0.5,
        "bh": 0.4 * 0.5, "bl": 7.0 * 0.5,
        "rho0_f": 1.0, "U_f": 1.0, "c_f": 10.0, "Re": 100.0,
        "source": "tests/2d_examples/test_2d_fsi2/fsi2.h:14-35"}


@dataclass
class LoadCase:
    """One static configuration plus its analytic reference load."""

    name: str
    description: str
    source: str
    dx: float
    h_factor: float
    pos_solid: torch.Tensor
    vol_solid: torch.Tensor
    normal_solid: torch.Tensor
    pos_fluid: torch.Tensor
    vol_fluid: torch.Tensor
    pressure_fluid: torch.Tensor
    rho_fluid: torch.Tensor
    vel_fluid: torch.Tensor
    vel_solid: torch.Tensor
    acc_ext_fluid: torch.Tensor
    mu_fluid: torch.Tensor
    reference_force: torch.Tensor
    reference_label: str
    wetted_area: float
    fluid_state: str          # "analytic" | "solved"
    notes: str = ""

    @property
    def h(self) -> float:
        return self.h_factor * self.dx

    @property
    def support(self) -> float:
        """Wendland C2 support radius (kernel size 2.0, as in the reference)."""
        return 2.0 * self.h


# --------------------------------------------------------------------------- #
# Lattice helpers
# --------------------------------------------------------------------------- #
def _lattice(lo: List[float], hi: List[float], dx: float) -> torch.Tensor:
    """Cell-centre lattice covering ``[lo, hi)`` in every direction."""
    axes = [torch.arange(lo[a] + 0.5 * dx, hi[a] - 1.0e-12, dx, dtype=torch.float64)
            for a in range(len(lo))]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=1)


def _vol(n: int, dx: float, dim: int = 2) -> torch.Tensor:
    return torch.full((n,), dx ** dim, dtype=torch.float64)


def surface_line(start: List[float], end: List[float], dx: float) -> torch.Tensor:
    """One-particle-thick line of solid particles along a face, placed *on* the face.

    The load-transfer operator acts on the wetted surface particles, so the
    configuration used for the cross-validation is a single surface layer placed
    on the geometric face (as the body-fitted lattice of the reference does).
    """
    lo = torch.tensor(start, dtype=torch.float64)
    hi = torch.tensor(end, dtype=torch.float64)
    length = float((hi - lo).norm())
    n = max(int(round(length / dx)), 1)
    t = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)[:-1] + 0.5 / n
    return lo + t[:, None] * (hi - lo)


def _wall_body(face_start, face_end, dx, layers, inward):
    """Surface layer on the face plus ``layers-1`` inward offsets (solid body)."""
    face = surface_line(face_start, face_end, dx)
    parts = [face]
    direction = torch.tensor(inward, dtype=torch.float64)
    for k in range(1, max(int(layers), 1)):
        parts.append(face + k * dx * direction)
    return torch.cat(parts, dim=0)


def vertical_wall(dx: float, depth: float = HYDROSTATIC_FSI["Dam_H"],
                  length: float = HYDROSTATIC_FSI["Dam_L"],
                  rho: float = 1000.0, gravity: float = 9.81,
                  h_factor: float = 1.3, layers: int = 1) -> LoadCase:
    """Water column against a vertical wall (gravity *parallel* to the wall)."""
    fluid = _lattice([-length, 0.0], [0.0, depth], dx)
    wall = _wall_body([0.0, 0.0], [0.0, depth], dx, layers, inward=(1.0, 0.0))
    p = (rho * gravity * (depth - fluid[:, 1])).clamp_min(0.0)
    f_ref = torch.tensor([0.5 * rho * gravity * depth ** 2, 0.0], dtype=torch.float64)
    return LoadCase(
        name="vertical_wall", source=HYDROSTATIC_FSI["source"],
        description=(f"hydrostatic load on a vertical wall, depth {depth} m; the wall "
                     "normal carries no gravity component, so the OSS wall-pressure "
                     "extrapolation sees only the vertical part of e_ij"),
        dx=dx, h_factor=h_factor,
        pos_solid=wall, vol_solid=_vol(wall.shape[0], dx),
        normal_solid=torch.tensor([-1.0, 0.0], dtype=torch.float64).expand(wall.shape[0], 2).clone(),
        pos_fluid=fluid, vol_fluid=_vol(fluid.shape[0], dx),
        pressure_fluid=p, rho_fluid=torch.full_like(p, rho),
        vel_fluid=torch.zeros_like(fluid), vel_solid=torch.zeros_like(wall),
        acc_ext_fluid=torch.tensor([0.0, -gravity], dtype=torch.float64),
        mu_fluid=torch.zeros_like(p),
        reference_force=f_ref,
        reference_label="1/2 rho g H^2",
        wetted_area=depth, fluid_state="analytic",
    )


def horizontal_gate(dx: float, depth: float = HYDROSTATIC_FSI["Dam_H"],
                    length: float = HYDROSTATIC_FSI["Dam_L"],
                    rho: float = 1000.0, gravity: float = 9.81,
                    h_factor: float = 1.3, layers: int = 1) -> LoadCase:
    """Water column resting on a floor (gravity *perpendicular* to the wall)."""
    fluid = _lattice([0.0, 0.0], [length, depth], dx)
    wall = _wall_body([0.0, 0.0], [length, 0.0], dx, layers, inward=(0.0, -1.0))
    p = (rho * gravity * (depth - fluid[:, 1])).clamp_min(0.0)
    # the fluid sits *above* the floor, so the load it applies points down
    f_ref = torch.tensor([0.0, -rho * gravity * depth * length], dtype=torch.float64)
    return LoadCase(
        name="horizontal_gate", source=HYDROSTATIC_FSI["source"],
        description=(f"water column {length} x {depth} m resting on a floor; the wall "
                     "normal is aligned with gravity, which is the configuration the "
                     "Adami-style wall-pressure extrapolation is designed for"),
        dx=dx, h_factor=h_factor,
        pos_solid=wall, vol_solid=_vol(wall.shape[0], dx),
        normal_solid=torch.tensor([0.0, 1.0], dtype=torch.float64).expand(wall.shape[0], 2).clone(),
        pos_fluid=fluid, vol_fluid=_vol(fluid.shape[0], dx),
        pressure_fluid=p, rho_fluid=torch.full_like(p, rho),
        vel_fluid=torch.zeros_like(fluid), vel_solid=torch.zeros_like(wall),
        acc_ext_fluid=torch.tensor([0.0, -gravity], dtype=torch.float64),
        mu_fluid=torch.zeros_like(p),
        reference_force=f_ref,
        reference_label="rho g H L",
        wetted_area=length, fluid_state="analytic",
    )


def couette_shear(dx: float, gap: float = 0.1, length: float = 1.0,
                  speed: float = 1.0, mu: float = 0.01,
                  h_factor: float = 1.3, layers: int = 1) -> LoadCase:
    """Plane Couette flow over a floor: shear load ``tau = mu U / h`` on the wall."""
    fluid = _lattice([0.0, 0.0], [length, gap], dx)
    if fluid.shape[0] == 0:
        raise ValueError("couette gap too small for this dx")
    wall = surface_line([0.0, 0.0], [length, 0.0], dx)
    vel_f = torch.zeros_like(fluid)
    vel_f[:, 0] = speed * fluid[:, 1] / gap
    p = torch.zeros(fluid.shape[0], dtype=torch.float64)
    f_ref = torch.tensor([mu * speed / gap * length, 0.0], dtype=torch.float64)
    return LoadCase(
        name="couette_shear", source="plane Couette (textbook), viscosity branch of "
        "force_on_structure.hpp:55-72",
        description=(f"linear Couette profile U={speed} over a {gap} m gap, mu={mu}; "
                     "only the viscous branch is exercised"),
        dx=dx, h_factor=h_factor,
        pos_solid=wall, vol_solid=_vol(wall.shape[0], dx),
        normal_solid=torch.tensor([0.0, 1.0], dtype=torch.float64).expand(wall.shape[0], 2).clone(),
        pos_fluid=fluid, vol_fluid=_vol(fluid.shape[0], dx),
        pressure_fluid=p, rho_fluid=torch.ones_like(p),
        vel_fluid=vel_f, vel_solid=torch.zeros_like(wall),
        acc_ext_fluid=torch.zeros(2, dtype=torch.float64),
        mu_fluid=torch.full_like(p, mu),
        reference_force=f_ref,
        reference_label="mu U / h * L",
        wetted_area=length, fluid_state="analytic",
    )


def _fsi2_body(dx: float, pad: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-layer wetted surface of the fsi2 insert body (cylinder + beam).

    Built analytically along the perimeter instead of by carving a lattice, so
    that the surface carries exactly one particle per ``dx`` of wetted length
    (a two-layer shell would double-count the load of the closed body).
    """
    cx, cy = FSI2["insert_circle_center"]
    r = FSI2["insert_circle_radius"]
    bh, bl = FSI2["bh"], FSI2["bl"]
    # cylinder: perimeter 2 pi r, angles offset by half a spacing
    n_cyl = max(int(round(2.0 * math.pi * r / dx)), 8)
    theta = (torch.arange(n_cyl, dtype=torch.float64) + 0.5) * (2.0 * math.pi / n_cyl)
    cyl = torch.stack((cx + r * torch.cos(theta), cy + r * torch.sin(theta)), dim=1)
    n_cyl_v = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    # beam: top and bottom rows plus the tip, starting where the cylinder ends
    x0, x1 = cx + r, cx + r + bl
    y0, y1 = cy - 0.5 * bh, cy + 0.5 * bh
    n_row = max(int(round(bl / dx)), 1)
    xs = x0 + (torch.arange(n_row, dtype=torch.float64) + 0.5) * (bl / n_row)
    top = torch.stack((xs, torch.full_like(xs, y1)), dim=1)
    bot = torch.stack((xs, torch.full_like(xs, y0)), dim=1)
    n_col = max(int(round(bh / dx)), 1)
    ys = y0 + (torch.arange(n_col, dtype=torch.float64) + 0.5) * (bh / n_col)
    tip = torch.stack((torch.full_like(ys, x1), ys), dim=1)
    pos = torch.cat((cyl, top, bot, tip), dim=0)
    nrm = torch.cat((n_cyl_v,
                     torch.tensor([0.0, 1.0], dtype=torch.float64).expand_as(top),
                     torch.tensor([0.0, -1.0], dtype=torch.float64).expand_as(bot),
                     torch.tensor([1.0, 0.0], dtype=torch.float64).expand_as(tip)), dim=0)
    del pad
    return pos, nrm


def fsi2_insert_linear(dx: float, gradient=(0.0, -1.0), p0: float = 1.0,
                       h_factor: float = 1.3) -> LoadCase:
    """fsi2 insert body in an analytic *linear* pressure field (no gravity).

    The analytic load of a closed body in ``p = p0 + c.x`` is ``-c * A`` where
    ``A`` is the enclosed area; with ``c = (0, -1)`` the body feels ``+A`` in y.
    ``acc_ext_fluid = 0`` on purpose: the fsi2 benchmark has no gravity, so this
    case measures the operator on exactly the surface the OSS benchmark loads.
    """
    c = torch.tensor(gradient, dtype=torch.float64)
    pad = 4.0 * h_factor * dx
    solid, normal = _fsi2_body(dx, pad)
    cx, cy = FSI2["insert_circle_center"]
    r, bh, bl = FSI2["insert_circle_radius"], FSI2["bh"], FSI2["bl"]
    lo = torch.tensor([cx - r - pad, cy - 0.5 * bh - pad], dtype=torch.float64)
    hi = torch.tensor([cx + r + bl + pad * 0.0 + 3.0 * h_factor * dx,
                       cy + max(0.5 * bh, r) + pad], dtype=torch.float64)
    fluid = _lattice([float(lo[0]), float(lo[1])], [float(hi[0]), float(hi[1])], dx)
    # drop fluid points inside the body
    dcc = torch.linalg.vector_norm(fluid - torch.tensor([cx, cy], dtype=torch.float64),
                                   dim=1)
    inside_cyl = dcc < (r - 0.5 * dx)
    inside_beam = ((fluid[:, 0] > cx + r - 0.5 * dx) & (fluid[:, 0] < cx + r + bl)
                   & (fluid[:, 1] > cy - 0.5 * bh) & (fluid[:, 1] < cy + 0.5 * bh))
    fluid = fluid[~(inside_cyl | inside_beam)]
    p = p0 + fluid @ c
    area = math.pi * r * r + bh * bl
    f_ref = -c * area
    return LoadCase(
        name="fsi2_insert_linear", source=FSI2["source"],
        description=("fsi2 insert body (cylinder r=0.5 + beam 3.5x0.2) in a linear "
                     "pressure field with no gravity; analytic load = -c A"),
        dx=dx, h_factor=h_factor,
        pos_solid=solid, vol_solid=_vol(solid.shape[0], dx),
        normal_solid=normal,
        pos_fluid=fluid, vol_fluid=_vol(fluid.shape[0], dx),
        pressure_fluid=p, rho_fluid=torch.ones_like(p),
        vel_fluid=torch.zeros_like(fluid), vel_solid=torch.zeros_like(solid),
        acc_ext_fluid=torch.zeros(2, dtype=torch.float64),
        mu_fluid=torch.zeros_like(p),
        reference_force=f_ref,
        reference_label="-c A (linear field, closed body)",
        wetted_area=float("nan"), fluid_state="analytic",
        notes="analytic wall area only; the load is compared as a total force",
    )


def curved_disk_shear(dx: float, radius: float = 0.5, centre=(2.0, 2.0),
                      gamma: float = 1.0, beta: float = 1.0, mu: float = 0.01,
                      h_factor: float = 1.3) -> "LoadCase":
    """Curved wall: a closed disk in a *quadratic* shear field ``u_x = g y + b y^2/2``.

    This is the only case in the bundle with a non-flat wall, so it separates
    curvature error from the flat-wall truncation error of X1-X3.  The exact
    traction integral over the closed surface is

        F_x = oint mu (du_x/dy) n_y dl = mu [ g oint n_y dl + b oint y n_y dl ]
            = mu b * A,          A = pi R^2   (divergence theorem, oint n_y dl = 0)

    i.e. the linear part cancels identically and the quadratic part is the body
    average of the shear gradient -- a non-degenerate, analytically known scalar.
    ``vel_solid`` carries the *no-slip* value of the same field on the wall
    (``vel_solid == u(x_solid)``); feeding a mismatched solid velocity is not a
    variant of this case, it is invalid data (see the pitfall in the report).

    Geometry is the fsi2 cylinder (``fsi2.h:14-35``: r = 0.5 at (2, 2)) with the
    beam removed, because the beam's rear face is not wetted and would add an
    interior surface to the "closed body".
    """
    cx, cy = centre
    fluid = _lattice([cx - 4.0 * radius, cy - 4.0 * radius],
                     [cx + 4.0 * radius, cy + 4.0 * radius], dx)
    rc = torch.tensor([cx, cy], dtype=torch.float64)
    fluid = fluid[torch.linalg.vector_norm(fluid - rc, dim=1) > radius + 0.25 * dx]
    n_face = max(int(round(2.0 * math.pi * radius / dx)), 8)
    ang = torch.arange(n_face, dtype=torch.float64) * 2.0 * math.pi / n_face
    wall = torch.stack([cx + radius * torch.cos(ang), cy + radius * torch.sin(ang)], dim=1)
    normal = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)  # outward = into fluid

    def field(x: torch.Tensor) -> torch.Tensor:
        y = x[:, 1] - cy
        v = torch.zeros_like(x)
        v[:, 0] = gamma * y + 0.5 * beta * y * y
        return v

    f_ref = torch.tensor([mu * beta * math.pi * radius ** 2, 0.0], dtype=torch.float64)
    return LoadCase(
        name="curved_disk_shear",
        source="fsi2.h:14-35 (cylinder r=0.5 at (2,2)); viscous branch of "
               "force_on_structure.hpp:55-72 on a curved wall",
        description=(f"closed disk R={radius} in a quadratic shear field "
                     f"u_x = {gamma} y + {beta} y^2/2, mu={mu}; exact net load = mu beta A"),
        dx=dx, h_factor=h_factor,
        pos_solid=wall, vol_solid=_vol(wall.shape[0], dx), normal_solid=normal,
        pos_fluid=fluid, vol_fluid=_vol(fluid.shape[0], dx),
        pressure_fluid=torch.zeros(fluid.shape[0], dtype=torch.float64),
        rho_fluid=torch.ones(fluid.shape[0], dtype=torch.float64),
        vel_fluid=field(fluid), vel_solid=field(wall),
        acc_ext_fluid=torch.zeros(2, dtype=torch.float64),
        mu_fluid=torch.full((fluid.shape[0],), mu, dtype=torch.float64),
        reference_force=f_ref, reference_label="mu beta A (quadratic shear, disk)",
        wetted_area=2.0 * math.pi * radius, fluid_state="analytic",
    )


CASE_BUILDERS: Dict[str, Callable[..., LoadCase]] = {
    "vertical_wall": vertical_wall,
    "horizontal_gate": horizontal_gate,
    "couette_shear": couette_shear,
    "fsi2_insert_linear": fsi2_insert_linear,
    "curved_disk_shear": curved_disk_shear,
}


def dx_ladder(case: str, ladder: Optional[List[float]] = None) -> List[float]:
    default = {"vertical_wall": [0.05, 0.025, 0.0125],
               "horizontal_gate": [0.05, 0.025, 0.0125],
               "couette_shear": [0.025, 0.0125, 0.00625],
               "fsi2_insert_linear": [0.1, 0.05, 0.025],
               "curved_disk_shear": [0.1, 0.05, 0.025, 0.0125]}
    return list(ladder if ladder is not None else default[case])
