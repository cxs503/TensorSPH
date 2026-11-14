"""The start-up Couette series used by CASE-SPH-COUETTE-TRANSIENT is verified here.

The benchmark compares SPH against the closed form

    u(y,t) = U y/H + (2U/pi) sum_{n>=1} (-1)^n/n sin(n pi y/H) e^{-n^2 pi^2 nu t/H^2}

so a sign or factor error in that series would masquerade as a solver error.  The
series is therefore checked against an *independent* discretisation of the same
initial/boundary-value problem (Crank-Nicolson on the 1-D heat equation) and
against its analytic limits.  No reference number is transcribed from the
literature anywhere in this file: every target is either the PDE solution
computed here or a limit of the closed form.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.couette_transient.run import (  # noqa: E402
    couette_series,
    exact_scalars,
    flux_exact,
    shear_factor_exact,
)

HEIGHT = 0.1
U = 0.2
NU = 0.02


def crank_nicolson(t_end: float, n_nodes: int = 201, height: float = HEIGHT,
                   viscosity: float = NU, wall_speed: float = U):
    """Independent FD solution of u_t = nu u_yy, u(y,0)=0, u(0,t)=0, u(H,t)=U.

    Crank-Nicolson with ``nu dt / dy^2 = 0.25``, solved exactly (banded LU, no
    iteration) so the check is a discretisation of the PDE and not of the series.
    Returns ``(y, u)`` at ``t_end``.
    """
    from scipy.linalg import solve_banded

    y = np.linspace(0.0, height, n_nodes)
    dy = y[1] - y[0]
    dt_target = 0.25 * dy ** 2 / viscosity
    n_steps = max(1, int(round(t_end / dt_target)))
    dt = t_end / n_steps                       # land exactly on t_end
    r = viscosity * dt / (2.0 * dy ** 2)       # CN half-step coefficient

    u = np.zeros(n_nodes)
    u[-1] = wall_speed

    m = n_nodes - 2
    banded = np.zeros((3, m))
    banded[0, 1:] = -r            # super-diagonal
    banded[1, :] = 1.0 + 2.0 * r  # diagonal
    banded[2, :-1] = -r           # sub-diagonal

    for _ in range(n_steps):
        rhs = r * u[:-2] + (1.0 - 2.0 * r) * u[1:-1] + r * u[2:]
        rhs[0] += r * u[0]          # bottom Dirichlet (0)
        rhs[-1] += r * u[-1]        # top Dirichlet (U)
        u[1:-1] = solve_banded((1, 1), banded, rhs)
    return y, u


@pytest.mark.parametrize("t_star", [0.02, 0.1, 0.5])
def test_series_matches_crank_nicolson(t_star: float):
    t = t_star * HEIGHT ** 2 / NU
    y, u_fd = crank_nicolson(t)
    u_series = couette_series(y, t, HEIGHT, U, NU)
    err = np.max(np.abs(u_series - u_fd)) / U
    assert err < 2.0e-3, f"t*={t_star}: series vs CN max error {err:.2e} U"


def test_series_satisfies_initial_and_steady_limits():
    y = np.linspace(0.05 * HEIGHT, 0.95 * HEIGHT, 50)
    # t* -> 0 in the interior: the profile the solver starts from is zero
    u0 = couette_series(y, 1e-4 * HEIGHT ** 2 / NU, HEIGHT, U, NU)
    assert np.max(np.abs(u0)) / U < 0.02
    # t* -> infinity: the linear steady profile
    u_inf = couette_series(y, 3.0 * HEIGHT ** 2 / NU, HEIGHT, U, NU)
    assert np.max(np.abs(u_inf - U * y / HEIGHT)) / U < 1e-8


def test_series_satisfies_boundary_conditions():
    t = 0.1 * HEIGHT ** 2 / NU
    u = couette_series(np.array([0.0, HEIGHT]), t, HEIGHT, U, NU)
    assert abs(u[0]) < 1e-12 * U
    assert abs(u[1] - U) < 1e-12 * U


def test_flux_and_shear_exact_limits():
    # zero flux at the start, half the steady flux as t* -> infinity
    assert abs(flux_exact(0.0, HEIGHT, U)) < 1e-12
    assert abs(flux_exact(50.0, HEIGHT, U) / (0.5 * U * HEIGHT) - 1.0) < 1e-12
    # wall shear starts at zero (impulsive start) and reaches rho nu U/H
    assert abs(shear_factor_exact(0.0)) < 1e-12
    assert abs(shear_factor_exact(50.0) - 1.0) < 1e-12
    scalars = exact_scalars(0.1 * HEIGHT ** 2 / NU, HEIGHT, U, NU)
    assert 0.0 < scalars["wall_shear"] < 1000.0 * NU * U / HEIGHT


def test_flux_is_consistent_with_the_profile_series():
    """int u(y,t) dy from the profile must equal the closed-form flux."""
    t_star = 0.12
    t = t_star * HEIGHT ** 2 / NU
    y = np.linspace(0.0, HEIGHT, 20001)
    u = couette_series(y, t, HEIGHT, U, NU)
    q = float(np.trapezoid(u, y))
    assert abs(q / flux_exact(t_star, HEIGHT, U) - 1.0) < 2e-4


def test_shear_factor_consistent_with_profile_derivative():
    """du/dy at the wall from the series must give the same shear factor."""
    t_star = 0.15
    t = t_star * HEIGHT ** 2 / NU
    dy = 1e-6 * HEIGHT
    u = couette_series(np.array([0.0, dy]), t, HEIGHT, U, NU)
    slope = (u[1] - u[0]) / dy
    assert abs(slope / (U / HEIGHT) - shear_factor_exact(t_star)) < 1e-4