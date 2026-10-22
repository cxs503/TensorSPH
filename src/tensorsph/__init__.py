"""TensorSPH: PyTorch-native smoothed particle hydrodynamics."""

from .droplet_impact import (
    Config,
    SimulationUnstableError,
    __version__,
    acoustic_courant,
    apply_boundary,
    boundary_box,
    cubic_spline_gradient,
    cubic_spline_kernel,
    density_from_neighbor_pairs,
    make_particles,
    resolve_device,
    rhs,
    rhs_all_pairs,
    run,
    suggested_dt,
    tait_pressure,
    validate_config,
    wall_lattice,
)
from .neighbor_search import brute_force_pairs, cell_linked_pairs

__all__ = [
    "Config", "SimulationUnstableError", "__version__", "acoustic_courant",
    "apply_boundary", "boundary_box", "cubic_spline_gradient",
    "cubic_spline_kernel", "density_from_neighbor_pairs", "make_particles",
    "resolve_device", "rhs", "rhs_all_pairs", "run", "suggested_dt",
    "tait_pressure", "validate_config", "wall_lattice",
    "brute_force_pairs", "cell_linked_pairs",
]
