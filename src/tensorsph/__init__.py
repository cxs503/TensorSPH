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
    hydrostatic_density,
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
from .dam_break import DamBreakConfig, make_column, run_case
from .neighbor_search import (
    PairListCache,
    brute_force_pairs,
    cell_linked_pairs,
    cell_linked_pairs_host,
)

__all__ = [
    "Config", "SimulationUnstableError", "__version__", "acoustic_courant",
    "apply_boundary", "boundary_box", "cubic_spline_gradient",
    "cubic_spline_kernel", "density_from_neighbor_pairs", "hydrostatic_density",
    "make_particles",
    "resolve_device", "rhs", "rhs_all_pairs", "run", "suggested_dt",
    "tait_pressure", "validate_config", "wall_lattice",
    "PairListCache", "brute_force_pairs", "cell_linked_pairs",
    "cell_linked_pairs_host", "DamBreakConfig", "make_column", "run_case",
]
