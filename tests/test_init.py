"""Tests for the package re-exports in ``tensorsph.__init__``.

The ``__init__.py`` re-exports the public API from sub-modules.  These tests
verify that every name listed in ``__all__`` is actually importable and has
the expected type (function, class, or constant).  This is a smoke test that
catches broken imports early (e.g. after a refactor that renames a function
but forgets to update the re-export).
"""
from __future__ import annotations

import inspect


import tensorsph


# --------------------------------------------------------------------------- #
# Positive cases
# --------------------------------------------------------------------------- #

def test_all_names_are_importable():
    """Every name in __all__ must be accessible as ``tensorsph.<name>``."""
    missing = []
    for name in tensorsph.__all__:
        if not hasattr(tensorsph, name):
            missing.append(name)
    assert not missing, f"Missing re-exports: {missing}"


def test_key_classes_are_classes():
    """Config and DamBreakConfig must be classes."""
    assert inspect.isclass(tensorsph.Config)
    assert inspect.isclass(tensorsph.DamBreakConfig)


def test_key_functions_are_callable():
    """Core solver functions must be callable."""
    callables = [
        "rhs", "run", "make_particles", "cubic_spline_kernel",
        "cubic_spline_gradient", "cell_linked_pairs", "brute_force_pairs",
        "make_column", "run_case",
    ]
    for name in callables:
        assert hasattr(tensorsph, name), f"Missing: {name}"
        assert callable(getattr(tensorsph, name)), f"Not callable: {name}"


def test_version_is_string():
    """__version__ must be a non-empty string."""
    assert isinstance(tensorsph.__version__, str)
    assert len(tensorsph.__version__) > 0


def test_surface_tension_re_exports():
    """Surface-tension functions must be accessible from the top level."""
    st_funcs = [
        "compute_color_gradient", "compute_normal", "compute_curvature",
        "compute_surface_tension_force", "capillary_dt_limit",
        "apply_contact_angle_boundary",
    ]
    for name in st_funcs:
        assert hasattr(tensorsph, name), f"Missing: {name}"
        assert callable(getattr(tensorsph, name)), f"Not callable: {name}"


def test_neighbor_search_re_exports():
    """Neighbor-search functions must be accessible from the top level."""
    ns_funcs = [
        "PairListCache", "brute_force_pairs", "cell_linked_pairs",
        "cell_linked_pairs_host",
    ]
    for name in ns_funcs:
        assert hasattr(tensorsph, name), f"Missing: {name}"


def test_initial_density_re_exports():
    """Initial-density functions must be accessible from the top level."""
    id_funcs = [
        "calibration_factor", "calibrate_mass", "lattice_bias",
        "lattice_kernel_sum", "lattice_offsets",
        "measure_complete_support_bias", "spurious_pressure",
        "spurious_pressure_ratio",
    ]
    for name in id_funcs:
        assert hasattr(tensorsph, name), f"Missing: {name}"


def test_all_is_a_list():
    """__all__ must be a list or tuple, not a string or dict."""
    assert isinstance(tensorsph.__all__, (list, tuple))


def test_no_duplicate_names_in_all():
    """__all__ must not contain duplicates."""
    assert len(tensorsph.__all__) == len(set(tensorsph.__all__))


# --------------------------------------------------------------------------- #
# Negative cases (error injection must turn red)
# --------------------------------------------------------------------------- #

def test_nonexistent_name_not_in_all():
    """A made-up name must NOT be in __all__."""
    assert "nonexistent_function_xyz" not in tensorsph.__all__


def test_all_has_reasonable_size():
    """__all__ should have a reasonable number of exports (>10, <100)."""
    n = len(tensorsph.__all__)
    assert 10 < n < 100, f"__all__ has {n} entries, expected 10-100"
