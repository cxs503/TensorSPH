"""Regression-test protocol for TensorSPH (DTW baselines).

Ported from the practice of ``Xiangyu-Hu/SPHinXsys``
(``src/shared/regression_test/``): a case is not judged by a single-run
tolerance but by the distance between the current observation series and a set
of accepted baseline generations, with dynamic-time-warping (DTW) distances and
a multi-generation convergence rule.

Public surface:

* :mod:`tensorsph.regression_test.dtw` -- the DTW distance (faithful port plus a
  canonical and an independent shortest-path implementation used to cross-check
  it);
* :mod:`tensorsph.regression_test.protocol` -- the generation/baseline protocol
  (``Run_N_result.xml`` / ``dtwdistance.xml`` / ``runtimes.dat``);
* :mod:`tensorsph.regression_test.observe` -- turning a TensorSPH case run into
  an observation series.

See ``docs/XVAL-SPH-DTW-REGRESSION-BASELINE-PROTOCOL.md`` for the cross
validation against the open-source implementation and for the list of
deliberate differences.
"""

from .dtw import (
    DTWLengthError,
    band_edge_shortcuts,
    dtw_distance,
    dtw_distance_shortest_path,
    dtw_distance_strict,
)
from .protocol import (
    DTWRegressionFailure,
    DTWRegressionTest,
    ProtocolConfig,
    run_generation,
)

__all__ = [
    "DTWLengthError",
    "DTWRegressionFailure",
    "DTWRegressionTest",
    "ProtocolConfig",
    "band_edge_shortcuts",
    "dtw_distance",
    "dtw_distance_shortest_path",
    "dtw_distance_strict",
    "run_generation",
]