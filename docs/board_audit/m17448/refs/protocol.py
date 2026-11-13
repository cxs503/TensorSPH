"""Multi-generation regression-test protocol with DTW baselines.

Reference practice
------------------
``Xiangyu-Hu/SPHinXsys``:

* ``src/shared/regression_test/regression_test_base.hpp`` -- file layout and the
  generation bookkeeping (``_Run_N_result.xml``, ``_runtimes.dat``);
* ``src/shared/regression_test/dynamic_time_warping_method.hpp`` -- the DTW
  variant (``generateDataBase``, ``updateDTWDistance``, ``compareDTWDistance``,
  ``resultTest``);
* ``src/shared/regression_test/dynamic_time_warping_method.h`` -- the two entry
  points a case calls, ``generateDataBase(threshold)`` and ``testResult()``.

The protocol, restated from those sources:

1. Every run of a case appends one *generation*.  ``<dynamic>_<quantity>_runtimes.dat``
   holds three lines: ``converged`` (``true``/``false``), ``number_of_run`` and
   ``label_for_repeat``.  The constructor reads it; the destructor rewrites it
   and increments ``number_of_run`` while ``converged`` is still ``false``.
2. ``generateDataBase(threshold)`` stores the current observation series as
   ``_Run_<number_of_run-1>_result.xml`` and folds it into the accepted
   distances: for every previous generation ``n`` the DTW distance between the
   current series and generation ``n`` is computed, and the accepted distance is
   the running maximum ``max(previous_accepted, new_pairwise, running)``.  The
   distance therefore only ever grows -- a baseline is never loosened.
3. Convergence of the database is declared only when the accepted distance
   changes by at most ``threshold`` for **four successive generations**
   (``label_for_repeat == 4``); any larger revision resets the counter.
4. ``testResult()`` compares the current series against every stored
   generation and fails when the current distance exceeds ``1.01`` times the
   accepted distance (a 1% band on top of the accepted baseline).

Deliberate differences from the reference implementation are listed in
``docs/XVAL-SPH-DTW-REGRESSION-BASELINE-PROTOCOL.md`` §5 (XML dialect, no
extreme-value filter, explicit ``finalize()`` instead of a C++ destructor,
exceptions instead of ``exit(1)``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree

import numpy as np

from .dtw import DTWLengthError, dtw_distance, dtw_distance_strict

__all__ = [
    "DTWRegressionFailure",
    "DTWRegressionTest",
    "ProtocolConfig",
    "run_generation",
]

ACCEPTANCE_MARGIN = 1.01     # ``dtw_distance_current > 1.01 * accepted`` -> fail
REPEAT_REQUIRED = 4          # ``label_for_repeat_ == 4`` -> converged


class DTWRegressionFailure(AssertionError):
    """The current observation series is outside the accepted DTW baseline."""


@dataclass
class ProtocolConfig:
    """Where the protocol lives and how strict it is.

    ``workdir`` is the equivalent of the reference ``input`` folder: it holds
    the generation files, the accepted distances and the runtime bookkeeping.
    ``dynamics_name`` and ``quantity_names`` build the file names
    ``<dynamics>_<quantity>_Run_N_result.xml``, so they must be stable across
    generations.
    """

    workdir: Path
    dynamics_name: str = "Dynamics"
    quantity_names: Sequence[str] = ("front",)
    window: int = 5                 # DTW half-band width (reference: 5)
    mode: str = "sphinxsys"         # "sphinxsys" (faithful) or "strict"
    length_guard: float | None = 0.2
    acceptance_margin: float = ACCEPTANCE_MARGIN
    repeat_required: int = REPEAT_REQUIRED

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.quantity_names = tuple(self.quantity_names)
        if self.mode not in {"sphinxsys", "strict"}:
            raise ValueError(f"mode must be 'sphinxsys' or 'strict', got {self.mode!r}")
        if not self.quantity_names:
            raise ValueError("at least one observed quantity is required")

    # -- file names, mirroring the reference layout -------------------------
    def _stem(self) -> str:
        return f"{self.dynamics_name}_{'_'.join(self.quantity_names)}"

    def result_path(self, run_index: int) -> Path:
        return self.workdir / f"{self._stem()}_Run_{run_index}_result.xml"

    @property
    def distance_path(self) -> Path:
        return self.workdir / f"{self._stem()}_dtwdistance.xml"

    @property
    def runtimes_path(self) -> Path:
        return self.workdir / f"{self._stem()}_runtimes.dat"


class DTWRegressionTest:
    """One generation of the protocol; mirrors the reference class.

    Typical use, exactly as a case file does it in the reference practice::

        test = DTWRegressionTest(cfg)
        for values in observation_series:      # snapshot by snapshot
            test.remember_observations(values)
        test.transpose_the_index()
        if generating_baseline:
            test.generate_database(threshold=0.01)
        else:
            test.test_result()                 # raises DTWRegressionFailure
        test.finalize()                        # writes runtimes.dat
    """

    def __init__(self, config: ProtocolConfig) -> None:
        self.config = config
        self.config.workdir.mkdir(parents=True, exist_ok=True)

        self.observation = len(config.quantity_names)
        self.current_result: list[list[float]] = []   # snapshot x observation
        self.current_result_trans: np.ndarray | None = None   # observation x snapshot
        self.result_in: np.ndarray | None = None      # observation x snapshot (read)
        self.accepted = np.zeros(self.observation)    # dtw_distance_
        self.accepted_new = np.zeros(self.observation)  # dtw_distance_new_
        self.number_of_snapshot_old = 0
        self.snapshot_difference = 0

        converged, number_of_run, label = self.read_runtimes()
        self.converged = converged
        self.number_of_run = number_of_run
        self.label_for_repeat = label

        #: pair-wise distances computed while folding a generation in
        self.pairwise: dict[int, list[float]] = {}
        #: distances seen by ``test_result``
        self.checked: dict[int, list[float]] = {}
        #: (generation, quantity) pairs where a differing series produced a zero
        #: distance -- the zero-initialised band can truncate the alignment path
        #: (see docs, section on the reference quirk)
        self.degenerate: list[dict] = []

    # -- observation bookkeeping -------------------------------------------
    def remember_observations(self, values: Sequence[float]) -> None:
        """Append one snapshot of observed values (one per observed quantity)."""
        row = [float(value) for value in values]
        if len(row) != self.observation:
            raise ValueError(
                f"expected {self.observation} observed quantities, got {len(row)}")
        self.current_result.append(row)

    def transpose_the_index(self) -> None:
        if not self.current_result:
            raise ValueError("no observations were remembered")
        self.current_result_trans = np.asarray(self.current_result, dtype=float).T
        self.snapshot = len(self.current_result)

    # -- runtime bookkeeping (runtimes.dat) --------------------------------
    def read_runtimes(self) -> tuple[str, int, int]:
        path = self.config.runtimes_path
        if not path.exists():
            return "false", 1, 0
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"{path} is malformed (3 lines expected): {lines}")
        return lines[0], int(lines[1]), int(lines[2])

    def finalize(self) -> None:
        """Write ``runtimes.dat``; equivalent of the reference destructor.

        A run that has not been folded into the database yet is a new
        generation, so ``number_of_run`` advances while ``converged`` is false.
        """
        if self.converged == "false":
            self.number_of_run += 1
        path = self.config.runtimes_path
        path.write_text(f"{self.converged}\n{self.number_of_run}\n{self.label_for_repeat}\n")

    # -- generation files ---------------------------------------------------
    def write_result_to_xml(self, run_index: int) -> Path:
        if self.current_result_trans is None:
            raise ValueError("call transpose_the_index() before writing a generation")
        snapshots = self.current_result_trans.shape[1]
        root = ElementTree.Element("xml")
        snapshot_element = ElementTree.SubElement(root, "Snapshot_Element")
        ElementTree.SubElement(
            snapshot_element, "Snapshot",
            {"number_of_snapshot_for_local_result_": str(snapshots)})
        result_element = ElementTree.SubElement(root, "Result_Element")
        for k in range(self.observation):
            attributes = {
                f"snapshot_{l}": _fmt(self.current_result_trans[k][l])
                for l in range(snapshots)
            }
            ElementTree.SubElement(result_element, f"Particle_{k}", attributes)
        path = self.config.result_path(run_index)
        ElementTree.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
        self.number_of_snapshot_old = snapshots
        self.snapshot_difference = 0
        return path

    def read_result_from_xml(self, run_index: int) -> np.ndarray:
        path = self.config.result_path(run_index)
        if not path.exists():
            raise FileNotFoundError(f"generation {run_index} is missing: {path}")
        root = ElementTree.parse(path).getroot()
        snapshot_element = root.find("Snapshot_Element")
        stored = int(snapshot_element.find("Snapshot").get(
            "number_of_snapshot_for_local_result_"))
        result_element = root.find("Result_Element")
        particles = list(result_element)
        if len(particles) != self.observation:
            raise ValueError(
                f"{path} stores {len(particles)} observed quantities, "
                f"configuration expects {self.observation}")
        series = np.zeros((self.observation, stored))
        for k, particle in enumerate(particles):
            for l in range(stored):
                series[k][l] = float(particle.get(f"snapshot_{l}"))
        # A shorter new result trims the stored ones to a common length, so that
        # all series fed to the DTW have the same number of snapshots.
        self.number_of_snapshot_old = stored
        self.snapshot_difference = stored - getattr(self, "snapshot", stored)
        if self.snapshot_difference > 0:
            rows = series.shape[1] - self.snapshot_difference
            series = series[:, :rows]
            self.number_of_snapshot_old = rows
        self.result_in = series
        return series

    # -- accepted distances ------------------------------------------------
    def read_dtw_distance_from_xml(self) -> np.ndarray:
        if self.number_of_run <= 1:
            return self.accepted
        path = self.config.distance_path
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing but {self.number_of_run} generations exist")
        root = ElementTree.parse(path).getroot()
        for element in root.iter():
            if element is root:
                continue
            for k, name in enumerate(self.config.quantity_names):
                value = element.get(f"{name}_{k}")
                if value is not None:
                    self.accepted[k] = float(value)
        return self.accepted

    def write_dtw_distance_to_xml(self) -> Path:
        root = ElementTree.Element("xml")
        attributes = {
            f"{name}_{k}": _fmt(self.accepted_new[k])
            for k, name in enumerate(self.config.quantity_names)
        }
        ElementTree.SubElement(root, "DTWDistance", attributes)
        path = self.config.distance_path
        ElementTree.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
        return path

    def _register_degeneracy(self, label: int, quantity_index: int,
                             series_a: np.ndarray, series_b: np.ndarray,
                             distance: float) -> None:
        if distance != 0.0:
            return
        if series_a.shape == series_b.shape and np.array_equal(series_a, series_b):
            return
        self.degenerate.append({
            "generation": label,
            "quantity": self.config.quantity_names[quantity_index],
            "lengths": [int(series_a.size), int(series_b.size)],
            "max_abs_difference": float(np.max(np.abs(
                series_a - series_b)) if series_a.shape == series_b.shape else np.inf),
        })

    def _distance(self, series_a: np.ndarray, series_b: np.ndarray,
                  label: int | None = None) -> list[float]:
        function = dtw_distance if self.config.mode == "sphinxsys" else dtw_distance_strict
        distances = []
        for k in range(self.observation):
            try:
                distances.append(function(
                    series_a[k], series_b[k],
                    window=self.config.window if self.config.mode == "strict"
                    else self.config.window,
                    length_guard=self.config.length_guard))
            except DTWLengthError as error:  # never silently skip a quantity
                raise DTWLengthError(f"quantity {self.config.quantity_names[k]}: {error}") from error
            if label is not None and self.config.mode == "sphinxsys":
                self._register_degeneracy(label, k, series_a[k], series_b[k], distances[-1])
        return distances

    def setup_the_test(self) -> None:
        self.accepted = np.zeros(self.observation)
        self.accepted_new = np.zeros(self.observation)
        if self.number_of_run > 1 and not self.config.distance_path.exists():
            raise FileNotFoundError(
                f"{self.config.distance_path} is missing but {self.number_of_run} generations exist")

    def update_dtw_distance(self) -> list[float]:
        """Fold ``result_in`` into the accepted distances (running maximum)."""
        if self.number_of_run <= 1 or self.result_in is None:
            return []
        local = self._distance(self.current_result_trans, self.result_in,
                               label=len(self.pairwise))
        for k in range(self.observation):
            self.accepted_new[k] = max(local[k], self.accepted[k], self.accepted_new[k])
        return local

    def compare_dtw_distance(self, threshold: float) -> bool:
        """Convergence rule: four successive revisions within ``threshold``."""
        if self.number_of_run <= 1:
            return True
        revision = np.abs(self.accepted - self.accepted_new)
        not_converged = int(np.sum(revision > threshold))
        if not_converged:
            self.converged = "false"
            self.label_for_repeat = 0
            return False
        if self.label_for_repeat == self.config.repeat_required:
            self.converged = "true"
            return True
        self.converged = "false"
        self.label_for_repeat += 1
        return True

    # -- the two entry points ----------------------------------------------
    def generate_database(self, threshold: float) -> dict:
        """Add the current run to the baseline database (reference: generateDataBase)."""
        if self.current_result_trans is None:
            raise ValueError("call transpose_the_index() first")
        if self.converged != "false":
            return {"converged": self.converged, "skipped": True}
        self.setup_the_test()
        self.read_dtw_distance_from_xml()
        for run_index in range(self.number_of_run - 1):
            self.read_result_from_xml(run_index)
            local = self.update_dtw_distance()
            if local:
                self.pairwise[run_index] = local
        self.write_result_to_xml(self.number_of_run - 1)
        self.write_dtw_distance_to_xml()
        converged_now = self.compare_dtw_distance(threshold)
        return {
            "converged": self.converged,
            "converges_this_generation": bool(converged_now),
            "label_for_repeat": self.label_for_repeat,
            "number_of_run": self.number_of_run,
            "accepted": self.accepted.tolist(),
            "accepted_new": self.accepted_new.tolist(),
            "pairwise": {str(k): v for k, v in self.pairwise.items()},
            "degenerate_zero_distances": list(self.degenerate),
        }

    def test_result(self) -> dict:
        """Compare the current run with the database (reference: testResult)."""
        if self.current_result_trans is None:
            raise ValueError("call transpose_the_index() first")
        self.setup_the_test()
        self.read_dtw_distance_from_xml()
        failures = []
        for run_index in range(self.number_of_run):
            if not self.config.result_path(run_index).exists():
                continue  # a generation that was not preserved is skipped
            self.read_result_from_xml(run_index)
            current = self._distance(self.result_in, self.current_result_trans,
                                     label=run_index)
            self.checked[run_index] = current
            for k in range(self.observation):
                limit = self.config.acceptance_margin * self.accepted[k]
                if current[k] > limit:
                    failures.append({
                        "quantity": self.config.quantity_names[k],
                        "generation": run_index,
                        "current": current[k],
                        "accepted": self.accepted[k],
                        "limit": limit,
                        "ratio": current[k] / limit if limit else math.inf,
                    })
        report = {
            "checked": {str(k): v for k, v in self.checked.items()},
            "accepted": self.accepted.tolist(),
            "failures": failures,
            "passed": not failures,
            "degenerate_zero_distances": list(self.degenerate),
        }
        if failures:
            raise DTWRegressionFailure(json.dumps(report, indent=2))
        return report


def _fmt(value: float) -> str:
    return repr(float(value))


def run_generation(workdir, observation_series, threshold: float = 0.01,
                   generate: bool = True, **config_kwargs) -> dict:
    """One protocol generation for an already-observed series.

    ``observation_series`` is ``snapshot x observation``: a row per snapshot, a
    column per observed quantity.  ``generate=True`` extends the baseline
    database, ``generate=False`` tests the series against it and raises
    :class:`DTWRegressionFailure` when the accepted distance is exceeded.
    """
    config = ProtocolConfig(workdir=workdir, **config_kwargs)
    test = DTWRegressionTest(config)
    state_before = {
        "converged": test.converged,
        "number_of_run": test.number_of_run,
        "label_for_repeat": test.label_for_repeat,
    }
    for row in observation_series:
        test.remember_observations(row)
    test.transpose_the_index()
    try:
        if generate:
            report = test.generate_database(threshold)
        else:
            report = test.test_result()
    finally:
        test.finalize()
    report["state_before"] = state_before
    report["state_after"] = {
        "converged": test.converged,
        "number_of_run": test.number_of_run,
        "label_for_repeat": test.label_for_repeat,
    }
    report["snapshots"] = int(test.current_result_trans.shape[1])
    return report