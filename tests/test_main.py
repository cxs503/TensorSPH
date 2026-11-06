"""Tests for the CLI entry point ``python -m tensorsph`` (__main__.py).

The module-level ``argparse`` parser and the run/exit logic are exercised
without actually launching a full simulation (which would be slow).  We
test argument parsing, version output, and the error-handling paths.
"""
from __future__ import annotations

import subprocess
import sys



def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Invoke ``python -m tensorsph`` with the given extra arguments."""
    return subprocess.run(
        [sys.executable, "-m", "tensorsph"] + args,
        capture_output=True, text=True, timeout=30,
    )


# --------------------------------------------------------------------------- #
# Positive cases
# --------------------------------------------------------------------------- #

def test_version_flag():
    """``--version`` prints the version and exits 0."""
    result = _run_cli(["--version"])
    assert result.returncode == 0
    assert "tensorsph" in result.stdout


def test_help_flag():
    """``--help`` prints usage and exits 0."""
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "--output" in result.stdout
    assert "--steps" in result.stdout
    assert "--dt" in result.stdout
    assert "--boundary" in result.stdout
    assert "--device" in result.stdout
    assert "--no-plot" in result.stdout


def test_no_plot_flag_parsed():
    """``--no-plot`` is accepted and does not crash the parser."""
    result = _run_cli(["--no-plot", "--steps", "1"])
    # With only 1 step the simulation finishes quickly; we just check it
    # doesn't fail on argument parsing.
    assert result.returncode in (0, 1)  # 1 = unstable (acceptable for 1 step)


def test_boundary_choices():
    """Only 'wall' and 'clamp' are accepted for --boundary."""
    for choice in ("wall", "clamp"):
        result = _run_cli(["--boundary", choice, "--steps", "1"])
        assert result.returncode in (0, 1)


def test_device_auto():
    """``--device auto`` is accepted."""
    result = _run_cli(["--device", "auto", "--steps", "1"])
    assert result.returncode in (0, 1)


def test_zero_steps_completes_immediately():
    """Zero steps completes without error (no integration performed)."""
    result = _run_cli(["--steps", "0"])
    assert result.returncode == 0
    assert "Completed 0 steps" in result.stdout


# --------------------------------------------------------------------------- #
# Negative cases (error injection must turn red)
# --------------------------------------------------------------------------- #

def test_invalid_boundary_rejected():
    """An unknown --boundary value causes argparse to exit non-zero."""
    result = _run_cli(["--boundary", "invalid"])
    assert result.returncode != 0
    assert "invalid" in result.stderr.lower() or "error" in result.stderr.lower()


def test_non_integer_steps_rejected():
    """``--steps abc`` is rejected by argparse."""
    result = _run_cli(["--steps", "abc"])
    assert result.returncode != 0


def test_non_float_dt_rejected():
    """``--dt xyz`` is rejected by argparse."""
    result = _run_cli(["--dt", "xyz"])
    assert result.returncode != 0
