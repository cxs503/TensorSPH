"""Uncertainty accounting and window bookkeeping for the Langevin LJ thermostat.

Guards the machinery added for
``docs/verification_fluctuating_hydro_thermostat.md``: the ledger reading
``fluctuating_hydro.thermostat.mean_temperature.n128`` (+8.81 %) is a startup
transient whose significance can only be judged against a block-averaged
standard error, so the error estimator and the discarded-window bookkeeping are
themselves testable objects.

The tests assert invariants (analytic SE for i.i.d. data, zero for a constant
series, window lengths that must match the samples actually collected) rather
than merely executing the code path.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import pytest
import torch

from tensorsph.fluctuating_hydro import (
    LJFluidConfig,
    block_average_se,
    run_lj_fluid,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO / "benchmarks" / "fluctuating_hydro" / "thermostat_audit.json"
AUDIT_SHA = REPO / "benchmarks" / "fluctuating_hydro" / "thermostat_audit.sha256"


class TestBlockAverageSE:
    def test_matches_analytic_se_for_iid_gaussian(self):
        """For i.i.d. data SE -> std/sqrt(N); block averaging must agree."""
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(8000, generator=gen, dtype=torch.float64)
        mean, se, blocks = block_average_se(x)
        analytic = float(x.std(unbiased=True)) / math.sqrt(x.numel())
        assert blocks >= 2
        assert abs(mean - float(x.mean())) < 1e-12
        assert abs(se / analytic - 1.0) < 0.25, (se, analytic)

    def test_constant_series_has_zero_error(self):
        mean, se, blocks = block_average_se([0.7] * 100)
        assert mean == pytest.approx(0.7)
        assert se == 0.0
        assert blocks >= 2

    def test_too_short_series_reports_nan(self):
        _mean, se, blocks = block_average_se([1.0, 2.0, 3.0])
        assert blocks == 0
        assert math.isnan(se)

    def test_block_count_is_honoured_and_clamped(self):
        x = [float(i % 5) for i in range(50)]
        _m, se6, blocks6 = block_average_se(x, n_blocks=6)
        _m, se10, blocks10 = block_average_se(x, n_blocks=10)
        assert (blocks6, blocks10) == (6, 10)
        # Asking for more blocks than samples must clamp, not crash.
        _m, _se, blocks = block_average_se([1.0, 2.0, 3.0, 4.0], n_blocks=99)
        assert blocks == 4

    def test_correlated_series_gives_larger_error_than_naive(self):
        """An AR(1) series with rho = 0.9: block SE must beat std/sqrt(N)."""
        gen = torch.Generator().manual_seed(11)
        n, rho = 4000, 0.9
        noise = torch.randn(n, generator=gen, dtype=torch.float64)
        x = torch.zeros(n, dtype=torch.float64)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + noise[i]
        _m, se, _b = block_average_se(x)
        naive = float(x.std(unbiased=True)) / math.sqrt(n)
        assert se > 2.0 * naive, (se, naive)


class TestEquilibrationWindow:
    def test_window_split_is_the_configured_fraction(self):
        cfg = LJFluidConfig(n_particles=64, n_steps=300, sample_every=10,
                            equilibration_fraction=0.5)
        res = run_lj_fluid(cfg)
        assert res.equilibration_steps == 150
        assert res.production_steps == 150
        # One sample every 10 steps of production, at steps 0, 10, ..., 140.
        assert res.n_samples_temperature == 15
        assert len(res.temperature_samples) == 15
        assert res.mean_temperature == pytest.approx(
            sum(res.temperature_samples) / 15, rel=1e-12)
        assert math.isfinite(res.mean_temperature_se)
        assert res.mean_temperature_se > 0.0
        assert 14 <= len(res.equilibration_temperature) <= 16

    def test_historical_default_is_unchanged(self):
        """The stored artifact was produced with a 40 % discarded window."""
        cfg = LJFluidConfig(n_particles=64, n_steps=200, sample_every=10)
        res = run_lj_fluid(cfg)
        assert res.equilibration_steps == 80
        assert res.production_steps == 120

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_invalid_fraction_raises(self, fraction):
        cfg = LJFluidConfig(n_particles=64, n_steps=100, equilibration_fraction=fraction)
        with pytest.raises(ValueError, match="equilibration_fraction"):
            run_lj_fluid(cfg)


@pytest.mark.slow
class TestStoredReadingIsReproduced:
    def test_800_step_run_reproduces_the_ledger_number(self):
        """The audit must discuss the same run the ledger quotes."""
        cfg = LJFluidConfig(n_particles=128, n_steps=800, sample_every=10, seed=42)
        res = run_lj_fluid(cfg)
        assert res.mean_temperature == pytest.approx(1.0880601294532628, abs=1e-6)
        # ... and that reading is above the set point, while the converged window
        # is below it: the transient is not a small perturbation of the answer.
        assert res.mean_temperature > 1.0
        assert (res.mean_temperature - 1.0) / res.mean_temperature_se > 1.0


class TestAuditArtifact:
    """The committed artifact is the evidence behind the verification record."""

    def _payload(self):
        if not AUDIT_JSON.exists():
            pytest.skip("audit artifact not generated in this checkout")
        return json.loads(AUDIT_JSON.read_text())

    def test_hash_matches_the_committed_digest(self):
        payload = self._payload()
        recorded = AUDIT_SHA.read_text().split()[0]
        assert hashlib.sha256(AUDIT_JSON.read_bytes()).hexdigest() == recorded
        assert payload["replay"]["mean_temperature"] == pytest.approx(1.0880601294532628, abs=1e-6)

    def test_audit_verdicts_record_the_open_defect(self):
        """The artifact records an honest FAIL, not a desired PASS.

        ``B``/``C`` (window and ensemble inside 3 sigma of the set point) must stay
        FAIL until the steady-state offset is explained and removed -- flipping
        them without new evidence is what this test exists to prevent.
        """
        verdicts = self._payload()["verdicts"]
        assert verdicts["A_replay_reproduces_stored_artifact"] is True
        assert verdicts["E_se_plateau_stable_across_block_counts"] is True
        assert verdicts["B_long_window_enters_3se_band"] is False
        assert verdicts["C_ensemble_consistent_with_target"] is False

    def test_ensemble_records_a_significant_cold_bias(self):
        """The converged state is below the set point at >3 sigma, not above it."""
        ens = self._payload()["ensemble_summary"]
        assert ens["n_seeds"] >= 4
        assert ens["inverse_variance_combined"] < 1.0
        assert ens["dev_over_se"] < -3.0

    def test_stored_reading_is_not_a_thermostat_measurement(self):
        """Guards the claim: the short window reports the transient, not the answer."""
        payload = self._payload()
        replay = payload["replay"]
        assert replay["n_samples"] < 60
        # The transient sits *above* the set point, the converged state below it.
        assert replay["mean_temperature"] > 1.0
        assert payload["ladder"][-1]["mean_temperature"] < 1.0
        assert payload["ensemble_summary"]["mean_of_seed_means"] < 1.0