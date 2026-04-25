from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import pytest

from AEIC.config import config
from AEIC.performance.model_selector import SimplePerformanceModelSelector
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel
from AEIC.performance.models.legacy import ROCDFilter
from AEIC.performance.types import AircraftState, SimpleFlightRules, ThrustMode


def test_performance_model_initialization():
    """PerformanceModel builds config, and performance tables."""

    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, LegacyPerformanceModel)

    assert model.lto_performance.ICAO_UID == '01P11CM121'
    # Positive content check on the converted LTO data — a regression that
    # left LTO un-loaded would surface as fuel_flow[IDLE] == 0.0, which the
    # `is not None` smoke this replaces could not have caught.
    assert model.lto.fuel_flow[ThrustMode.IDLE] > 0

    # Cruise (ROCDFilter.ZERO) is the phase exercised by the
    # SimpleFlightRules.CRUISE evaluate call below; pin its axes.
    table = model.performance_table(ROCDFilter.ZERO)
    assert table.fl
    assert table.mass

    # End-to-end sanity check on the table-was-built-and-wired contract:
    # cruise evaluation at min mass should return finite, positive
    # airspeed and fuel flow.
    state = AircraftState(altitude=1828.8, aircraft_mass='min')  # FL 60
    perf = model.evaluate(state, SimpleFlightRules.CRUISE)
    assert math.isfinite(perf.true_airspeed) and perf.true_airspeed > 0
    assert math.isfinite(perf.fuel_flow) and perf.fuel_flow > 0


def test_performance_model_selection(performance_model_selector, sample_missions):
    pms = [performance_model_selector(m).aircraft_name for m in sample_missions]
    assert pms == [
        'B738',
        'B738',
        'B738',
        'B738',
        'B738',
        'A380',
        'A380',
        'B738',
        'B738',
        'A380',
    ]


def _real_default_pm() -> Path:
    """A real performance-model TOML usable as the `default` entry in
    selector-error tests that need a parseable default before the
    error-under-test fires."""
    return config.file_location('performance/simple_selector/738.toml')


def test_simple_selector_init_missing_directory(tmp_path):
    with pytest.raises(ValueError, match='directory does not exist'):
        SimplePerformanceModelSelector(tmp_path / 'does_not_exist')


def test_simple_selector_init_missing_config(tmp_path):
    with pytest.raises(ValueError, match='does not contain config.toml'):
        SimplePerformanceModelSelector(tmp_path)


def test_simple_selector_init_synonym_referencing_nonexistent_file(tmp_path, caplog):
    # On main this guard is a logger.warning, not a raise — selector
    # construction succeeds and records a warning the operator can act
    # on. Pin both: construction does not raise, and the warning carries
    # the offending file name.
    shutil.copy(_real_default_pm(), tmp_path / '738.toml')
    (tmp_path / 'config.toml').write_text('default = 738\n"319" = "NONEXISTENT"\n')
    with caplog.at_level(logging.WARNING, logger='AEIC.performance.model_selector'):
        SimplePerformanceModelSelector(tmp_path)
    assert any(
        'NONEXISTENT' in r.message and 'does not exist' in r.message
        for r in caplog.records
    )


def test_simple_selector_init_missing_default(tmp_path, caplog):
    # On main this guard is a logger.warning, not a raise — the selector
    # tolerates a missing default (returning None for unknown aircraft
    # types) but logs so the misconfiguration is visible.
    shutil.copy(_real_default_pm(), tmp_path / '738.toml')
    (tmp_path / 'config.toml').write_text('"319" = 738\n')  # no default
    with caplog.at_level(logging.WARNING, logger='AEIC.performance.model_selector'):
        SimplePerformanceModelSelector(tmp_path)
    assert any('"default" entry' in r.message for r in caplog.records)


def test_simple_selector_caches_repeated_lookups(
    performance_model_selector, sample_missions
):
    """Same-aircraft-type lookups must return the cached instance — the
    LRU cache hit path. Without this the selector re-parses the TOML on
    every dispatch.
    """
    same = [
        m
        for m in sample_missions
        if m.aircraft_type == sample_missions[0].aircraft_type
    ]
    assert len(same) >= 2
    assert performance_model_selector(same[0]) is performance_model_selector(same[1])
