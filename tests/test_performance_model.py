from __future__ import annotations

import math

from AEIC.config import config
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel
from AEIC.performance.models.legacy import ROCDFilter
from AEIC.performance.types import AircraftState, SimpleFlightRules


def test_performance_model_initialization():
    """PerformanceModel builds config, and performance tables."""

    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, LegacyPerformanceModel)

    assert model.lto_performance is not None
    assert model.lto_performance.ICAO_UID == '01P11CM121'

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
