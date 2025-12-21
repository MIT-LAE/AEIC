from __future__ import annotations

import numpy as np
import pytest

from AEIC.config import config
from AEIC.performance import PerformanceModel, TablePerformanceModel
from AEIC.performance.table import PerformanceTable


def test_performance_model_initialization():
    """PerformanceModel builds config, and performance tables."""

    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, TablePerformanceModel)

    assert model.lto_performance is not None
    assert model.lto_performance.ICAO_UID == '01P11CM121'

    # These will go when the performance table is refactored.
    fp = model.flight_performance
    assert fp._performance_table is not None
    assert fp._performance_table_cols is not None
    assert fp._performance_table_colnames is not None
    assert isinstance(fp._performance_table, np.ndarray)
    assert fp._performance_table.ndim == 4
    assert fp._performance_table_colnames == ['fl', 'tas', 'rocd', 'mass']

    dimension_lengths = tuple(len(col) for col in fp._performance_table_cols)
    assert fp._performance_table.shape == dimension_lengths


def test_create_performance_table_maps_multi_dimensional_grid():
    cols = ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS']
    rows = []
    for fl in (330, 350):
        for tas in (220, 240):
            for rocd in (-500, 0):
                for mass in (60000, 70000):
                    fuel_flow = round(
                        0.5 + 0.001 * fl + 0.0001 * tas + 0.00001 * abs(rocd), 6
                    )
                    rows.append([fl, fuel_flow, tas, rocd, mass])
    model = PerformanceTable(cols=cols, data=rows)
    assert model._performance_table is not None
    assert model._performance_table_cols is not None

    fl_values, tas_values, rocd_values, mass_values = model._performance_table_cols
    assert fl_values == [330, 350]
    assert tas_values == [220, 240]
    assert rocd_values == [-500, 0]
    assert mass_values == [60000, 70000]

    expect = 0.5 + 0.001 * 350 + 0.0001 * 240 + 0.00001 * 0
    assert model._performance_table[
        fl_values.index(350), tas_values.index(240), rocd_values.index(0), 0
    ] == pytest.approx(expect)


def test_create_performance_table_missing_output_column():
    with pytest.raises(
        ValueError, match='Missing required "fuel_flow" column in performance table'
    ):
        _ = PerformanceTable(cols=['FL', 'TAS'], data=[[330, 220]])
