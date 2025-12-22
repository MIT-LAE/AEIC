from __future__ import annotations

import numpy as np
import pytest

from AEIC.config import config
from AEIC.performance import PerformanceModel, TablePerformanceModel
from AEIC.performance.table import PerformanceTable, PerformanceTableInput


def test_performance_model_initialization():
    """PerformanceModel builds config, and performance tables."""

    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, TablePerformanceModel)

    assert model.lto_performance is not None
    assert model.lto_performance.ICAO_UID == '01P11CM121'

    # These will go when the performance table is refactored.
    pt = model.performance_table
    assert isinstance(pt._old_table, np.ndarray)
    assert pt._old_table.ndim == 4


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
    model = PerformanceTable(PerformanceTableInput(cols=cols, data=rows))

    assert model.fl == [330, 350]
    assert model.tas == [220, 240]
    assert model.rocd == [-500, 0]
    assert model.mass == [60000, 70000]

    expect = 0.5 + 0.001 * 350 + 0.0001 * 240 + 0.00001 * 0
    assert model._old_table[
        model.fl.index(350), model.tas.index(240), model.rocd.index(0), 0
    ] == pytest.approx(expect)


def test_create_performance_table_missing_output_column():
    with pytest.raises(
        ValueError, match='Missing required "fuel_flow" column in performance table'
    ):
        _ = PerformanceTableInput(cols=['FL', 'TAS'], data=[[330, 220]])
