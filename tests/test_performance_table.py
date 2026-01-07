# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import pytest

from AEIC.config import config
from AEIC.performance.models import PerformanceModel, TablePerformanceModel
from AEIC.performance.utils.performance_table import (
    PerformanceTable,
    PerformanceTableInput,
    ROCDFilter,
)


def test_create_performance_table():
    def fuel_flow(fl: int, tas: int, rocd: int) -> float:
        return round(0.5 + 0.001 * fl + 0.0001 * tas + 0.00001 * abs(rocd), 6)

    cols = ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS']
    rows = []
    for fl in (330, 350):
        for tas in (220, 240):
            for rocd in (-500, 0):
                for mass in (60000, 70000):
                    rows.append([fl, fuel_flow(fl, tas, rocd), tas, rocd, mass])
    model = PerformanceTable.from_input(PerformanceTableInput(cols=cols, data=rows))

    assert model.fl == [330, 350]
    assert model.tas == [220, 240]
    assert model.rocd == [-500, 0]
    assert model.mass == [60000, 70000]

    assert model.df[
        (model.df.fl == 350)
        & (model.df.tas == 240)
        & (model.df.rocd == 0)
        & (model.df.mass == 60000)
    ].fuel_flow.values[0] == fuel_flow(350, 240, 0)


def test_create_performance_table_missing_output_column():
    with pytest.raises(
        ValueError, match='Missing required "fuel_flow" column in performance table'
    ):
        _ = PerformanceTableInput(cols=['FL', 'TAS'], data=[[330, 220]])


def test_performance_table_subsetting():
    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, TablePerformanceModel)
    table = model.performance_table

    sub_table_1 = table.subset(rocd=ROCDFilter.POSITIVE)
    assert len(sub_table_1.fl) <= len(table.fl)
    assert len(sub_table_1.mass) <= len(table.mass)
    assert all(rocd > 0 for rocd in sub_table_1.rocd)

    sub_table_2 = table.subset(rocd=ROCDFilter.NEGATIVE)
    assert all(rocd < 0 for rocd in sub_table_2.rocd)
    assert len(sub_table_2.mass) <= len(table.mass)
