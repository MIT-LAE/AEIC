# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import pytest

from AEIC.config import config
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel
from AEIC.performance.models.legacy import (
    PerformanceTable,
    PerformanceTableInput,
    ROCDFilter,
)


def test_create_performance_table():
    def fuel_flow(fl: int, tas: int, rocd: int) -> float:
        return round(0.5 + 0.001 * fl + 0.0001 * tas + 0.00001 * abs(rocd), 6)

    def tas(fl: int, rocd: int) -> int:
        if rocd > 0:
            return 200 + (fl - 300) // 10
        elif rocd < 0:
            return 240 + (fl - 300) // 10
        else:
            return 220 + (fl - 300) // 10

    cols = ['FL', 'FUEL_FLOW', 'TAS', 'ROCD', 'MASS']
    rows = []
    for fl in (330, 350):
        for rocd in (-500, 0, 500):
            for mass in (60000, 70000, 80000):
                v = tas(fl, rocd)
                ff = fuel_flow(fl, v, rocd)
                rows.append([fl, ff, v, rocd, mass])
    model = PerformanceTable.from_input(PerformanceTableInput(cols=cols, data=rows))

    assert model.fl == [330, 350]
    assert model.mass == [60000, 70000, 80000]

    assert model.df[
        (model.df.fl == 350)
        & (model.df.tas == tas(350, 0))
        & (model.df.rocd == 0)
        & (model.df.mass == 60000)
    ].fuel_flow.values[0] == fuel_flow(350, tas(350, 0), 0)


def test_create_performance_table_missing_output_column():
    with pytest.raises(
        ValueError, match='Missing required "fuel_flow" column in performance table'
    ):
        _ = PerformanceTableInput(cols=['FL', 'TAS'], data=[[330, 220]])


def test_performance_table_subsetting():
    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, LegacyPerformanceModel)
    table = model.performance_table

    sub_table_1 = table.subset(rocd=ROCDFilter.POSITIVE)
    assert len(sub_table_1.fl) <= len(table.fl)
    assert len(sub_table_1.mass) <= len(table.mass)
    assert all(rocd > 0 for rocd in sub_table_1.df.rocd)

    sub_table_2 = table.subset(rocd=ROCDFilter.NEGATIVE)
    assert all(rocd < 0 for rocd in sub_table_2.df.rocd)
    assert len(sub_table_2.mass) <= len(table.mass)
