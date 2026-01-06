# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import tomllib

import pytest

from AEIC.config import config
from AEIC.performance.models import PerformanceModel, TablePerformanceModel
from AEIC.performance.utils.performance_table import (
    LegacyPerformanceTable,
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

    sub_table_1 = table.subset(fl=310, rocd=ROCDFilter.POSITIVE)
    assert len(sub_table_1.fl) == 2
    assert len(sub_table_1.mass) == len(table.mass)

    sub_table_2 = table.subset(rocd=ROCDFilter.NEGATIVE, mass='max')
    assert all(rocd < 0 for rocd in sub_table_2.rocd)
    assert len(sub_table_2.mass) == 1

    sub_table_3 = table.subset(mass=65000)
    assert len(sub_table_3.fl) == len(table.fl)
    assert len(sub_table_3.mass) == 2


def test_legacy_performance_table_ok():
    with open(
        config.file_location('performance/sample_performance_model.toml'), 'rb'
    ) as fp:
        data = tomllib.load(fp)

    ptin = PerformanceTableInput(
        data=data['flight_performance']['data'], cols=data['flight_performance']['cols']
    )
    tab1 = LegacyPerformanceTable.from_input(ptin)
    assert len(tab1.mass) == 3


def test_legacy_performance_table_bad(test_data_dir):
    for bad_case in range(1, 4):
        fname = test_data_dir / 'performance' / f'bad_performance_model_{bad_case}.toml'
        with open(fname, 'rb') as fp:
            data = tomllib.load(fp)
        ptin = PerformanceTableInput(
            data=data['flight_performance']['data'],
            cols=data['flight_performance']['cols'],
        )
        with pytest.raises(ValueError):
            tab = LegacyPerformanceTable.from_input(ptin)
            assert len(tab.mass) == 3
