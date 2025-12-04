from __future__ import annotations

import numpy as np
import pytest

from AEIC.performance_model import (
    PerformanceConfig,
    PerformanceInputMode,
    PerformanceModel,
)


def test_performance_model_initialization():
    """PerformanceModel builds config, missions, and performance tables."""

    model = PerformanceModel('IO/default_config.toml')

    assert isinstance(model.config, PerformanceConfig)
    assert (
        model.config.performance_model_input is PerformanceInputMode.PERFORMANCE_MODEL
    )
    assert model.config.missions_folder == 'missions'
    assert model.config.missions_in_file.endswith('.toml')

    assert isinstance(model.missions, list) and len(model.missions) > 0

    assert hasattr(model, 'ac_params')
    assert model.ac_params.cas_cruise_lo == pytest.approx(
        model.model_info['speeds']['cruise']['cas_lo']
    )
    assert getattr(model, 'engine_model', None) is not None

    assert isinstance(model.LTO_data, dict)
    assert model.LTO_data['ICAO_UID'] == '01P11CM121'

    assert isinstance(model.performance_table, np.ndarray)
    assert model.performance_table.ndim == 4
    assert model.performance_table_colnames == ['FL', 'TAS', 'ROCD', 'MASS']

    dimension_lengths = tuple(len(col) for col in model.performance_table_cols)
    assert model.performance_table.shape == dimension_lengths


def test_performance_config_from_mapping_requires_sections():
    base = {
        'Missions': {'missions_folder': 'missions', 'missions_in_file': 'sample.toml'},
        'General Information': {
            'performance_model_input': 'opf',
            'performance_model_input_file': 'data/perf.toml',
        },
        'Emissions': {
            'EDB_input_file': 'engines/example.edb',
            'LTO_input_mode': 'performance_model',
        },
    }
    config = PerformanceConfig.from_mapping(base)
    assert config.performance_model_input is PerformanceInputMode.OPF
    assert config.performance_model_input_file == 'data/perf.toml'
    assert config.edb_input_file == 'engines/example.edb'

    for missing_key in ('Missions', 'General Information', 'Emissions'):
        incomplete = dict(base)
        incomplete.pop(missing_key)
        with pytest.raises(ValueError):
            PerformanceConfig.from_mapping(incomplete)


def test_create_performance_table_maps_multi_dimensional_grid():
    model = PerformanceModel.__new__(PerformanceModel)
    cols = ['FL', 'FUEL_FLOW', 'TAS', 'ROCD']
    rows = []
    for fl in (330, 350):
        for tas in (220, 240):
            for rocd in (-500, 0):
                fuel_flow = round(
                    0.5 + 0.001 * fl + 0.0001 * tas + 0.00001 * abs(rocd), 6
                )
                rows.append([fl, fuel_flow, tas, rocd])
    model.create_performance_table({'cols': cols, 'data': rows})

    fl_values, tas_values, rocd_values = model.performance_table_cols
    assert fl_values == [330, 350]
    assert tas_values == [220, 240]
    assert rocd_values == [-500, 0]

    fl_idx = fl_values.index(350)
    tas_idx = tas_values.index(240)
    rocd_idx = rocd_values.index(0)
    expected = 0.5 + 0.001 * 350 + 0.0001 * 240 + 0.00001 * 0
    assert model.performance_table[fl_idx, tas_idx, rocd_idx] == pytest.approx(expected)


def test_create_performance_table_missing_output_column():
    model = PerformanceModel.__new__(PerformanceModel)
    data = {'cols': ['FL', 'TAS'], 'data': [[330, 220]]}
    with pytest.raises(ValueError, match="FUEL_FLOW column not found"):
        model.create_performance_table(data)


def test_get_engine_by_uid_reads_matching_engine(tmp_path, monkeypatch):
    engine_file = tmp_path / 'engines.toml'
    engine_file.write_text(
        """
        [[engine]]
        UID = "MATCH"
        thrust = 42
        bypass = 10.5

        [[engine]]
        UID = "OTHER"
        thrust = 99
        """
    )
    monkeypatch.setattr(
        'AEIC.performance_model.file_location', lambda path: str(engine_file)
    )
    model = PerformanceModel.__new__(PerformanceModel)
    match = model.get_engine_by_uid('MATCH', 'ignored')
    assert match['thrust'] == 42
    assert model.get_engine_by_uid('MISSING', 'ignored') is None
