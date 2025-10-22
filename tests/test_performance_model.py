import numpy as np

from AEIC.performance_model import PerformanceModel


def test_performance_model_initialization():
    """Test initialization and basic structure of the PerformanceModel."""

    model = PerformanceModel('IO/default_config.toml')

    # Test keys in config
    assert isinstance(model.config, dict)
    assert "performance_model_input" in model.config

    # Test missions
    assert isinstance(model.missions, list)
    assert model.missions[0].load_factor == 1.0

    # Check aircraft parameters were loaded
    assert hasattr(model, "ac_params")
    assert model.ac_params.cas_cruise_lo == 128.611

    # Check engine model initialized
    assert hasattr(model, "engine_model")

    # Check LTO data
    assert isinstance(model.LTO_data, dict)
    assert model.LTO_data['ICAO_UID'] == '01P11CM121'

    # Check performance table exists
    assert isinstance(model.performance_table, np.ndarray)
    assert (
        model.performance_table.shape == (26, 51, 105, 3)
        or model.performance_table.ndim == 4
    )

    # Check input column names and values
    assert "FL" in model.performance_table_colnames
    assert len(model.performance_table_cols) == len(model.performance_table_colnames)
