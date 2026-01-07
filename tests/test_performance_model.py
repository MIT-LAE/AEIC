from __future__ import annotations

from AEIC.config import config
from AEIC.performance.models import LegacyPerformanceModel, PerformanceModel


def test_performance_model_initialization():
    """PerformanceModel builds config, and performance tables."""

    model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )
    assert isinstance(model, LegacyPerformanceModel)

    assert model.lto_performance is not None
    assert model.lto_performance.ICAO_UID == '01P11CM121'

    # TODO: Add tests for performance table.
