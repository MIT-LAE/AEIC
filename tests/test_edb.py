import pandas as pd
import pytest

from AEIC.config import config
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import ThrustMode, ThrustModeValues


class DummyExcelFile:
    """Lightweight stand-in for pandas.ExcelFile to avoid real I/O in tests."""

    def __init__(self, gaseous_df: pd.DataFrame, nvpm_df: pd.DataFrame):
        self._gaseous_df = gaseous_df
        self._nvpm_df = nvpm_df
        self.sheet_names = ['Gaseous Emissions and Smoke', 'nvPM Emissions']

    def parse(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name == 'Gaseous Emissions and Smoke':
            return self._gaseous_df
        if sheet_name == 'nvPM Emissions':
            return self._nvpm_df
        raise ValueError(f"Unknown sheet requested: {sheet_name}")


def test_get_EDB_data_for_engine_raises_when_uid_absent(tmp_path, monkeypatch):
    gaseous = pd.DataFrame({'UID No': ['100']})
    nvpm = pd.DataFrame({'UID No': ['200']})

    dummy_path = tmp_path / "edb.xlsx"
    dummy_path.touch()

    monkeypatch.setattr(
        "AEIC.performance.edb.pd.ExcelFile",
        lambda _: DummyExcelFile(gaseous, nvpm),
    )

    with pytest.raises(
        ValueError, match="UID 300 not found in sheet 'Gaseous Emissions and Smoke'."
    ):
        EDBEntry.get_engine(dummy_path, uid="300")


def test_get_EDB_data_for_engine_returns_engine_data():
    UID = "01P11CM121"

    engine_info = EDBEntry.get_engine(
        config.file_location('engines/sample_edb.xlsx'), UID
    )

    assert engine_info.engine == "CFM56-7B27E"
    assert engine_info.uid == UID
    assert engine_info.engine_type == "TF"
    assert engine_info.BP_Ratio == 5.1
    assert engine_info.fuel_flow == ThrustModeValues(
        {
            ThrustMode.IDLE: 0.11,
            ThrustMode.APPROACH: 0.343,
            ThrustMode.CLIMB: 1.031,
            ThrustMode.TAKEOFF: 1.293,
        }
    )
    assert engine_info.CO_EI_matrix == ThrustModeValues(
        {
            ThrustMode.IDLE: 29.39,
            ThrustMode.APPROACH: 2.82,
            ThrustMode.CLIMB: 0.17,
            ThrustMode.TAKEOFF: 0.31,
        }
    )
    assert engine_info.HC_EI_matrix == ThrustModeValues(
        {
            ThrustMode.IDLE: 1.54,
            ThrustMode.APPROACH: 0.05,
            ThrustMode.CLIMB: 0.02,
            ThrustMode.TAKEOFF: 0.03,
        }
    )
    assert engine_info.EI_NOx_matrix == ThrustModeValues(
        {
            ThrustMode.IDLE: 4.36,
            ThrustMode.APPROACH: 9.09,
            ThrustMode.CLIMB: 17.89,
            ThrustMode.TAKEOFF: 23.94,
        }
    )
    assert engine_info.SN_matrix == ThrustModeValues(
        {
            ThrustMode.IDLE: 2.1,
            ThrustMode.APPROACH: 2.1,
            ThrustMode.CLIMB: 11.2,
            ThrustMode.TAKEOFF: 13.4,
        }
    )
    assert engine_info.nvPM_mass_matrix == ThrustModeValues(
        {
            ThrustMode.IDLE: 0.74,
            ThrustMode.APPROACH: 1.72,
            ThrustMode.CLIMB: 44.0,
            ThrustMode.TAKEOFF: 70.8,
        }
    )
    assert engine_info.nvPM_num_matrix == ThrustModeValues(
        {
            ThrustMode.IDLE: 26600000000000.0,
            ThrustMode.APPROACH: 71000000000000.0,
            ThrustMode.CLIMB: 433000000000000.0,
            ThrustMode.TAKEOFF: 402000000000000.0,
        }
    )
    assert engine_info.PR == ThrustModeValues(
        {
            ThrustMode.IDLE: 29.0,
            ThrustMode.APPROACH: 29.0,
            ThrustMode.CLIMB: 29.0,
            ThrustMode.TAKEOFF: 29.0,
        }
    )
    assert engine_info.EImass_max == 70.8
    assert engine_info.EInum_max == 433000000000000.0
