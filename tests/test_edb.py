import pandas as pd
import pytest

from AEIC.config import config
from AEIC.performance.types import LTOThrustMode
from AEIC.performance.utils.edb import EDBEntry


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
        "AEIC.performance.utils.edb.pd.ExcelFile",
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
    assert engine_info.fuel_flow == {
        LTOThrustMode.IDLE: 0.11,
        LTOThrustMode.APPROACH: 0.343,
        LTOThrustMode.CLIMB: 1.031,
        LTOThrustMode.TAKEOFF: 1.293,
    }
    assert engine_info.CO_EI_matrix == {
        LTOThrustMode.IDLE: 29.39,
        LTOThrustMode.APPROACH: 2.82,
        LTOThrustMode.CLIMB: 0.17,
        LTOThrustMode.TAKEOFF: 0.31,
    }
    assert engine_info.HC_EI_matrix == {
        LTOThrustMode.IDLE: 1.54,
        LTOThrustMode.APPROACH: 0.05,
        LTOThrustMode.CLIMB: 0.02,
        LTOThrustMode.TAKEOFF: 0.03,
    }
    assert engine_info.EI_NOx_matrix == {
        LTOThrustMode.IDLE: 4.36,
        LTOThrustMode.APPROACH: 9.09,
        LTOThrustMode.CLIMB: 17.89,
        LTOThrustMode.TAKEOFF: 23.94,
    }
    assert engine_info.SN_matrix == {
        LTOThrustMode.IDLE: 2.1,
        LTOThrustMode.APPROACH: 2.1,
        LTOThrustMode.CLIMB: 11.2,
        LTOThrustMode.TAKEOFF: 13.4,
    }
    assert engine_info.nvPM_mass_matrix == {
        LTOThrustMode.IDLE: 0.74,
        LTOThrustMode.APPROACH: 1.72,
        LTOThrustMode.CLIMB: 44.0,
        LTOThrustMode.TAKEOFF: 70.8,
    }
    assert engine_info.nvPM_num_matrix == {
        LTOThrustMode.IDLE: 26600000000000.0,
        LTOThrustMode.APPROACH: 71000000000000.0,
        LTOThrustMode.CLIMB: 433000000000000.0,
        LTOThrustMode.TAKEOFF: 402000000000000.0,
    }
    assert engine_info.PR == {
        LTOThrustMode.IDLE: 29.0,
        LTOThrustMode.APPROACH: 29.0,
        LTOThrustMode.CLIMB: 29.0,
        LTOThrustMode.TAKEOFF: 29.0,
    }
    assert engine_info.SNmax == {
        LTOThrustMode.IDLE: 13.38,
        LTOThrustMode.APPROACH: 13.38,
        LTOThrustMode.CLIMB: 13.38,
        LTOThrustMode.TAKEOFF: 13.38,
    }
    assert engine_info.EImass_max == 70.8
    assert engine_info.EInum_max == 433000000000000.0
