import pandas as pd
import pytest

from AEIC.utils.read_EDB_data import get_EDB_data_for_engine


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


def test_get_EDB_data_for_engine_missing_file():
    with pytest.raises(FileNotFoundError):
        get_EDB_data_for_engine("does-not-exist.xlsx", "123")


def test_get_EDB_data_for_engine_raises_when_uid_absent(tmp_path, monkeypatch):
    gaseous = pd.DataFrame({'UID No': ['100']})
    nvpm = pd.DataFrame({'UID No': ['200']})

    dummy_path = tmp_path / "edb.xlsx"
    dummy_path.touch()

    monkeypatch.setattr(
        "AEIC.utils.read_EDB_data.pd.ExcelFile",
        lambda _: DummyExcelFile(gaseous, nvpm),
    )

    with pytest.raises(
        ValueError, match="UID 300 not found in sheet 'Gaseous Emissions and Smoke'."
    ):
        get_EDB_data_for_engine(str(dummy_path), uid="300")


def test_get_EDB_data_for_engine_returns_engine_data():
    UID = "01P11CM121"

    engine_info = get_EDB_data_for_engine("engines/sample_edb.xlsx", UID)

    assert engine_info['ENGINE'] == "CFM56-7B27E"
    assert engine_info['UID'] == UID
    assert engine_info['ENGINE_TYPE'] == "TF"
    assert engine_info['BP_Ratio'] == 5.1
    assert engine_info['fuelflow_KGperS'] == [0.11, 0.343, 1.031, 1.293]
    assert engine_info['CO_EI_matrix'] == [
        29.39,
        2.82,
        0.17,
        0.31,
    ]
    assert engine_info['HC_EI_matrix'] == [
        1.54,
        0.05,
        0.02,
        0.03,
    ]
    assert engine_info['NOX_EI_matrix'] == [
        4.36,
        9.09,
        17.89,
        23.94,
    ]
    assert engine_info['SN_matrix'] == [
        2.1,
        2.1,
        11.2,
        13.4,
    ]
    assert engine_info['nvPM_mass_matrix'] == [
        0.74,
        1.72,
        44.0,
        70.8,
    ]
    assert engine_info['nvPM_num_matrix'] == [
        26600000000000.0,
        71000000000000.0,
        433000000000000.0,
        402000000000000.0,
    ]
    assert engine_info['PR'] == [
        29.0,
        29.0,
        29.0,
        29.0,
    ]
    assert engine_info['SNmax'] == [
        13.38,
        13.38,
        13.38,
        13.38,
    ]
    assert engine_info['EImass_max'] == 70.8
    assert engine_info['EInum_max'] == 433000000000000.0
