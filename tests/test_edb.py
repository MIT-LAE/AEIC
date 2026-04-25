import re

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


@pytest.mark.parametrize(
    'gaseous_uids,nvpm_uids,query_uid,expected_sheet',
    [
        # UID absent from both sheets → gaseous-sheet check fires first.
        (['100'], ['200'], '300', 'Gaseous Emissions and Smoke'),
        # UID in gaseous but not in nvPM → nvPM-sheet check fires.
        (['100', '300'], ['200'], '300', 'nvPM Emissions'),
    ],
    ids=['absent_from_both', 'present_in_gaseous_only'],
)
def test_get_EDB_data_for_engine_raises_when_uid_absent(
    tmp_path, monkeypatch, gaseous_uids, nvpm_uids, query_uid, expected_sheet
):
    gaseous = pd.DataFrame({'UID No': gaseous_uids})
    nvpm = pd.DataFrame({'UID No': nvpm_uids})

    dummy_path = tmp_path / "edb.xlsx"
    dummy_path.touch()

    monkeypatch.setattr(
        "AEIC.performance.edb.pd.ExcelFile",
        lambda _: DummyExcelFile(gaseous, nvpm),
    )

    expected_msg = f"UID {query_uid} not found in sheet '{expected_sheet}'."
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        EDBEntry.get_engine(dummy_path, uid=query_uid)


_SAMPLE_EDB_UID = "01P11CM121"


@pytest.fixture
def sample_engine_info():
    """Parse `engines/sample_edb.xlsx` per parametrized case. Function
    scope is required because the autouse `default_config` fixture loads
    the AEIC `Config` per test, and `config.file_location` resolves the
    sheet path against that.
    """
    return EDBEntry.get_engine(
        config.file_location('engines/sample_edb.xlsx'), _SAMPLE_EDB_UID
    )


@pytest.mark.parametrize(
    'attr,expected',
    [
        ('engine', "CFM56-7B27E"),
        ('uid', _SAMPLE_EDB_UID),
        ('engine_type', "TF"),
        ('BP_Ratio', 5.1),
        (
            'fuel_flow',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 0.11,
                    ThrustMode.APPROACH: 0.343,
                    ThrustMode.CLIMB: 1.031,
                    ThrustMode.TAKEOFF: 1.293,
                }
            ),
        ),
        (
            'CO_EI_matrix',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 29.39,
                    ThrustMode.APPROACH: 2.82,
                    ThrustMode.CLIMB: 0.17,
                    ThrustMode.TAKEOFF: 0.31,
                }
            ),
        ),
        (
            'HC_EI_matrix',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 1.54,
                    ThrustMode.APPROACH: 0.05,
                    ThrustMode.CLIMB: 0.02,
                    ThrustMode.TAKEOFF: 0.03,
                }
            ),
        ),
        (
            'EI_NOx_matrix',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 4.36,
                    ThrustMode.APPROACH: 9.09,
                    ThrustMode.CLIMB: 17.89,
                    ThrustMode.TAKEOFF: 23.94,
                }
            ),
        ),
        (
            'SN_matrix',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 2.1,
                    ThrustMode.APPROACH: 2.1,
                    ThrustMode.CLIMB: 11.2,
                    ThrustMode.TAKEOFF: 13.4,
                }
            ),
        ),
        (
            'nvPM_mass_matrix',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 0.74,
                    ThrustMode.APPROACH: 1.72,
                    ThrustMode.CLIMB: 44.0,
                    ThrustMode.TAKEOFF: 70.8,
                }
            ),
        ),
        (
            'nvPM_num_matrix',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 26600000000000.0,
                    ThrustMode.APPROACH: 71000000000000.0,
                    ThrustMode.CLIMB: 433000000000000.0,
                    ThrustMode.TAKEOFF: 402000000000000.0,
                }
            ),
        ),
        (
            'PR',
            ThrustModeValues(
                {
                    ThrustMode.IDLE: 29.0,
                    ThrustMode.APPROACH: 29.0,
                    ThrustMode.CLIMB: 29.0,
                    ThrustMode.TAKEOFF: 29.0,
                }
            ),
        ),
        ('EImass_max', 70.8),
        ('EInum_max', 433000000000000.0),
    ],
)
def test_get_EDB_data_for_engine_returns_engine_data(
    sample_engine_info, attr, expected
):
    """Each attribute on the parsed EDB entry is checked individually so
    a regression in (e.g.) the nvPM-num column extraction lands on its
    own test ID rather than failing the first `assert` in a wall.
    """
    assert getattr(sample_engine_info, attr) == expected
