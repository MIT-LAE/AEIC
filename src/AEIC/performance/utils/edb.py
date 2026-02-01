# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from AEIC.config import config
from AEIC.types import LTOPerformance, ModeValues, ThrustMode


@dataclass
class EDBEntry:
    engine: str
    uid: str
    engine_type: str
    BP_Ratio: float
    rated_thrust: float
    fuel_flow: ModeValues
    CO_EI_matrix: ModeValues
    HC_EI_matrix: ModeValues
    EI_NOx_matrix: ModeValues
    SN_matrix: ModeValues
    nvPM_mass_matrix: ModeValues
    nvPM_num_matrix: ModeValues
    PR: ModeValues
    EImass_max: float
    EImass_max_thrust: float
    EInum_max: float
    EInum_max_thrust: float

    def __hash__(self) -> int:
        return hash(self.engine + ':' + self.uid)

    def make_lto_performance(self, thrust_fractions: list[float]) -> LTOPerformance:
        return LTOPerformance(
            source='EDB',
            ICAO_UID=self.uid,
            rated_thrust=self.rated_thrust * 1000.0,
            thrust_pct=ModeValues(
                {m: t * 100 for m, t in zip(ThrustMode, thrust_fractions)}
            ),
            fuel_flow=self.fuel_flow,
            EI_NOx=self.EI_NOx_matrix,
            EI_HC=self.HC_EI_matrix,
            EI_CO=self.CO_EI_matrix,
        )

    @classmethod
    def get_engine(cls, excel_file: Path, uid: str) -> EDBEntry:
        """Reads the EDB Excel workbook and returns dict with EDB engine data
        for UID given, combining data from the "Gaseous Emissions and Smoke"
        and "nvPM Emissions" sheets."""
        try:
            xls = pd.ExcelFile(excel_file)
        except Exception as exc:
            raise ValueError(
                f"Unable to open EDB workbook at {config.edb_input_file}: {exc}"
            ) from exc

        gaseous_sheet = 'Gaseous Emissions and Smoke'
        nvpm_sheet = 'nvPM Emissions'

        missing_sheets = [
            sheet
            for sheet in (gaseous_sheet, nvpm_sheet)
            if sheet not in xls.sheet_names
        ]
        if missing_sheets:
            missing = ', '.join(missing_sheets)
            raise ValueError(f"EDB workbook is missing required sheets: {missing}")

        gaseous = xls.parse(gaseous_sheet)
        assert isinstance(gaseous, pd.DataFrame)
        nvpm = xls.parse(nvpm_sheet)
        assert isinstance(nvpm, pd.DataFrame)

        if 'UID No' not in gaseous.columns or 'UID No' not in nvpm.columns:
            raise ValueError("UID No column is missing from one or both sheets.")

        uid_str = str(uid)
        gaseous_uids = gaseous['UID No'].astype(str)
        nvpm_uids = nvpm['UID No'].astype(str)

        if uid_str not in set(gaseous_uids):
            raise ValueError(f"UID {uid_str} not found in sheet '{gaseous_sheet}'.")
        if uid_str not in set(nvpm_uids):
            raise ValueError(f"UID {uid_str} not found in sheet '{nvpm_sheet}'.")

        # Select the first matching row in each sheet (wide-format)
        g = gaseous[gaseous_uids == uid_str].iloc[0]
        n = nvpm[nvpm_uids == uid_str].iloc[0]

        # Define the four LTO modes in the desired order
        modes = [
            (ThrustMode.IDLE, 'Idle'),
            (ThrustMode.APPROACH, 'App'),
            (ThrustMode.CLIMB, 'C/O'),
            (ThrustMode.TAKEOFF, 'T/O'),
        ]

        # Extract data for each mode.
        def mode_dict(template: str, gaseous: bool = True) -> ModeValues:
            row = g if gaseous else n
            return ModeValues(
                {mode: float(row[template.format(mode=label)]) for mode, label in modes}
            )

        # Build a dict for this engine
        return cls(
            engine=g['Engine Identification'],
            uid=str(uid),
            engine_type=g['Eng Type'],
            BP_Ratio=float(g['B/P Ratio']),
            rated_thrust=float(g['Rated Thrust (kN)']),
            fuel_flow=mode_dict('Fuel Flow {mode} (kg/sec)'),
            CO_EI_matrix=mode_dict('CO EI {mode} (g/kg)'),
            HC_EI_matrix=mode_dict('HC EI {mode} (g/kg)'),
            EI_NOx_matrix=mode_dict('NOx EI {mode} (g/kg)'),
            SN_matrix=mode_dict('SN {mode}'),
            nvPM_mass_matrix=mode_dict('nvPM EImass {mode} (mg/kg)', gaseous=False),
            nvPM_num_matrix=mode_dict('nvPM EInum {mode} (#/kg)', gaseous=False),
            PR=mode_dict('Pressure Ratio'),
            EImass_max=float(n['nvPM EImass Max (mg/kg)']),
            EImass_max_thrust=-1.0,
            EInum_max=float(n['nvPM EInum Max (#/kg)']),
            EInum_max_thrust=-1.0,
        )
