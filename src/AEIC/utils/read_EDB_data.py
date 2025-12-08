import gc

import pandas as pd

from AEIC.utils.files import file_location


def get_EDB_data_for_engine(edb_path: str, uid: str) -> dict:
    """
    Reads the EDB Excel workbook and returns dict with
    EDB engine data for UID given,
    combining data from the "Gaseous Emissions and Smoke" and "nvPM Emissions" sheets.
    """
    edb_file = file_location(edb_path)

    try:
        xls = pd.ExcelFile(edb_file)
    except Exception as exc:
        raise ValueError(f"Unable to open EDB workbook at {edb_file}: {exc}") from exc

    gaseous_sheet = 'Gaseous Emissions and Smoke'
    nvpm_sheet = 'nvPM Emissions'

    missing_sheets = [
        sheet for sheet in (gaseous_sheet, nvpm_sheet) if sheet not in xls.sheet_names
    ]
    if missing_sheets:
        missing = ', '.join(missing_sheets)
        raise ValueError(f"EDB workbook is missing required sheets: {missing}")

    gaseous = xls.parse(gaseous_sheet)
    nvpm = xls.parse(nvpm_sheet)

    if 'UID No' not in gaseous.columns or 'UID No' not in nvpm.columns:
        raise ValueError("UID No column is missing from one or both sheets.")

    uid_str = str(uid)
    gaseous_uids = gaseous['UID No'].astype(str)
    nvpm_uids = nvpm['UID No'].astype(str)

    if uid_str not in set(gaseous_uids):
        raise ValueError(f"UID {uid_str} not found in sheet '{gaseous_sheet}'.")
    if uid_str not in set(nvpm_uids):
        raise ValueError(f"UID {uid_str} not found in sheet '{nvpm_sheet}'.")

    # Define the four LTO modes in the desired order
    modes = ['Idle', 'App', 'C/O', 'T/O']

    # Select the first matching row in each sheet (wide-format)
    g = gaseous[gaseous_uids == uid_str].iloc[0]
    n = nvpm[nvpm_uids == uid_str].iloc[0]

    # Build a dict for this engine
    entry = {
        'ENGINE': g['Engine Identification'],
        'UID': str(uid),
        'ENGINE_TYPE': g['Eng Type'],
        'BP_Ratio': float(g['B/P Ratio']),
        'fuelflow_KGperS': [float(g[f'Fuel Flow {mode} (kg/sec)']) for mode in modes],
        'CO_EI_matrix': [float(g[f'CO EI {mode} (g/kg)']) for mode in modes],
        'HC_EI_matrix': [float(g[f'HC EI {mode} (g/kg)']) for mode in modes],
        'NOX_EI_matrix': [float(g[f'NOx EI {mode} (g/kg)']) for mode in modes],
        'SN_matrix': [float(g[f'SN {mode}']) for mode in modes],
        'nvPM_mass_matrix': [float(n[f'nvPM EImass {mode} (mg/kg)']) for mode in modes],
        'nvPM_num_matrix': [float(n[f'nvPM EInum {mode} (#/kg)']) for mode in modes],
        'PR': [float(g['Pressure Ratio'])] * len(modes),
        'SNmax': [float(g['SN Max'])] * len(modes),
        'EImass_max': float(n['nvPM EImass Max (mg/kg)']),
        'EImass_max_thrust': -1.0,
        'EInum_max': float(n['nvPM EInum Max (#/kg)']),
        'EInum_max_thrust': -1.0,
    }
    del xls
    gc.collect()
    return entry
