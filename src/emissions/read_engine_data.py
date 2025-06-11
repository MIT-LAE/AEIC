import os
import warnings
import numpy as np
import pandas as pd

def read_engine_info(src_folder,
                     eqp_input,
                     lto_input,
                     only_fuel=False,
                     pmnvol_switch_LTO='FOA3'):
    """
    Reads and processes engine information, converting the MATLAB script logic into Python.

    Parameters
    ----------
    src_folder : str
        Base directory where the CSV files are located.
    eqp_input : str
        Filename for the Aircraft-Engine assignment CSV (e.g., 'EQP_ENG_2022_newENGINEnames.csv').
    lto_input : str
        Filename for the Engine-EI assignment CSV (e.g., 'ENG_EI_UIDs_modeSN_2022_newENGINEnames.csv').
    only_fuel : bool, optional
        If True, only read fuel flow data and return early.
    pmnvol_switch_LTO : str, optional
        Method for nvPM calculations. Options include 'FOA3', 'NewSNCI', 'SCOPE11', 'FOX', 'dop', 'SST', 'FOX_alt'.

    Returns
    -------
    dict
        A dictionary containing processed arrays and DataFrames corresponding to:
        - EQP, ENGINE, NUM, ENGINE_TYPE, BP_Ratio, WNSF, NATSGRP, WVCAT, FUEL, APU_list
        - fuelflow_KGperS
        - (If not only_fuel) CO_EI_matrix, HC_EI_matrix, NOX_EI_matrix, SN_matrix,
          nvPM_mass_matrix, nvPM_num_matrix, UID, PR, SNmax, EImass_max, EImass_max_thrust,
          EInum_max, EInum_max_thrust,
          PMnvolEI_best_ICAOthrust, PMnvolEI_upper_ICAOthrust, PMnvolEI_lower_ICAOthrust (when applicable),
          PMnvolEIN_best_ICAOthrust, PMnvolEIN_upper_ICAOthrust, PMnvolEIN_lower_ICAOthrust (for 'SCOPE11' when applicable),
          PMnvolEI_new_ICAOthrust (for FOX/dop/SST/FOX_alt methods).
    """

    # --- 1) Read Aircraft‐Engine assignment (EQP_ENG_*.csv) ---
    eqp_path = os.path.join(src_folder, eqp_input)
    if not os.path.isfile(eqp_path):
        warnings.warn(f"\nCould not find file for aircraft-engine pairing {eqp_input}.")
        eqp_path = os.path.join(src_folder, 'EQP_ENG_2022_newENGINEnames.csv')
        warnings.warn(f"\nSetting to {eqp_path}.\n")

    # Load with pandas; header row is assumed to contain exactly:
    # EQP, ENGINE, NUM, ENGINE_TYPE, BP_Ratio, WNSF, NATSGRP, WVCAT, FUEL, APU
    df_eqp = pd.read_csv(eqp_path, header=0, dtype=str, na_values=['', 'NaN'])
    df_eqp['NUM'] = pd.to_numeric(df_eqp['NUM'], errors='coerce')
    df_eqp['B/P ratio'] = pd.to_numeric(df_eqp['B/P ratio'], errors='coerce')
    df_eqp['NATSGRP'] = pd.to_numeric(df_eqp['NATSGRP'], errors='coerce')

    EQP = df_eqp['EQUIPMENT'].astype(str).values
    ENGINE = df_eqp['ENGINE'].astype(str).values
    NUM = df_eqp['NUM'].values
    ENGINE_TYPE = df_eqp['Eng type'].astype(str).values
    BP_Ratio = df_eqp['B/P ratio'].values
    WNSF = df_eqp['W/N/S/F'].astype(str).values
    NATSGRP = df_eqp['NATSGRP'].values
    WVCAT = df_eqp['WV CAT'].astype(str).values
    FUEL = df_eqp['FUEL'].astype(str).values
    APU_list = df_eqp['APU'].astype(str).values  # renamed so it doesn’t shadow any Python built-ins

    # --- 2) Read Engine–EI assignment (ENG_EI_UIDs_*.csv) ---
    eng_ei_path = os.path.join(src_folder, lto_input)
    if not os.path.isfile(eng_ei_path):
        warnings.warn(f"\nCould not find file for LTO emission characteristics {lto_input}.")
        eng_ei_path = os.path.join(src_folder, 'ENG_EI_UIDs_modeSN_2022_newENGINEnames.csv')
        warnings.warn(f"\nSetting to {eng_ei_path}.\n")

    df_eng_ei = pd.read_csv(eng_ei_path, header=0, na_values=['', 'NaN'])
    # Rename columns so they match MATLAB variables:
    df_eng_ei = df_eng_ei.rename(columns={
        'ENG_NAME': 'ENG_NAME',
        'MODE': 'MODE',
        'CO_EI': 'CO_EI',
        'HC_EI': 'HC_EI',
        'NOX_EI': 'NOX_EI',
        'SOX_EI': 'SOX_EI',
        'SMOKE_NUM': 'SMOKE_NUM',
        'FUEL_KG/S': 'FUEL_KGperS',
        'ICAO_UID': 'UID_ref',
        'MODE_SN': 'MODE_SN',
        'BC_EI_mass': 'BC_EI_mass',
        'BC_EI_num': 'BC_EI_num'
    })

    # If only_fuel==True, we only need (ENG_NAME, MODE, FUEL_KGperS); otherwise read everything
    if only_fuel:
        df_eng_ei_small = df_eng_ei[['ENG_NAME', 'MODE', 'FUEL_KGperS']].copy()
    else:
        df_eng_ei_small = df_eng_ei[['ENG_NAME', 'MODE', 'CO_EI', 'HC_EI', 'NOX_EI',
                                     'FUEL_KGperS', 'UID_ref', 'SMOKE_NUM',
                                     'BC_EI_mass', 'BC_EI_num']].copy()
    df_eng_ei = None  # free memory

    df_eng_ei_small['MODE'] = pd.to_numeric(df_eng_ei_small['MODE'], errors='coerce').astype(int)

    # Remap MODE values exactly as in MATLAB:
    #   old 4 → 1, old 1 → 2, old 2 → 3, old 3 → 4
    mode_map = {4: 1, 1: 2, 2: 3, 3: 4}
    df_eng_ei_small['MODE_orig'] = df_eng_ei_small['MODE']
    df_eng_ei_small['MODE'] = df_eng_ei_small['MODE_orig'].map(mode_map).fillna(df_eng_ei_small['MODE_orig']).astype(int)

    # --- 3) Build modepositions[i,:] for each ENGINE[i] ---
    # “modepositions” is an (n_engines × 4) integer array of row‐indices in df_eng_ei_small.
    n_engines = len(ENGINE)
    modepositions = np.zeros((n_engines, 4), dtype=int)
    ename_series = df_eng_ei_small['ENG_NAME'].values

    for i, eng_name in enumerate(ENGINE):
        matches = np.where(ename_series == eng_name)[0]
        if len(matches) != 4:
            warnings.warn(f"Expected 4 LTO‐mode rows for engine “{eng_name}”, found {len(matches)}.")
            # If fewer than 4, pad with –1; if more, take the first 4
            if len(matches) < 4:
                padded = np.concatenate([matches, -np.ones(4 - len(matches), dtype=int)])
                modepositions[i, :] = padded
            else:
                modepositions[i, :] = matches[:4]
        else:
            modepositions[i, :] = matches

    # --- 4) Build fuelflow_KGperS matrix (n_engines × 4) ---
    fuelflow_KGperS = np.zeros((n_engines, 4))
    for i in range(n_engines):
        for j in range(4):
            idx = modepositions[i, j]
            if idx >= 0:
                fuelflow_KGperS[i, j] = df_eng_ei_small.iloc[idx]['FUEL_KGperS']
            else:
                fuelflow_KGperS[i, j] = np.nan
    # Circular shift to reorder columns → Idle(7%) at column 0, Approach(30%) at 1, Climb‐out(85%) at 2, Take‐off(100%) at 3
    fuelflow_KGperS = np.roll(fuelflow_KGperS, shift=1, axis=1)

    # --- 5) Read APU_EI.csv (APU emission indices) ---
    apu_ei_path = os.path.join(src_folder, 'APU_EI.csv')
    if os.path.isfile(apu_ei_path):
        df_apu = pd.read_csv(apu_ei_path, header=0, na_values=['', 'NaN'])
        # Expected columns:
        #   “APU” (string), “Defra APU” (string), “Fuel (kg/s)”, “Nox (g/kg)”, “CO (g/kg)”, “HC (g/kg)”, “PM10 (g/kg)”
        df_apu = df_apu.rename(columns={
            'APU': 'APU_name_ref',
            'Fuel (kg/s)': 'APU_fuelflow_ref',
            'Nox (g/kg)': 'APU_NOxEI_ref',
            'CO (g/kg)': 'APU_COEI_ref',
            'HC (g/kg)': 'APU_HCEI_ref',
            'PM10 (g/kg)': 'APU_PM10EI_ref'
        })
        APU_name_ref = df_apu['APU_name_ref'].astype(str).values
        APU_fuelflow_ref = pd.to_numeric(df_apu['APU_fuelflow_ref'], errors='coerce').values
        APU_NOxEI_ref = pd.to_numeric(df_apu['APU_NOxEI_ref'], errors='coerce').values
        APU_COEI_ref = pd.to_numeric(df_apu['APU_COEI_ref'], errors='coerce').values
        APU_HCEI_ref = pd.to_numeric(df_apu['APU_HCEI_ref'], errors='coerce').values
        APU_PM10EI_ref = pd.to_numeric(df_apu['APU_PM10EI_ref'], errors='coerce').values
    else:
        APU_name_ref = np.array([])
        APU_fuelflow_ref = np.array([])
        APU_NOxEI_ref = np.array([])
        APU_COEI_ref = np.array([])
        APU_HCEI_ref = np.array([])
        APU_PM10EI_ref = np.array([])

    # If only_fuel==True, return immediately (matching the MATLAB “if onlyFuel” block).
    if only_fuel:
        return {
            'EQP': EQP,
            'ENGINE': ENGINE,
            'NUM': NUM,
            'ENGINE_TYPE': ENGINE_TYPE,
            'BP_Ratio': BP_Ratio,
            'WNSF': WNSF,
            'NATSGRP': NATSGRP,
            'WVCAT': WVCAT,
            'FUEL': FUEL,
            'APU_list': APU_list,
            'fuelflow_KGperS': fuelflow_KGperS,
            'APU_name_ref': APU_name_ref,
            'APU_fuelflow_ref': APU_fuelflow_ref,
            'APU_NOxEI_ref': APU_NOxEI_ref,
            'APU_COEI_ref': APU_COEI_ref,
            'APU_HCEI_ref': APU_HCEI_ref,
            'APU_PM10EI_ref': APU_PM10EI_ref
        }

    # --- 6) Build CO_EI_matrix, HC_EI_matrix, NOX_EI_matrix (each n_engines × 4) ---
    CO_EI_matrix = np.zeros((n_engines, 4))
    HC_EI_matrix = np.zeros((n_engines, 4))
    NOX_EI_matrix = np.zeros((n_engines, 4))

    for i in range(n_engines):
        for j in range(4):
            idx = modepositions[i, j]
            if idx >= 0:
                co_val = df_eng_ei_small.iloc[idx]['CO_EI']
                CO_EI_matrix[i, j] = co_val if not np.isnan(co_val) else 0.1
                hc_val = df_eng_ei_small.iloc[idx]['HC_EI']
                HC_EI_matrix[i, j] = hc_val if not np.isnan(hc_val) else 0.1
                nox_val = df_eng_ei_small.iloc[idx]['NOX_EI']
                NOX_EI_matrix[i, j] = nox_val if not np.isnan(nox_val) else 0.01
            else:
                CO_EI_matrix[i, j] = 0.1
                HC_EI_matrix[i, j] = 0.1
                NOX_EI_matrix[i, j] = 0.01

    # Circular‐shift columns to match “Idle, Approach, Climb, Takeoff”
    CO_EI_matrix = np.roll(CO_EI_matrix, 1, axis=1)
    HC_EI_matrix = np.roll(HC_EI_matrix, 1, axis=1)
    NOX_EI_matrix = np.roll(NOX_EI_matrix, 1, axis=1)

    # Replace any exact zeros with the small non‐zero to avoid log(0) later
    CO_EI_matrix[CO_EI_matrix == 0] = 0.1
    HC_EI_matrix[HC_EI_matrix == 0] = 0.1
    NOX_EI_matrix[NOX_EI_matrix == 0] = 0.01

    # --- 7) Build smoke number and nvPM matrices (n_engines × 4) ---
    SN_matrix = np.zeros((n_engines, 4))
    nvPM_mass_matrix = np.zeros((n_engines, 4))
    nvPM_num_matrix = np.zeros((n_engines, 4))

    for i in range(n_engines):
        for j in range(4):
            idx = modepositions[i, j]
            if idx >= 0:
                SN_matrix[i, j] = df_eng_ei_small.iloc[idx]['SMOKE_NUM']
                nvPM_mass_matrix[i, j] = df_eng_ei_small.iloc[idx]['BC_EI_mass']
                nvPM_num_matrix[i, j] = df_eng_ei_small.iloc[idx]['BC_EI_num']
            else:
                SN_matrix[i, j] = -1
                nvPM_mass_matrix[i, j] = -1
                nvPM_num_matrix[i, j] = -1

    SN_matrix = np.roll(SN_matrix, 1, axis=1)
    nvPM_mass_matrix = np.roll(nvPM_mass_matrix, 1, axis=1)
    nvPM_num_matrix = np.roll(nvPM_num_matrix, 1, axis=1)

    # --- 8) Build UID list (ICAO ID) ---
    UID = np.empty((n_engines,), dtype=object)
    for i in range(n_engines):
        idx = modepositions[i, 0]
        if idx >= 0:
            UID[i] = df_eng_ei_small.iloc[idx]['UID_ref']
        else:
            UID[i] = '-'

    # --- 9) Load SNmax & PR data from “UID_PR_SNmax_newENGINEnames.csv” ---
    snmax_path = os.path.join(src_folder, 'UID_PR_SNmax_newENGINEnames.csv')
    df_snmax = pd.read_csv(snmax_path, header=0, na_values=['', 'NaN']).rename(columns={
        'UID': 'UID_icao',
        'PR': 'PR_icao',
        'SN_max': 'SNmax_icao',
        'EI_mass_max': 'EImass_max_icao',
        'EI_mass_max_thrust': 'EImass_max_thrust_icao',
        'EI_num_max': 'EInum_max_icao',
        'EI_num_max_thrust': 'EInum_max_thrust_icao'
    })

    PR = np.zeros_like(NOX_EI_matrix)
    SNmax = np.zeros_like(NOX_EI_matrix)
    EImass_max = np.zeros((n_engines,))
    EImass_max_thrust = np.zeros((n_engines,))
    EInum_max = np.zeros((n_engines,))
    EInum_max_thrust = np.zeros((n_engines,))

    for i in range(n_engines):
        if UID[i] == '-' or len(UID[i].strip()) == 0:
            continue
        matches = df_snmax.index[df_snmax['UID_icao'] == UID[i]].tolist()
        if len(matches) == 0:
            continue
        idx_sn = matches[0]
        pr_val = df_snmax.loc[idx_sn, 'PR_icao']
        snmax_val = df_snmax.loc[idx_sn, 'SNmax_icao']
        PR[i, :] = pr_val
        SNmax[i, :] = snmax_val
        EImass_max[i] = df_snmax.loc[idx_sn, 'EImass_max_icao']
        EImass_max_thrust[i] = df_snmax.loc[idx_sn, 'EImass_max_thrust_icao']
        EInum_max[i] = df_snmax.loc[idx_sn, 'EInum_max_icao']
        EInum_max_thrust[i] = df_snmax.loc[idx_sn, 'EInum_max_thrust_icao']

    # --- 10) Compute FOA3 / NewSNCI / SCOPE11 → CI & CI_upper (n_engines × 4) ---
    ICAO_thrust = np.array([0, 7, 30, 85, 100])  # Extra “0%” column will be handled later
    FoverF00 = np.array([0.07, 0.30, 0.85, 1.0])
    AFR = np.array([106, 83, 51, 45])

    CI_best = np.zeros_like(SN_matrix)
    CI_upper = np.zeros_like(SN_matrix)
    CI_lower = np.zeros_like(SN_matrix) if pmnvol_switch_LTO == 'SCOPE11' else None
    GMD = np.zeros_like(SN_matrix) if pmnvol_switch_LTO == 'SCOPE11' else None
    GMD_up = np.zeros_like(SN_matrix) if pmnvol_switch_LTO == 'SCOPE11' else None
    GMD_do = np.zeros_like(SN_matrix) if pmnvol_switch_LTO == 'SCOPE11' else None

    if pmnvol_switch_LTO == 'SCOPE11':
        import scipy.io as sio
        DataGMD = sio.loadmat(os.path.join(src_folder, 'GMD_CI.mat'))
        Datakslm = sio.loadmat(os.path.join(src_folder, 'kslm_CI.mat'))
        DataSCOPE = sio.loadmat(os.path.join(src_folder, 'SCOPE11_CI.mat'))

    for i in range(n_engines):
        for j in range(4):
            sn = SN_matrix[i, j]
            if pmnvol_switch_LTO in ['FOA3', 'NewSNCI']:
                # FOA3 or NewSNCI piecewise fits
                if sn <= 0:
                    CI_best[i, j] = 0
                    CI_upper[i, j] = 0
                elif (sn > 0) and (sn <= 30):
                    CI_best[i, j] = 0.0694 * sn ** 1.24
                    CI_upper[i, j] = 0.0012 * sn ** 2 + 0.1312 * sn + 0.2255
                else:
                    CI_best[i, j] = 0.236 * sn ** 1.126
                    CI_upper[i, j] = 0.0297 * sn ** 2 - 1.6238 * sn + 26.801

            elif pmnvol_switch_LTO == 'SCOPE11':
                # SCOPE11 median, upper, lower CI + GMD logic
                if sn < 0:
                    CI_best[i, j] = 0
                    CI_upper[i, j] = 0
                    CI_lower[i, j] = 0
                    GMD[i, j] = 20
                    GMD_up[i, j] = 20
                    GMD_do[i, j] = 20
                else:
                    SN_limited = min(sn, 40)
                    P4_LTO = 101325 * (1 + (PR[i, j] - 1) * FoverF00[j])
                    T3_LTO = 288.15 * (PR[i, j] ** 0.3175)
                    T4_LTO = (AFR[j] * 1.005 * T3_LTO + 43.2e3) / (1.250 * (1 + AFR[j]))
                    RHO4_LTO = P4_LTO / (287.058 * T4_LTO)

                    CBC_i = 0.6484 * np.exp(0.0766 * SN_limited) / (1 + np.exp(-1.098 * (SN_limited - 3.064)))
                    if ENGINE_TYPE[i] == 'MTF':
                        kslm = np.log((3.219 * CBC_i * (1 + BP_Ratio[i]) * 1000 + 312.5) /
                                      (CBC_i * (1 + BP_Ratio[i]) * 1000 + 42.6))
                    else:
                        kslm = np.log((3.219 * CBC_i * 1000 + 312.5) / (CBC_i * 1000 + 42.6))

                    CI_best[i, j] = kslm * CBC_i
                    if ENGINE_TYPE[i] == 'MTF':
                        CBC_c = CI_best[i, j] * (1 + BP_Ratio[i]) * RHO4_LTO / 1.20
                    else:
                        CBC_c = CI_best[i, j] * RHO4_LTO / 1.20
                    GMD[i, j] = 5.08 * (CBC_c * 1000) ** 0.185

                    CBC_i_up = np.interp(SN_limited,
                                         DataSCOPE['SN'].flatten(),
                                         DataSCOPE['conf_int_up'].flatten()) * 1e-3
                    if ENGINE_TYPE[i] == 'MTF':
                        kslm_up = np.interp(
                            min((1 + BP_Ratio[i]) * CBC_i_up * 1000, Datakslm['kslm'].flatten()),
                            Datakslm['kslm'].flatten(),
                            Datakslm['conf_int_up'].flatten()
                        )
                    else:
                        kslm_up = np.interp(CBC_i_up * 1000,
                                           Datakslm['kslm'].flatten(),
                                           Datakslm['conf_int_up'].flatten())
                    CI_upper[i, j] = kslm_up * CBC_i_up
                    if ENGINE_TYPE[i] == 'MTF':
                        CBC_c_up = CI_upper[i, j] * (1 + BP_Ratio[i]) * RHO4_LTO / 1.20
                    else:
                        CBC_c_up = CI_upper[i, j] * RHO4_LTO / 1.20
                    GMD_up[i, j] = np.interp(CBC_c_up * 1000,
                                             DataGMD['CBC_core'].flatten(),
                                             DataGMD['conf_int_up'].flatten())

                    CBC_i_do = np.interp(SN_limited,
                                         DataSCOPE['SN'].flatten(),
                                         DataSCOPE['conf_int_do'].flatten()) * 1e-3
                    if ENGINE_TYPE[i] == 'MTF':
                        kslm_do = np.interp(
                            min((1 + BP_Ratio[i]) * CBC_i_do * 1000, Datakslm['kslm'].flatten()),
                            Datakslm['kslm'].flatten(),
                            Datakslm['conf_int_do'].flatten()
                        )
                    else:
                        kslm_do = np.interp(CBC_i_do * 1000,
                                           Datakslm['kslm'].flatten(),
                                           Datakslm['conf_int_do'].flatten())
                    CI_lower[i, j] = kslm_do * CBC_i_do
                    if ENGINE_TYPE[i] == 'MTF':
                        CBC_c_do = CI_lower[i, j] * (1 + BP_Ratio[i]) * RHO4_LTO / 1.20
                    else:
                        CBC_c_do = CI_lower[i, j] * RHO4_LTO / 1.20
                    GMD_do[i, j] = np.interp(CBC_c_do * 1000,
                                             DataGMD['CBC_core'].flatten(),
                                             DataGMD['conf_int_do'].flatten())
            else:
                # For FOX / dop / SST / FOX_alt, handle later
                CI_best[i, j] = 0
                CI_upper[i, j] = 0
                if pmnvol_switch_LTO == 'SCOPE11':
                    CI_lower[i, j] = 0
                    GMD[i, j] = 20
                    GMD_up[i, j] = 20
                    GMD_do[i, j] = 20

    # --- 11) Build Q matrix (m^3/kg_fuel) for each thrust mode (Idle,30%,85%,100%) ---
    Q = np.zeros((n_engines, 4))
    for i in range(n_engines):
        for j in range(4):
            if ENGINE_TYPE[i] == 'MTF':
                Q[i, j] = 0.776 * AFR[j] * (1 + BP_Ratio[i]) + 0.767
            elif ENGINE_TYPE[i] == 'TF':
                Q[i, j] = 0.776 * AFR[j] + 0.767
            else:
                Q[i, j] = 0

    # --- 12) Compute PMnvolEI (g/kg_fuel) from CI * Q (mg/kg * m^3/kg → mg/kg), then /1000 → g/kg ---
    PMnvolEI_best = (CI_best * Q) / 1000.0
    PMnvolEI_upper = (CI_upper * Q) / 1000.0

    # Preallocate the “ICAO thrust (5 columns)” arrays by adding a “0%” column at index 0
    PMnvolEI_best_ICAOthrust = np.zeros((n_engines, 5))
    PMnvolEI_upper_ICAOthrust = np.zeros((n_engines, 5))
    PMnvolEI_lower_ICAOthrust = np.zeros((n_engines, 5)) if pmnvol_switch_LTO == 'SCOPE11' else None

    PMnvolEI_best_ICAOthrust[:, 0] = PMnvolEI_best[:, 0]
    PMnvolEI_best_ICAOthrust[:, 1:] = PMnvolEI_best
    PMnvolEI_upper_ICAOthrust[:, 0] = PMnvolEI_upper[:, 0]
    PMnvolEI_upper_ICAOthrust[:, 1:] = PMnvolEI_upper

    if pmnvol_switch_LTO == 'SCOPE11':
        PMnvolEI_lower = (CI_lower * Q) / 1000.0
        PMnvolEI_lower_ICAOthrust[:, 0] = PMnvolEI_lower[:, 0]
        PMnvolEI_lower_ICAOthrust[:, 1:] = PMnvolEI_lower

    # --- 13) If using SCOPE11, compute PM number emissions (PMnvolEIN_*) ---
    if pmnvol_switch_LTO == 'SCOPE11':
        PMnvolEIN_best_ICAOthrust = np.zeros((n_engines, 5))
        PMnvolEIN_upper_ICAOthrust = np.zeros((n_engines, 5))
        PMnvolEIN_lower_ICAOthrust = np.zeros((n_engines, 5))

        denom_factor = np.pi * 1000 * np.exp(4.5 * (np.log(1.8) ** 2))  # constant factor

        for i in range(n_engines):
            for k in range(5):
                gmd = GMD[i, k - 1] if k > 0 else GMD[i, 0]
                gmd_up = GMD_up[i, k - 1] if k > 0 else GMD_up[i, 0]
                gmd_do = GMD_do[i, k - 1] if k > 0 else GMD_do[i, 0]
                emi_best = PMnvolEI_best_ICAOthrust[i, k]
                emi_up = PMnvolEI_upper_ICAOthrust[i, k]
                emi_do = PMnvolEI_lower_ICAOthrust[i, k]
                emi_best_kg = emi_best * 1e-3
                emi_up_kg = emi_up * 1e-3
                emi_do_kg = emi_do * 1e-3

                if gmd > 0:
                    PMnvolEIN_best_ICAOthrust[i, k] = (6 * emi_best_kg) / (denom_factor * (gmd * 1e-9) ** 3)
                else:
                    PMnvolEIN_best_ICAOthrust[i, k] = 0

                if gmd_up > 0:
                    PMnvolEIN_upper_ICAOthrust[i, k] = (6 * emi_up_kg) / (denom_factor * (gmd_do * 1e-9) ** 3)
                else:
                    PMnvolEIN_upper_ICAOthrust[i, k] = 0

                if gmd_do > 0:
                    PMnvolEIN_lower_ICAOthrust[i, k] = (6 * emi_do_kg) / (denom_factor * (gmd_up * 1e-9) ** 3)
                else:
                    PMnvolEIN_lower_ICAOthrust[i, k] = 0
    else:
        PMnvolEIN_best_ICAOthrust = PMnvolEIN_upper_ICAOthrust = PMnvolEIN_lower_ICAOthrust = None

    # --- 14) If using FOX / dop / SST / FOX_alt methods, build PMnvolEI_new_ICAOthrust exactly as MATLAB did ---
    PMnvolEI_new_ICAOthrust = None
    if pmnvol_switch_LTO in ['FOX', 'dop', 'SST', 'FOX_alt']:
        # Build P3_LTO and T3_LTO for each engine & thrust mode
        # P3_LTO = (PR - 1)*(ICAO_thrust[1:]/100) + 1, shape (n_engines × 4)
        P3_LTO = (PR - 1) * np.reshape((ICAO_thrust[1:] / 100), (1, 4)) + 1
        T3_LTO = 7.5 * (P3_LTO * 101325) ** 0.318
        Tfl_LTO = 0.9 * T3_LTO + 2120  # as per Equation (10)

        PMnvolEI_new_ICAOthrust = np.zeros((n_engines, 5))  # extra “0%” column

        if pmnvol_switch_LTO == 'FOX':
            Aform = 356
            Aoxid = 608
            Eform = 6390.0
            Eoxid = 19778.0
            AFRmat = 1.0 / (0.0121 * (ICAO_thrust[1:] / 100) + 0.0078)
            AFRmat = np.tile(AFRmat, (n_engines, 1))
            Qfox = 0.776 * AFRmat + 0.877
            Qfox[np.where(ENGINE_TYPE == 'TP')[0], :] = 0  # turboprops

            CBC = fuelflow_KGperS * (Aform * np.exp(-Eform / Tfl_LTO) -
                                     AFRmat * Aoxid * np.exp(-Eoxid / Tfl_LTO)) / 1000.0
            CBC[CBC < 0] = 0
            PMnvolEI_new = CBC * Qfox
            PMnvolEI_new_ICAOthrust[:, 0] = PMnvolEI_new[:, 0]
            PMnvolEI_new_ICAOthrust[:, 1:] = PMnvolEI_new

        elif pmnvol_switch_LTO == 'dop':
            Adop = 10
            thrust_matrix = np.tile(ICAO_thrust[1:], (n_engines, 1))
            AFRmat = 1.0 / (0.0121 * (thrust_matrix / 100) + 0.0078)
            Qdop = 0.776 * AFRmat + 0.877

            CBC = (Adop *
                   ((0.0121 * (thrust_matrix / 100) + 0.0078) / 0.02) ** 2.5 *
                   (((PR - 1) * (thrust_matrix / 100) + 1) ** 0.92) *
                   ((NOX_EI_matrix / 20) ** 0.52))
            CBC[CBC < 0] = 0
            PMnvolEI_new = CBC * Qdop / 1000.0
            PMnvolEI_new_ICAOthrust[:, 0] = PMnvolEI_new[:, 0]
            PMnvolEI_new_ICAOthrust[:, 1:] = PMnvolEI_new

        elif pmnvol_switch_LTO == 'SST':
            # SST uses a static 4‐vector
            SST_vals = np.array([0.0100, 0.0100, 0.0600, 0.1100])
            # Prepend an extra 0% column
            PMnvolEI_new_ICAOthrust[:, 0] = SST_vals[0]
            PMnvolEI_new_ICAOthrust[:, 1:] = SST_vals[np.newaxis, :]

        elif pmnvol_switch_LTO == 'FOX_alt':
            Aform = 278.2
            Aoxid = 161.7
            Eform = 6390.0
            Eoxid = 19778.0
            AFRmat = 1.0 / (0.0121 * (ICAO_thrust[1:] / 100) + 0.0078)
            AFRmat = np.tile(AFRmat, (n_engines, 1))
            Qfox_alt = 0.776 * AFRmat + 0.877
            Qfox_alt[np.where(ENGINE_TYPE == 'TP')[0], :] = 0

            CBC = fuelflow_KGperS * (Aform * np.exp(-Eform / Tfl_LTO) -
                                     AFRmat * Aoxid * np.exp(-Eoxid / Tfl_LTO)) / 1000.0
            CBC[CBC < 0] = 0
            PMnvolEI_new = CBC * Qfox_alt
            PMnvolEI_new_ICAOthrust[:, 0] = PMnvolEI_new[:, 0]
            PMnvolEI_new_ICAOthrust[:, 1:] = PMnvolEI_new

    # --- 15) Load any measured nvPM data (EIBC_measdata.csv) and overwrite ---
    meas_bc_path = os.path.join(src_folder, 'EIBC_measdata.csv')
    if os.path.isfile(meas_bc_path) and (pmnvol_switch_LTO in ['FOX', 'dop', 'SST', 'FOX_alt', 'SCOPE11']):
        df_meas = pd.read_csv(meas_bc_path, header=0, na_values=['', 'NaN'])
        meas_engines = df_meas.iloc[:, 0].astype(str).values
        meas_bc_values = df_meas.iloc[:, 1:].values  # subsequent columns are measured EIs
        for idx_row, eng_name in enumerate(meas_engines):
            rows = np.where(ENGINE == eng_name)[0]
            if len(rows) == 0:
                continue
            meas_vals = meas_bc_values[idx_row, :]
            meas_vals = meas_vals[meas_vals != -111]  # remove missing‐value markers
            num_vals = len(meas_vals)
            for r in rows:
                PMnvolEI_new_ICAOthrust[r, 1:1 + num_vals] = meas_vals / 1000.0

    # If using SCOPE11, adjust column 0 (0% thrust) to equal column 1 (7%)
    if pmnvol_switch_LTO == 'SCOPE11':
        PMnvolEI_best_ICAOthrust[:, 0] = PMnvolEI_best_ICAOthrust[:, 1]
        PMnvolEI_upper_ICAOthrust[:, 0] = PMnvolEI_upper_ICAOthrust[:, 1]
        PMnvolEI_lower_ICAOthrust[:, 0] = PMnvolEI_lower_ICAOthrust[:, 1]
        PMnvolEIN_best_ICAOthrust[:, 0] = PMnvolEIN_best_ICAOthrust[:, 1]
        PMnvolEIN_upper_ICAOthrust[:, 0] = PMnvolEIN_upper_ICAOthrust[:, 1]
        PMnvolEIN_lower_ICAOthrust[:, 0] = PMnvolEIN_lower_ICAOthrust[:, 1]

    # --- Final packaging: return everything in a single dict ---
    return {
        'EQP': EQP,
        'ENGINE': ENGINE,
        'NUM': NUM,
        'ENGINE_TYPE': ENGINE_TYPE,
        'BP_Ratio': BP_Ratio,
        'WNSF': WNSF,
        'NATSGRP': NATSGRP,
        'WVCAT': WVCAT,
        'FUEL': FUEL,
        'APU_list': APU_list,
        'fuelflow_KGperS': fuelflow_KGperS,
        'APU_name_ref': APU_name_ref,
        'APU_fuelflow_ref': APU_fuelflow_ref,
        'APU_NOxEI_ref': APU_NOxEI_ref,
        'APU_COEI_ref': APU_COEI_ref,
        'APU_HCEI_ref': APU_HCEI_ref,
        'APU_PM10EI_ref': APU_PM10EI_ref,
        'CO_EI_matrix': CO_EI_matrix,
        'HC_EI_matrix': HC_EI_matrix,
        'NOX_EI_matrix': NOX_EI_matrix,
        'SN_matrix': SN_matrix,
        'nvPM_mass_matrix': nvPM_mass_matrix,
        'nvPM_num_matrix': nvPM_num_matrix,
        'UID': UID,
        'PR': PR,
        'SNmax': SNmax,
        'EImass_max': EImass_max,
        'EImass_max_thrust': EImass_max_thrust,
        'EInum_max': EInum_max,
        'EInum_max_thrust': EInum_max_thrust,
        'PMnvolEI_best_ICAOthrust': PMnvolEI_best_ICAOthrust,
        'PMnvolEI_upper_ICAOthrust': PMnvolEI_upper_ICAOthrust,
        'PMnvolEI_lower_ICAOthrust': PMnvolEI_lower_ICAOthrust,
        'PMnvolEIN_best_ICAOthrust': PMnvolEIN_best_ICAOthrust,
        'PMnvolEIN_upper_ICAOthrust': PMnvolEIN_upper_ICAOthrust,
        'PMnvolEIN_lower_ICAOthrust': PMnvolEIN_lower_ICAOthrust,
        'PMnvolEI_new_ICAOthrust': PMnvolEI_new_ICAOthrust
    }
