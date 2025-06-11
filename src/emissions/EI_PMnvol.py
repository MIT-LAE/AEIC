import numpy as np
from src.utils.standard_fuel import get_fuel_factor
from src.utils.consts import kappa

def EI_PMnvol(
    fuel_flow_flight: np.ndarray,
    fuel_flow_performance_model: np.ndarray,
    PMnvolEI_ICAOthrust: np.ndarray
) -> np.ndarray:
    """
    Calculate EI(PMnvol) (black carbon) [g/kg fuel] via linear interpolation
    from reference thrust-specific values.

    Parameters
    ----------
    thrusts : ndarray, shape (n_types, n_times)
        ICAO thrust settings (%) for each mode and time.
    PMnvolEI_ICAOthrust : ndarray, shape (n_types, 5)
        Emissions indices at thrusts [7, 30, 85, 100] (%).
    mcs7, mcs30, mcs85 : ints
        Monte Carlo switches (unused, commented out in original).
    rv7, rv30, rv85 : floats
        Random variates (unused, commented out in original).

    Returns
    -------
    PMnvolEI : ndarray, shape (n_types, n_times)
        Interpolated black carbon emissions index [g/kg fuel].
    """
    # Define reference thrust levels
    # ICAO_thrust = np.array([7, 30, 85, 100], dtype=float)

    # Allocate output
    PMnvolEI = np.zeros_like(fuel_flow_flight)

    # Perform interpolation row-by-row
    PMnvolEI = np.interp(fuel_flow_flight, fuel_flow_performance_model, PMnvolEI_ICAOthrust)

    return PMnvolEI

def EI_PMnvolN(thrusts: np.ndarray,
                      PMnvolEIN_ICAOthrust: np.ndarray) -> np.ndarray:
    """
    Interpolate non-volatile particulate matter number-based EI (PMnvolEI_N)
    across ICAO thrust settings.

    Parameters
    ----------
    thrusts : ndarray, shape (n_types, n_times)
        ICAO thrust settings (%) for each mode and time.
    PMnvolEIN_ICAOthrust : ndarray, shape (n_types, 4)
        Reference EI values at thrusts [7, 30, 85, 100] (%).

    Returns
    -------
    PMnvolEI_N : ndarray, shape (n_types, n_times)
        Interpolated non-volatile PM EI [g/kg fuel].
    """
    # Define reference thrust levels
    ICAO_thrust = np.array([7, 30, 85, 100], dtype=float)
    PMnvolEI_N = np.interp(thrusts, ICAO_thrust, PMnvolEIN_ICAOthrust)

    return PMnvolEI_N


def PMnvol_MEEM(EDB_data,altitudes,Tamb_cruise,Pamb_cruise,machFlight,fuel_flow_cruise,pmnvolSwitch_cruise = 'SCOPE11'):
    # This implementation follows the implementation by Ahrens et al (2022)
    # The nvPM Mission Emissions Estimation Methodology (MEEM) 
    # which is based on SCOPE11, and Peck et al (2013)  

    # -----------------------------
    # 0. Get fuel factor (sea-level equivalent fuel flow):
    # -----------------------------
    fuelFactor = get_fuel_factor(fuel_flow_cruise, Pamb_cruise, machFlight)

    # [idle, approach, climb-out, take-off]
    # geometric mean diameter (nm)
    GMD_mode = np.array([20, 20, 40, 40])

    # Step 0: find EI from SN if necessary
    # Check if EI measurements for this aircraft exist
    SN_mat_tmp = EDB_data['SN_matrix']
    ismeasure = False

    if np.min(EDB_data['nvPM_mass_matrix']) < 0:
        # ground level SN to use
        if np.min(SN_mat_tmp) < 0:
            # use max SN
            SN_ref = np.full_like(SN_mat_tmp, np.max(SN_mat_tmp))
        else:
            # else use SN matrix
            SN_ref = SN_mat_tmp.copy()

        # apply the SMOKE11 correlation to find the unadjusted CImass (mg/m^3)
        CI_mass = np.zeros(4)
        for idx in range(4):
            CI_mass[idx] = (
                0.6484 * np.exp(0.0766 * SN_ref[idx])
                / (1 + np.exp(-1.098 * (SN_ref[idx] - 3.064)))
            )  # mg/m^3

        # define the air fuel ratio (AFR) according to Wayson (ratio)
        # [idle, approach, climb-out, take-off]
        AFR_mode = np.array([106, 83, 51, 45])

        # find appropriate bypass ratio for the engine
        eng_type = EDB_data['ENGINE_TYPE']
        if eng_type == 'MTF':
            bypass_ratio = EDB_data['BP_Ratio']
        elif eng_type == 'TF':
            bypass_ratio = 0.0
        else:
            bypass_ratio = 0.0

        # Find the thr volumetric flow rate (m^3/kg)
        Q_mode = 0.776 * AFR_mode * (1 + bypass_ratio) + 0.767

        # Find the adjustment factor kslm
        # CImass adjusted from mg/kg to microgram/kg in this calculation
        if (eng_type == 'MTF') or (eng_type == 'TF'):
            kslm = np.log(
                (3.219 * CI_mass * (1 + bypass_ratio) * 1000 + 312.5)
                / (CI_mass * (1 + bypass_ratio) * 1000 + 42.6)
            )
        else:
            kslm = np.zeros(4)

        # replace the calculated values into the measurements matrix
        # nvPM_mass_mode in mg/kg
        nvPM_mass_mode = CI_mass * Q_mode * kslm  # mg/kg
        ismeasure = True

    else:
        # EI from measurements (mg/kg)
        nvPM_mass_mode = EDB_data['nvPM_mass_matrix'].astype(float)


    # if not defined, do the number calculation
    if np.min(EDB_data['nvPM_num_matrix']) < 0:
        # do num calculation
        # find number emissions #/kg fuel
        nvPM_num_mode = (
            6.0
            * nvPM_mass_mode
            / (
                np.pi
                * 1e9
                * ((GMD_mode * 1.00e-9) ** 3)
                * np.exp(4.5 * (np.log(1.8) ** 2))
            )
        )
    else:
        nvPM_num_mode = EDB_data['nvPM_num_matrix'].astype(float)


    # Step 1: Determine In-flight thermodynamic conditions
    max_pressure_ratio = np.array(EDB_data['PR'])[0]
    gamma = kappa

    # first find combustor efficiencies as defined in Ahrens et al
    # Find if aircraft is in climb, cruise, or descent
    # altitudes   # 1D array
    alt_change = np.concatenate((np.diff(altitudes), np.zeros(1)))
    eta_comp = np.zeros_like(alt_change)
    eta_comp[alt_change >= 0] = 0.88
    eta_comp[alt_change < 0] = 0.7

    # find pressure coefficient variable
    pressure_coef = np.zeros_like(alt_change)
    max_alt = np.max(altitudes)
    lin_vary_alt = (altitudes - 3000) / (max_alt - 3000)
    pressure_coef[alt_change > 0] = 0.85 + (1.15 - 0.85) * lin_vary_alt[alt_change > 0]
    pressure_coef[alt_change == 0] = 0.95
    pressure_coef[alt_change < 0] = 0.12

    # These variables define the temperature and pressure at altitude
    # Tamb_cruise and Pamb_cruise are arrays defined elsewhere (BADA standard atmosphere)
    Ttamb_total = Tamb_cruise #* (1 + ((gamma - 1) / 2) * machFlight ** 2)
    Ptamb_total = Pamb_cruise #* (1 + ((gamma - 1) / 2) * machFlight ** 2) ** (gamma / (gamma - 1))

    # find conditions at the combustor inlet
    P3 = Ptamb_total * (1 + pressure_coef * (max_pressure_ratio - 1.0))
    T3 = Ttamb_total * (1 + (1.0 / eta_comp) * ((P3 / Ptamb_total) ** ((gamma - 1) / gamma) - 1))

    # Step 2: Find conditions at the ground/reference state
    P_ground = 101325.0  # Pa
    T_ground = 288.15  # K

    # reference values
    T3_ref = T3.copy()  # assume same as T3
    P3_ref = P_ground * (1 + eta_comp * (T3_ref / T_ground - 1)) ** (gamma / (gamma - 1))

    # find F_Foo_ref
    FG_over_Foo = (P3_ref / P_ground - 1) / (max_pressure_ratio - 1)


    # Step 3: Interpolation
    # Mass:
    if (np.isnan(EDB_data['EImass_max_thrust'])) or (EDB_data['EImass_max_thrust'] < 0):
        # Use four-point interpolation
        EI_nvPM_mass_interp = np.concatenate(
            ([nvPM_mass_mode[0]], nvPM_mass_mode, [nvPM_mass_mode[-1]])
        )
        thrust_EImass_interp = np.array([-10, 0.07, 0.3, 0.85, 1, 100])
    else:
        if EDB_data['EImass_max_thrust'] == 0.575:
            EI_nvPM_mass_interp = np.concatenate(
                (
                    [nvPM_mass_mode[0]],
                    nvPM_mass_mode[0:2],
                    [EDB_data['EImass_max']],
                    nvPM_mass_mode[2:],
                    [nvPM_mass_mode[-1]],
                )
            )
            thrust_EImass_interp = np.array([-10, 0.07, 0.3, 0.575, 0.85, 1, 100])
        elif EDB_data['EImass_max_thrust'] == 0.925:
            EI_nvPM_mass_interp = np.concatenate(
                (
                    [nvPM_mass_mode[0]],
                    nvPM_mass_mode[0:3],
                    [EDB_data['EImass_max']],
                    [nvPM_mass_mode[3]],
                    [nvPM_mass_mode[-1]],
                )
            )
            thrust_EImass_interp = np.array([-10, 0.07, 0.3, 0.85, 0.925, 1, 100])
        else:
            raise ValueError(" EImass_max_thrust not recognized")
        
    # Num:
    if (np.isnan(EDB_data['EInum_max_thrust'])) or (EDB_data['EInum_max_thrust'] < 0):
        # Use four-point interpolation
        EI_nvPM_num_interp = np.concatenate(
            ([nvPM_num_mode[0]], nvPM_num_mode, [nvPM_num_mode[-1]])
        )
        thrust_EInum_interp = np.array([-10, 0.07, 0.3, 0.85, 1, 100])
    else:
        if EDB_data['EInum_max_thrust'] == 0.575:
            EI_nvPM_num_interp = np.concatenate(
                (
                    [nvPM_num_mode[0]],
                    nvPM_num_mode[0:2],
                    [EDB_data['EInum_max']],
                    nvPM_num_mode[2:],
                    [nvPM_num_mode[-1]],
                )
            )
            thrust_EInum_interp = np.array([-10, 0.07, 0.3, 0.575, 0.85, 1, 100])
        elif EDB_data['EInum_max_thrust'] == 0.925:
            EI_nvPM_num_interp = np.concatenate(
                (
                    [nvPM_num_mode[0]],
                    nvPM_num_mode[0:3],
                    [EDB_data['EInum_max']],
                    [nvPM_num_mode[3]],
                    [nvPM_num_mode[-1]],
                )
            )
            thrust_EInum_interp = np.array([-10, 0.07, 0.3, 0.85, 0.925, 1, 100])
        else:
            raise ValueError("cruiseEmissions_byFlight: EInum_max_thrust not recognized")
        
    # prepare GMD for interpolation
    thrust_GMD_interp = np.array([-10, 0.07, 0.3, 0.85, 1, 100])
    GMD_interp = np.concatenate(([GMD_mode[0]], GMD_mode, [GMD_mode[-1]]))

    # now interpolate (mg/kg or #/kg)
    fg_over = FG_over_Foo  
    EI_ref_mass = np.interp(fg_over, thrust_EImass_interp, EI_nvPM_mass_interp)
    EI_ref_num = np.interp(fg_over, thrust_EInum_interp, EI_nvPM_num_interp)
    GMD_ref = np.interp(fg_over, thrust_GMD_interp, GMD_interp)

    EI_PMnvol_GMD = GMD_ref

    # Step 4: Now apply DoppelHeuer-Lecht to altitude (g/kg & #/kg)
    # also convert mg/kg to g/kg
    EI_PMnvol = (
        1e-3 * EI_ref_mass * (P3 / P3_ref) ** 1.35 * (1.1 ** 2.5)
    )
    EI_PMnvolN = EI_ref_num * EI_PMnvol / (1e-3 * EI_ref_mass)

    # correct the shape to fit with previous definition (transpose)
    EI_PMnvol = EI_PMnvol.T
    EI_PMnvol_GMD = EI_PMnvol_GMD.T
    EI_PMnvolN = EI_PMnvolN.T

    # do final checks and replace if necessary
    if np.max(SN_mat_tmp) < 0:
        print(f"SN definition for AC not sufficient {SN_mat_tmp}")
        EI_PMnvol = np.zeros_like(fuelFactor)
        if pmnvolSwitch_cruise == 'SCOPE11':
            EI_PMnvol_GMD = np.zeros_like(fuelFactor.T)
            EI_PMnvol_GMD = np.zeros_like(fuelFactor.T)

    if np.sum(EI_PMnvol < 0) > 1:
        print("Warning! Negative EI(BC) at cruise")
        EI_PMnvol[EI_PMnvol < 0] = 0.0

    return EI_PMnvol_GMD,EI_PMnvol,EI_PMnvolN
