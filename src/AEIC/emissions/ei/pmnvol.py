import functools

import numpy as np

from AEIC.constants import T0, kappa, p0
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import ThrustMode, ThrustModeValues


def PMnvol_MEEM(
    EDB_data: EDBEntry,
    altitudes: np.ndarray,
    Tamb_cruise: np.ndarray,
    Pamb_cruise: np.ndarray,
    machFlight: np.ndarray,
):
    """
    Estimate non-volatile particulate matter (nvPM) emissions at cruise using the
    Mission Emissions Estimation Methodology (MEEM) based on Ahrens et al. (2022),
    SCOPE11, and the methodology of Peck et al. (2013).

    This function computes:
    - Geometric mean diameter (GMD) of emitted particles
    - Mass-based emissions index (EI) in g/kg of fuel
    - Number-based emissions index (EI) in #/kg of fuel

    Parameters
    ----------
    EDB_data : dict
        EDB data containing engine type, bypass ratio, pressure ratio,
        smoke number (SN) matrix, and emission indices (mass and number).
    altitudes : ndarray
        Array of flight altitudes [m] over the mission trajectory.
    Tamb_cruise : ndarray
        Ambient temperature [K] at each point in the trajectory.
    Pamb_cruise : ndarray
        Ambient pressure [Pa] at each point in the trajectory.
    machFlight : ndarray
        Mach number at each point in the trajectory.

    Returns
    -------
    EI_PMnvol_GMD : ndarray
        Geometric mean diameter of particles [nm], constant along the trajectory.
    EI_PMnvol : ndarray
        Emissions index of non-volatile PM mass [g/kg fuel] along the trajectory.
    EI_PMnvolN : ndarray
        Emissions index of non-volatile PM number [#/kg fuel] along the trajectory.

    Notes
    -----
    - If `nvPM_mass_matrix` or `nvPM_num_matrix` is undefined or negative in the
      `EDB_data`, this function reconstructs the values using the SN matrix and
      correlations from the literature.
    - Adjustments for altitude and in-flight thermodynamic conditions are made using
      combustor inlet temperature and pressure estimates derived from ambient
      conditions and engine pressure ratio.
    - Interpolated values account for max thrust EI values where provided.
    - Results with invalid SN or negative EI are set to zero with a warning.
    """
    # ---------------------------------------------------------------------
    # CONSTANTS & LOOK-UP TABLES
    # ---------------------------------------------------------------------
    GMD_mode = np.array([20, 20, 40, 40])  # nm  (idle-app-climb-TO)
    AFR_mode = np.array([106, 83, 51, 45])  # Wayson

    # ---------------------------------------------------------------------
    # (0)  MODE EIs (mg kg⁻¹ or # kg⁻¹)
    # ---------------------------------------------------------------------
    SN = EDB_data.SN_matrix.as_array()
    EI_mass_mode = EDB_data.nvPM_mass_matrix.as_array()
    EI_num_mode = EDB_data.nvPM_num_matrix.as_array()

    if np.min(EI_mass_mode) < 0:
        # Smoke->mass correlation
        CI_mass = 0.6484 * np.exp(0.0766 * SN) / (1 + np.exp(-1.098 * (SN - 3.064)))

        bypass_ratio = EDB_data.BP_Ratio if EDB_data.engine_type == "MTF" else 0.0
        Q_mode = 0.776 * AFR_mode * (1 + bypass_ratio) + 0.767
        kslm = np.log(
            (3.219 * CI_mass * (1 + bypass_ratio) * 1e3 + 312.5)
            / (CI_mass * (1 + bypass_ratio) * 1e3 + 42.6)
        )

        EI_mass_mode = CI_mass * Q_mode * kslm  # mg kg⁻¹

    if np.min(EI_num_mode) < 0:
        EI_num_mode = (6.0 * EI_mass_mode) / (
            np.pi * 1e9 * (GMD_mode * 1e-9) ** 3 * np.exp(4.5 * (np.log(1.8) ** 2))
        )

    # ---------------------------------------------------------------------
    # (1)  COMBUSTOR INLET CONDITIONS ALONG TRAJECTORY
    # ---------------------------------------------------------------------
    max_pr = EDB_data.PR[ThrustMode.IDLE]

    # climb (+) / level (0) / descent (–)
    alt_rate = np.diff(altitudes, prepend=altitudes[0])
    eta_comp = np.where(alt_rate >= 0, 0.88, 0.70)

    max_alt = altitudes.max()
    lin_vary_alt = (altitudes - 3_000.0) / max(1.0, max_alt - 3_000.0)
    pressure_coef = np.where(
        alt_rate > 0,
        0.85 + (1.15 - 0.85) * lin_vary_alt,
        np.where(alt_rate == 0, 0.95, 0.12),
    )

    # convert ambient -> *total* T/P first
    Tt_amb = Tamb_cruise * (1 + (kappa - 1) / 2 * machFlight**2)
    Pt_amb = Pamb_cruise * (1 + (kappa - 1) / 2 * machFlight**2) ** (
        kappa / (kappa - 1)
    )

    P3 = Pt_amb * (1 + pressure_coef * (max_pr - 1))
    T3 = Tt_amb * (1 + (1 / eta_comp) * ((P3 / Pt_amb) ** ((kappa - 1) / kappa) - 1))

    # ---------------------------------------------------------------------
    # (2)  GROUND / REFERENCE
    # ---------------------------------------------------------------------
    T3_ref = T3.copy()
    P3_ref = p0 * (1 + eta_comp * (T3_ref / T0 - 1)) ** (kappa / (kappa - 1))

    FG_over_Foo = (P3_ref / p0 - 1) / (max_pr - 1)  # (N,)

    # ---------------------------------------------------------------------
    # (3)  INTERPOLATION VERSUS THRUST SETTING
    # ---------------------------------------------------------------------
    def build_interp(arr_mode, Tmax: float, Tmax_thrust: float):
        if np.isnan(Tmax_thrust) or Tmax_thrust < 0:
            # four-point
            arr = np.concatenate(([arr_mode[0]], arr_mode, [arr_mode[-1]]))
            tgrid = np.array([-10, 0.07, 0.3, 0.85, 1.0, 100])
        else:
            if np.isclose(Tmax_thrust, 0.575):
                arr = np.concatenate(
                    ([arr_mode[0]], arr_mode[0:2], [Tmax], arr_mode[2:], [arr_mode[-1]])
                )
                tgrid = np.array([-10, 0.07, 0.3, 0.575, 0.85, 1.0, 100])
            elif np.isclose(Tmax_thrust, 0.925):
                arr = np.concatenate(
                    ([arr_mode[0]], arr_mode[0:3], [Tmax], arr_mode[3:], [arr_mode[-1]])
                )
                tgrid = np.array([-10, 0.07, 0.3, 0.85, 0.925, 1.0, 100])
            else:
                raise ValueError("Unrecognised *_max_thrust value.")
        return tgrid, arr

    t_mass, arr_mass = build_interp(
        EI_mass_mode, EDB_data.EImass_max, EDB_data.EImass_max_thrust
    )
    t_num, arr_num = build_interp(
        EI_num_mode, EDB_data.EInum_max, EDB_data.EInum_max_thrust
    )

    t_GMD = np.array([-10, 0.07, 0.3, 0.85, 1.0, 100])
    arr_GMD = np.concatenate(([GMD_mode[0]], GMD_mode, [GMD_mode[-1]]))

    EI_ref_mass = np.interp(FG_over_Foo, t_mass, arr_mass)  # mg kg⁻¹
    EI_ref_num = np.interp(FG_over_Foo, t_num, arr_num)  # # kg⁻¹
    GMD_ref = np.interp(FG_over_Foo, t_GMD, arr_GMD)  # nm

    # ---------------------------------------------------------------------
    # (4)  ALTITUDE ADJUSTMENT
    # ---------------------------------------------------------------------
    EI_mass = 1e-3 * EI_ref_mass * (P3 / P3_ref) ** 1.35 * (1.1**2.5)  # g kg⁻¹
    EI_num = EI_ref_num * EI_mass / (1e-3 * EI_ref_mass)  # # kg⁻¹

    # ---------------------------------------------------------------------
    # (5)  SANITY CHECKS
    # ---------------------------------------------------------------------
    if np.max(SN) < 0:  # no valid smoke numbers
        EI_mass.fill(0.0)
        EI_num.fill(0.0)
        GMD_ref.fill(0.0)

    neg = EI_mass < 0
    if neg.any():
        print("Warning: negative EI_mass set to zero at", np.where(neg)[0])
        EI_mass[neg] = 0.0

    return GMD_ref, EI_mass, EI_num


@functools.cache
def calculate_nvPM_scope11_LTO(
    SN_matrix: ThrustModeValues, engine_type: str, BP_Ratio: float
) -> ThrustModeValues:
    """
    Calculate PM non-volatile Emission Index (EI) using SCOPE11 methodology (2019).

    Parameters
    ----------
    SN_matrix : ThrustModeValues
        Smoke number matrix for each ICAO mode.
    ENGINE_TYPE : str
        Engine type ('TF', 'MTF', etc.).
    BP_Ratio : float
        Bypass ratio.

    Returns
    -------
    PMnvolEI_best_ICAOthrust : ThrustModeValues
        Emission index of non-volatile PM mass [g/kg_fuel],
        including 0% thrust extrapolation as first column.
    """

    # Air to fuel ration at four LTO points, estimated by Wayson et al. (2009)
    AFR = ThrustModeValues(106, 83, 51, 45)

    # Geometrical mean diameter estimations at LTO points (nm)
    GMD = ThrustModeValues(20.0, 20.0, 40.0, 40.0)

    # EI nvPM mass (g/kg)
    nvPM_EI_mass_g_per_kg = ThrustModeValues(0.0, mutable=True)
    # EI nvPM particle number (particles/kg)
    nvPM_EI_num_particle_per_kg = ThrustModeValues(0.0, mutable=True)
    # --- Volumetric Fuel Flow Q [m³/kg_fuel]
    Q = ThrustModeValues(0.0, mutable=True)

    for mode in ThrustMode:
        SN = SN_matrix[mode]

        # --- Skip invalid SN
        if SN == -1 or SN == 0:
            continue

        # --- Exit Plane BC Concentration C_BC,e [ug/m3]
        SN = min(SN, 40)
        CI_mass = 648.4 * np.exp(0.0766 * SN) / (1 + np.exp(-1.099 * (SN - 3.064)))

        # --- System loss multiplier (kslm)
        if engine_type == 'MTF' or engine_type == 'TF':
            kslm = np.log(
                (3.219 * CI_mass * (1.0 + BP_Ratio) + 312.5)
                / (CI_mass * (1.0 + BP_Ratio) + 42.6)
            )
        else:
            kslm = 0.0

        Q[mode] = 0.776 * AFR[mode] * (1.0 + BP_Ratio) + 0.767

        # Unit change: μg/m^3 -> mg/m^3
        EI_mass_mg_per_kg = (CI_mass * Q[mode] * kslm) * 1e-3

        gmd_nm = GMD[mode]
        mean_particle_mass_mg = (
            (np.pi / 6.0)
            * 1.0e9  # 1 g/cm^3 = 1e9 mg/m^3, as used implicitly in the paper's Eq. (5)
            * (gmd_nm / 1.0e9) ** 3
            * np.exp(4.5 * (np.log(1.8) ** 2))
        )
        nvPM_EI_num_particle_per_kg[mode] = (
            EI_mass_mg_per_kg / mean_particle_mass_mg
            if mean_particle_mass_mg > 0
            else 0.0
        )
        nvPM_EI_mass_g_per_kg[mode] = EI_mass_mg_per_kg / 1000
    # Freeze the return value because we're caching.
    nvPM_EI_num_particle_per_kg.freeze()
    nvPM_EI_mass_g_per_kg.freeze()
    return (
        nvPM_EI_num_particle_per_kg,
        nvPM_EI_mass_g_per_kg,
    )  # TODO: make a nvPM struct
