import functools
from dataclasses import dataclass

import numpy as np

from AEIC.constants import T0, kappa, p0
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import ThrustMode, ThrustModeValues
from AEIC.units import METERS_TO_FEET


@dataclass
class nvPMProfile:
    mass: ThrustModeValues
    number: ThrustModeValues | None


def nvPM_MEEM(
    edb_data: EDBEntry,
    altitudes: np.ndarray,
    rocd: np.ndarray,
    amb_temperature: np.ndarray,
    amb_pressure: np.ndarray,
    mach_number: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate non-volatile particulate matter (nvPM) emissions at cruise using the
    Mission Emissions Estimation Methodology (MEEM) based on Ahrens et al. (2022),
    SCOPE11, and the methodology of Peck et al. (2013).

    Parameters
    ----------
    edb_data : EDBEntry
        Engine database data for the selected engine.
    altitudes : ndarray
        Array of flight altitudes [m] over the mission trajectory.
    rocd : ndarray
        Rate of climb/descent [m/s]. Positive = climb, zero = cruise,
        negative = descent.
    amb_temperature : ndarray
        Ambient temperature [K] at each point in the trajectory.
    amb_pressure : ndarray
        Ambient pressure [Pa] at each point in the trajectory.
    mach_number : ndarray
        Mach number at each point in the trajectory.

    Returns
    -------
    EI_nvPM : ndarray
        Emission index of non-volatile PM mass [g/kg fuel] along the trajectory.
    EI_nvPM_N : ndarray
        Emission index of non-volatile PM particle number [#/kg fuel]
        along the trajectory.

    Notes
    -----
    - Step 3 uses the 4-point linear interpolation method only (2.4): EI values at
      the four ICAO LTO thrust settings (7/30/85/100%) with no peak point inserted.
    - Results with invalid SN or negative EI are set to zero with a warning.
    """
    # -------------------------------------------------------------------------
    # (0)  MODE EIs  —  Section 2.1 (Step 0)
    #
    # MEEM requires EImass [mg/kg] and EInum [#/kg] at the four ICAO LTO
    # thrust settings (7/30/85/100 %).  Two sources are available:
    #   (a) Direct EDB nvPM measurements: used when all four mode
    #       values are positive and therefore valid.
    #   (b) SCOPE11 fallback: when the engine only has smoke-number (SN) data,
    #       calculate_nvPM_scope11_LTO() converts SN → EImass/EInum via the
    #       SCOPE11 correlations (Eqs. 1–5 in the paper, Fig. 2).
    # -------------------------------------------------------------------------
    use_edb_nvpm = all(
        edb_data.nvPM_mass_matrix[mode] > 0.0 and edb_data.nvPM_num_matrix[mode] > 0.0
        for mode in ThrustMode
    )

    if use_edb_nvpm:
        # Path (a): use EDB nvPM data directly.
        EI_mass_mode = edb_data.nvPM_mass_matrix.as_array()  # mg/kg
        EI_num_mode = edb_data.nvPM_num_matrix.as_array()  # #/kg
    else:
        # Path (b): SCOPE11 fallback — derives EI from smoke numbers.
        # Returns mass in g/kg; convert to mg/kg (* 1000) to keep units
        profile = calculate_nvPM_scope11_LTO(
            edb_data.SN_matrix,
            edb_data.engine_type,
            edb_data.BP_Ratio,
        )
        EI_mass_mode = 1000.0 * profile.mass.as_array()  # g/kg → mg/kg
        EI_num_mode = (
            profile.number.as_array()
            if profile.number is not None
            else np.zeros_like(EI_mass_mode)
        )

    # -------------------------------------------------------------------------
    # (1)  IN-FLIGHT THERMODYNAMIC CONDITIONS  —  Section 2.2 (Step 1)
    #
    # Estimate combustor inlet pressure P3 and temperature T3 at each
    # trajectory point.  The paper's approach (Eqs. 6–9, Fig. 4, Table 3)
    # -------------------------------------------------------------------------
    opr_pi00 = edb_data.PR[ThrustMode.TAKEOFF]

    # Compressor efficiency (Table 3): 0.88 for climb/cruise, 0.70 for descent.
    eta_comp = np.where(rocd < 0, 0.70, 0.88)

    altitudes_ft = altitudes * METERS_TO_FEET
    # TODO: Implement better way to get cruise altitude
    cruise_altitude_ft = altitudes_ft.max()

    # Linear interpolation weight for climb: goes 0→1 as altitude rises from
    # 3000 ft to cruise altitude, implementing Eq. (9).
    lin_vary_alt = (altitudes_ft - 3_000.0) / max(1.0, cruise_altitude_ft - 3_000.0)
    pressure_coef_climb = np.clip(
        0.85 + (1.15 - 0.85) * lin_vary_alt,
        0.85,
        1.15,
    )
    # Altitude pressure coefficient (Table 3 / Eq. 9):
    #   Climb  → varies linearly 0.85→1.15 with altitude (Eq. 9)
    #   Cruise → fixed at 0.95
    #   Descent→ fixed at 0.12
    pressure_coef = np.where(
        rocd > 0,
        pressure_coef_climb,  # climb
        np.where(rocd == 0.0, 0.95, 0.12),  # cruise / descent
    )

    # Step 1a: total (stagnation) ambient temperature and pressure.
    # These account for ram compression at the engine face at Mach > 0.
    Tt_amb = amb_temperature * (1 + (kappa - 1) / 2 * mach_number**2)
    Pt_amb = amb_pressure * (1 + (kappa - 1) / 2 * mach_number**2) ** (
        kappa / (kappa - 1)
    )

    # Step 1b: combustor inlet pressure at altitude (Eq. 7).
    P3 = Pt_amb * (1 + pressure_coef * (opr_pi00 - 1))

    # Step 1c: combustor inlet temperature at altitude (Eq. 8).
    T3 = Tt_amb * (1 + (1 / eta_comp) * ((P3 / Pt_amb) ** ((kappa - 1) / kappa) - 1))

    # -------------------------------------------------------------------------
    # (2)  GROUND REFERENCE CONDITIONS  —  Section 2.3 (Step 2)
    #
    # The Ground Reference (GR) state is defined as the sea-level static
    # operating point that produces the same T3 as the current altitude point
    # (T3GR ≡ T3Alt)
    # -------------------------------------------------------------------------

    # T3GR = T3Alt.
    T3_ref = T3.copy()

    # P3GR from Eq. (11): inverse compressor map at sea-level conditions.
    # p0 and T0 are standard sea-level values (101325 Pa, 288.15 K).
    P3_ref = p0 * (1 + eta_comp * (T3_ref / T0 - 1)) ** (kappa / (kappa - 1))

    # FGR/F00 from Eq. (12): normalised ground-reference thrust fraction.
    # This locates the operating point on the 0–1 thrust axis used in Step 3.
    FG_over_Foo_raw = (P3_ref / p0 - 1) / (opr_pi00 - 1)  # shape: (N,)

    # -------------------------------------------------------------------------
    # (3)  INTERPOLATION VS. THRUST SETTING  —  Section 2.4 (Step 3, Fig. 11)
    #
    # Linearly interpolate EImass and EInum across the four
    # ICAO LTO thrust settings — Idle (7%), Approach (30%), Climb (85%),
    # Take-Off (100%) — to obtain ground-reference EI at the FGR/F00 computed
    # in Step 2.
    # -------------------------------------------------------------------------

    # Thurst fractions at each ThrustMode
    tgrid = (
        np.array([mode.thrust_percentage for mode in ThrustMode], dtype=float) / 100.0
    )

    # Clamp thrust fraction to the valid LTO range to avoid extrapolation.
    FG_over_Foo = np.clip(FG_over_Foo_raw, 0.07, 1.0)

    # Linear interpolation on the 4-point thrust–EI grid.
    EI_ref_mass = np.interp(FG_over_Foo, tgrid, EI_mass_mode)  # mg/kg at GR
    EI_ref_num = np.interp(FG_over_Foo, tgrid, EI_num_mode)  # #/kg  at GR

    # -------------------------------------------------------------------------
    # (4)  ALTITUDE ADJUSTMENT  —  Section 2.5 (Step 4, Eqs. 15–16, Fig. 15)
    #
    # Transpose the ground-reference EI to actual altitude conditions.
    # -------------------------------------------------------------------------

    # EImass at altitude [g/kg]: convert from mg/kg (*1e-3), apply pressure
    # scaling and enrichment factor.
    EI_mass = 1e-3 * EI_ref_mass * (P3 / P3_ref) ** 1.35 * (1.1**2.5)  # g/kg

    # EInum at altitude [#/kg]: preserve the GR number-to-mass ratio (Eq. 16).
    EI_num = np.zeros_like(EI_ref_num)
    valid_mass = EI_ref_mass > 0.0
    EI_num[valid_mass] = (
        EI_ref_num[valid_mass] * EI_mass[valid_mass] / (1e-3 * EI_ref_mass[valid_mass])
    )

    return EI_mass, EI_num


@functools.cache
def calculate_nvPM_scope11_LTO(
    SN_matrix: ThrustModeValues, engine_type: str, BP_Ratio: float
) -> nvPMProfile:
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
    nvPMProfile
        nvPM mass and number emission indices [g/kg and #/kg fuel].
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
        if engine_type == 'MTF':
            bypass_factor = 1.0 + BP_Ratio
        elif engine_type == 'TF':
            bypass_factor = 1.0
        else:
            bypass_factor = 1.0

        if engine_type == 'MTF' or engine_type == 'TF':
            kslm = np.log(
                (3.219 * CI_mass * bypass_factor + 312.5)
                / (CI_mass * bypass_factor + 42.6)
            )
        else:
            kslm = 0.0

        Q[mode] = 0.776 * AFR[mode] * bypass_factor + 0.767

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
    return nvPMProfile(
        nvPM_EI_mass_g_per_kg,
        nvPM_EI_num_particle_per_kg,
    )
