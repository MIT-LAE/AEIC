from __future__ import annotations

import numpy as np
import pytest

from AEIC.config.emissions import ClimbDescentMode
from AEIC.emissions import compute_emissions
from AEIC.emissions.ei.hcco import EI_HCCO
from AEIC.emissions.ei.nox import BFFM2_EINOx, NOx_speciation
from AEIC.emissions.gse import get_GSE_emissions
from AEIC.emissions.trajectory import (
    _calculate_EI_nvPM,
    _trajectory_slice,
)
from AEIC.emissions.types import AtmosphericState
from AEIC.emissions.utils import (
    get_SLS_equivalent_fuel_flow,
    scope11_profile,
)
from AEIC.missions import Mission
from AEIC.missions.mission import iso_to_timestamp
from AEIC.performance.apu import APU
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import (
    LTOPerformance,
    ThrustMode,
    ThrustModeValues,
)
from AEIC.types import AircraftClass, Species, SpeciesValues

sample_mission = Mission(
    origin="BOS",
    destination="LAX",
    aircraft_type="738",
    departure=iso_to_timestamp('2019-01-01 12:00:00'),
    arrival=iso_to_timestamp('2019-01-01 18:00:00'),
    load_factor=1.0,
)


class DummyPerformanceModel:
    def __init__(self):
        self.edb = EDBEntry(
            engine='Test Engine',
            uid='TEST123',
            engine_type='TF',
            BP_Ratio=5.0,
            rated_thrust=100.0,
            fuel_flow=ThrustModeValues(0.25, 0.5, 0.9, 1.2),
            CO_EI_matrix=ThrustModeValues(20.0, 15.0, 10.0, 5.0),
            HC_EI_matrix=ThrustModeValues(4.0, 3.0, 2.0, 1.0),
            EI_NOx_matrix=ThrustModeValues(8.0, 12.0, 26.0, 32.0),
            SN_matrix=ThrustModeValues(6.0, 8.0, 11.0, 13.0),
            nvPM_mass_matrix=ThrustModeValues(5.0, 5.5, 6.0, 6.5),
            nvPM_num_matrix=ThrustModeValues(2.0e14, 2.1e14, 2.2e14, 2.3e14),
            PR=ThrustModeValues(22.0, 22.0, 22.0, 22.0),
            EImass_max=8.0,
            EImass_max_thrust=0.575,
            EInum_max=2.4e14,
            EInum_max_thrust=0.575,
        )
        self.lto = LTOPerformance(
            source='test',
            ICAO_UID='TEST123',
            rated_thrust=100.0 * 1000.0,
            thrust_pct=ThrustModeValues(7, 30, 85, 100),
            fuel_flow=ThrustModeValues(0.25, 0.5, 0.9, 1.2),
            EI_NOx=ThrustModeValues(8.0, 12.0, 32.0, 40.0),
            EI_HC=ThrustModeValues(4.0, 3.0, 1.5, 1.0),
            EI_CO=ThrustModeValues(20.0, 10.0, 3.0, 2.0),
        )
        self.apu = APU(
            name='Test APU',
            defra='00000',
            fuel_kg_per_s=0.03,
            PM10_g_per_kg=0.4,
            NOx_g_per_kg=0.05,
            HC_g_per_kg=0.02,
            CO_g_per_kg=0.03,
        )
        self.aircraft_class = AircraftClass.WIDE
        self.number_of_engines = 2


class DummyTrajectory:
    def __init__(self):
        self.name = 'DUMMY'
        self.n_climb = 2
        self.n_cruise = 2
        self.n_descent = 2
        self._npoints = self.n_climb + self.n_cruise + self.n_descent
        self.fuel_mass = np.array(
            [2000.0, 1994.0, 1987.5, 1975.0, 1960.0, 1945.0], dtype=float
        )
        self.fuel_flow = np.array([0.3, 0.35, 0.55, 0.65, 0.5, 0.32], dtype=float)
        self.altitude = np.array(
            [0.0, 1500.0, 6000.0, 11000.0, 9000.0, 2000.0], dtype=float
        )
        self.rate_of_climb = np.array([0.0, 6.0, 6.0, 6.0, -6.0, -6.0], dtype=float)
        self.true_airspeed = np.array(
            [120.0, 150.0, 190.0, 210.0, 180.0, 140.0], dtype=float
        )

    def __len__(self):
        return self._npoints


@pytest.fixture
def trajectory():
    return DummyTrajectory()


@pytest.fixture
def perf_model():
    return DummyPerformanceModel()


@pytest.fixture
def emissions(perf_model, fuel, trajectory):
    return compute_emissions(perf_model, fuel, trajectory)


def test_emissions_species(emissions):
    """`Emissions.species` covers exactly the `Species` enum — count
    equality alone would silently allow a regression that dropped one
    species and added a stray duplicate.
    """
    assert emissions.species == set(Species)


def _expected_trajectory_indices(perf_model, trajectory):
    # DELIBERATELY SELF-REFERENTIAL: this helper drives the same BFFM2,
    # EI_HCCO, SLS, and AtmosphericState paths that compute_emissions uses.
    # The test that consumes this (test_compute_emissions_pipeline_wiring)
    # is therefore a wiring/integration check — it verifies that
    # compute_emissions routes trajectory inputs through the science
    # helpers correctly, not that those helpers are scientifically
    # correct. Science correctness is covered independently by
    # test_emission_functions.py::TestBFFM2_EINOx,
    # ::TestEI_HCCO, and ::TestNOxSpeciation, all of which cite
    # notebook rounded-results cells.
    idx_slice = _trajectory_slice(trajectory)
    lto_inputs = perf_model.lto

    fuel_flow = trajectory.fuel_flow[idx_slice]

    altitudes = trajectory.altitude[idx_slice]
    tas = trajectory.true_airspeed[idx_slice]
    atmos = AtmosphericState(altitudes, tas)
    sls_flow = get_SLS_equivalent_fuel_flow(
        fuel_flow=fuel_flow,
        Pamb=atmos.pressure,
        Tamb=atmos.temperature,
        mach_number=atmos.mach,
        n_eng=perf_model.number_of_engines,
    )

    expected = {
        Species.CO2: np.full_like(fuel_flow, 3155.6),
        Species.H2O: np.full_like(fuel_flow, 1233.3865),
        Species.SO2: np.full_like(fuel_flow, 1.176),
        Species.SO4: np.full_like(fuel_flow, 0.036),
    }

    nox_result = BFFM2_EINOx(
        sls_equiv_fuel_flow=sls_flow,
        EI_NOx_matrix=lto_inputs.EI_NOx,
        fuelflow_performance=lto_inputs.fuel_flow,
        Tamb=atmos.temperature,
        Pamb=atmos.pressure,
    )
    expected[Species.NOx] = nox_result.NOxEI
    expected[Species.NO] = nox_result.NOEI
    expected[Species.NO2] = nox_result.NO2EI
    expected[Species.HONO] = nox_result.HONOEI
    no_prop = nox_result.noProp
    no2_prop = nox_result.no2Prop
    hono_prop = nox_result.honoProp

    expected[Species.HC] = EI_HCCO(
        sls_flow,
        lto_inputs.EI_HC,
        lto_inputs.fuel_flow,
        Tamb=atmos.temperature,
        Pamb=atmos.pressure,
    )
    expected[Species.CO] = EI_HCCO(
        sls_flow,
        lto_inputs.EI_CO,
        lto_inputs.fuel_flow,
        Tamb=atmos.temperature,
        Pamb=atmos.pressure,
    )

    return expected, (no_prop, no2_prop, hono_prop)


@pytest.mark.config_updates(emissions__nvpm_method='meem')
def test_compute_emissions_pipeline_wiring(perf_model, trajectory, emissions):
    """Verify compute_emissions wires the per-segment pipeline correctly.

    What this test checks:
    - compute_emissions routes trajectory inputs (fuel_flow, altitude,
      true_airspeed) through AtmosphericState, SLS, BFFM2_EINOx, and
      EI_HCCO in the expected order and with the expected arguments, so
      the values it stores in trajectory_indices match what those helpers
      return when driven directly with the same inputs.
    - The per-segment emissions identity holds:
      trajectory_emissions == trajectory_indices * fuel_burn_per_segment.
    - NOx correctly speciates into NO + NO2 + HONO using the factors
      returned by BFFM2_EINOx.
    - NOx speciation fractions satisfy the physical mass-conservation
      identity no_prop + no2_prop + hono_prop == 1 (independent of the
      helper — a true correctness signal).

    What this test does NOT check: the scientific correctness of
    BFFM2_EINOx / EI_HCCO / NOx_speciation / AtmosphericState /
    get_SLS_equivalent_fuel_flow themselves. That is covered by
    test_emission_functions.py (TestBFFM2_EINOx, TestEI_HCCO,
    TestNOxSpeciation) and test_atmospheric_state_and_sls_flow_shapes,
    which each assert against independently-grounded reference values
    (notebook rounded-results cells / ISA formulas).
    """
    expected, (no_prop, no2_prop, hono_prop) = _expected_trajectory_indices(
        perf_model, trajectory
    )
    idx_slice = _trajectory_slice(trajectory)
    for species, expected_values in expected.items():
        if species not in emissions.trajectory_indices:
            continue
        np.testing.assert_allclose(
            emissions.trajectory_indices[species][idx_slice], expected_values
        )
    fuel_burn = emissions.fuel_burn_per_segment[idx_slice]
    for field in emissions.trajectory_emissions:
        np.testing.assert_allclose(
            emissions.trajectory_emissions[field][idx_slice],
            emissions.trajectory_indices[field][idx_slice] * fuel_burn,
        )
    np.testing.assert_allclose(
        emissions.trajectory_indices[Species.NO][idx_slice],
        emissions.trajectory_indices[Species.NOx][idx_slice] * no_prop,
    )
    np.testing.assert_allclose(
        emissions.trajectory_indices[Species.NO2][idx_slice],
        emissions.trajectory_indices[Species.NOx][idx_slice] * no2_prop,
    )
    np.testing.assert_allclose(
        emissions.trajectory_indices[Species.HONO][idx_slice],
        emissions.trajectory_indices[Species.NOx][idx_slice] * hono_prop,
    )
    # Mass-conservation identity: NOx must fully partition into
    # NO + NO2 + HONO with no gain or loss. Independent of the helper
    # and of the SUT's speciation factor values — a genuine correctness
    # signal that catches any bug causing the fractions to drift from a
    # valid probability split.
    np.testing.assert_allclose(no_prop + no2_prop + hono_prop, 1.0)

    assert emissions.lifecycle_co2 is not None


@pytest.mark.config_updates(
    emissions__nvpm_method='meem', emissions__lifecycle_enabled=False
)
def test_sum_total_emissions_matches_components(perf_model, fuel, trajectory):
    emissions = compute_emissions(perf_model, fuel, trajectory)
    for species in emissions.total_emissions:
        expected = 0.0
        if species in emissions.trajectory_emissions:
            expected += np.sum(emissions.trajectory_emissions[species])
        if species in emissions.lto_emissions:
            expected += emissions.lto_emissions[species].sum()
        if species in emissions.apu_emissions:
            expected += emissions.apu_emissions[species]
        if species in emissions.gse_emissions:
            expected += emissions.gse_emissions[species]
        assert emissions.total_emissions[species] == pytest.approx(expected)


def test_scope11_profile_caching(perf_model):
    profile_first = scope11_profile(perf_model.edb)
    profile_second = scope11_profile(perf_model.edb)
    assert profile_first.mass is profile_second.mass


@pytest.mark.config_updates(emissions__climb_descent_mode=ClimbDescentMode.TRAJECTORY)
def test_lto_respects_traj_flag_true(perf_model, fuel, trajectory):
    """In TRAJECTORY mode the climb and descent emissions move to the
    trajectory side; the LTO APPROACH/CLIMB columns must be zero AND
    the trajectory side must carry the mass. Asserting only the LTO
    zero would let a regression that dropped the mass entirely pass.
    """
    output = compute_emissions(perf_model, fuel, trajectory)
    for m in [ThrustMode.APPROACH, ThrustMode.CLIMB]:
        assert all(
            np.isclose(output.lto_emissions[species][m], 0.0)
            for species in output.lto_emissions
        )
    climb_slice = slice(0, trajectory.n_climb)
    descent_slice = slice(len(trajectory) - trajectory.n_descent, len(trajectory))
    for species in (Species.NOx, Species.HC, Species.CO):
        assert np.sum(output.trajectory_emissions[species][climb_slice]) > 0
        assert np.sum(output.trajectory_emissions[species][descent_slice]) > 0


@pytest.mark.config_updates(emissions__climb_descent_mode=ClimbDescentMode.LTO)
def test_lto_respects_traj_flag_false(perf_model, fuel, trajectory):
    """Reciprocal: in LTO mode the climb and descent emissions live on
    the LTO side. APPROACH/CLIMB LTO columns must be non-zero, and the
    trajectory side must zero out everything outside cruise.
    """
    output = compute_emissions(perf_model, fuel, trajectory)
    for m in (ThrustMode.APPROACH, ThrustMode.CLIMB):
        assert any(
            output.lto_emissions[species][m] > 0 for species in output.lto_emissions
        )
    climb_slice = slice(0, trajectory.n_climb)
    cruise_slice = slice(trajectory.n_climb, len(trajectory) - trajectory.n_descent)
    descent_slice = slice(len(trajectory) - trajectory.n_descent, len(trajectory))
    for species, arr in output.trajectory_emissions.items():
        assert np.sum(arr[cruise_slice]) > 0
        assert np.allclose(arr[climb_slice], 0)
        assert np.allclose(arr[descent_slice], 0)


def test_lto_nox_split_consistent_with_speciation_factors(emissions):
    """Consistency check: the LTO NOx-split path multiplies by the
    same `NOx_speciation()` factors the SUT uses, so this verifies the
    pipeline applies the speciation step (not that the factors
    themselves are right). Factor correctness is asserted independently
    by `test_emission_functions.py::TestNOxSpeciation::test_NOx_speciation_results`.
    """
    speciation = NOx_speciation()

    def check(species: Species, prop: ThrustModeValues):
        v1 = emissions.lto_indices[species]
        v2 = emissions.lto_indices[Species.NOx] * prop
        np.testing.assert_allclose(v1.as_array(), v2.as_array())

    check(Species.NO, speciation.no)
    check(Species.NO2, speciation.no2)
    check(Species.HONO, speciation.hono)


@pytest.mark.config_updates(emissions__nvpm_method='meem')
def test_calculate_nvpm_meem_populates_fields(perf_model, trajectory):
    atmos = AtmosphericState(trajectory.altitude, trajectory.true_airspeed)
    result = _calculate_EI_nvPM(
        perf_model, trajectory.altitude, trajectory.rate_of_climb, atmos
    )
    expected_mass = np.array(
        [0.00829664, 0.00715419, 0.00522982, 0.00355236, 0.00313372, 0.00573352]
    )
    expected_number = np.array(
        [
            2.93573343e14,
            2.60188500e14,
            1.92857849e14,
            1.32653902e14,
            1.25348986e14,
            2.27148353e14,
        ]
    )
    np.testing.assert_allclose(result[Species.nvPM], expected_mass, atol=1e-8)
    if Species.nvPM_N in result:
        np.testing.assert_allclose(result[Species.nvPM_N], expected_number)


@pytest.mark.config_updates(emissions__nvpm_method='none')
def test_calculate_nvpm_none_disables_outputs(perf_model, trajectory):
    atmos = AtmosphericState(trajectory.altitude, trajectory.true_airspeed)
    result = _calculate_EI_nvPM(
        perf_model, trajectory.altitude, trajectory.rate_of_climb, atmos
    )
    assert Species.nvPM not in result
    assert Species.nvPM_N not in result


def _isa_reference(altitudes_m, tas_m_s):
    """Independent ISA reference per ICAO Doc 7488 / US Std Atmosphere 1976.

    Must not import from AEIC.utils.standard_atmosphere — that module is the
    SUT. Constants are inlined from the published standard so this helper is
    a standalone reference against which AtmosphericState can be checked.
    """
    T0 = 288.15
    p0 = 101325.0
    L = -0.0065
    g = 9.80665
    R = 287.05287
    kappa = 1.4
    h_tp = 11000.0

    h = np.asarray(altitudes_m, dtype=float)
    T = np.where(h <= h_tp, T0 + L * h, T0 + L * h_tp)
    T_tp = T0 + L * h_tp
    p_tp = p0 * (T_tp / T0) ** (-g / (L * R))
    p = np.where(
        h <= h_tp,
        p0 * (T / T0) ** (-g / (L * R)),
        p_tp * np.exp(-g / (R * T_tp) * (h - h_tp)),
    )
    M = np.asarray(tas_m_s, dtype=float) / np.sqrt(kappa * R * T)
    return T, p, M


def test_atmospheric_state_and_sls_flow_shapes(perf_model, trajectory):
    atmos = AtmosphericState(trajectory.altitude, trajectory.true_airspeed)
    # Expected values from published ISA formulas (ICAO Doc 7488 / US Std
    # Atmosphere 1976), computed in _isa_reference() without importing from
    # AEIC.utils.standard_atmosphere (the SUT).
    # Spot checks vs ICAO Doc 7488 Table A:
    #   h=0 m      -> T=288.150 K, p=101325 Pa  (sea level)
    #   h=11000 m  -> T=216.650 K, p≈22632 Pa   (tropopause)
    expected_temp, expected_pressure, expected_mach = _isa_reference(
        trajectory.altitude, trajectory.true_airspeed
    )
    np.testing.assert_allclose(atmos.temperature, expected_temp)
    np.testing.assert_allclose(atmos.pressure, expected_pressure)
    np.testing.assert_allclose(atmos.mach, expected_mach)

    sls_flow = get_SLS_equivalent_fuel_flow(
        fuel_flow=trajectory.fuel_flow,
        Pamb=atmos.pressure,
        Tamb=atmos.temperature,
        mach_number=atmos.mach,
        n_eng=perf_model.number_of_engines,
    )
    # NOTE: RESULTS FROM notebooks/test-cases.ipynb NOTEBOOK.
    expected_sls = np.array(
        [0.153777, 0.191545, 0.365257, 0.544758, 0.373176, 0.177299]
    )
    np.testing.assert_allclose(sls_flow, expected_sls, atol=1e-5)


def test_get_gse_emissions_matches_reference_profile(fuel):
    # Primary nominal per-LTO-cycle GSE emissions for the WIDE aircraft class
    # (units: grams / cycle). Source: AEIC v2 MATLAB implementation carried
    # forward into this codebase; legacy reference fixtures live under
    # tests/data/verification/legacy/ (see README). TODO: back-trace to a
    # primary external source (Stettler 2011, FAA AEDT, EPA EDMS, etc.) and
    # cite here once identified — this is the only remaining self-referential
    # piece of the test.
    co2_wide = 58_000.0
    nox_wide = 900.0
    hc_wide = 70.0
    co_wide = 300.0
    pm10_wide = 55.0

    # GSE-specific NOx speciation fractions (gse.py:31-33). NOTE: these are
    # not the ICAO NOx_speciation() factors used for trajectory/LTO splits.
    no_frac, no2_frac, hono_frac = 0.90, 0.09, 0.01

    # Fuel-sulfur mass balance (gse.py:36-46). Fuel-sulfur concentration
    # 5 ppm; 2 % converts to sulfate; molecular-weight ratios map sulfur
    # mass to SO2 / SO4 mass.
    fsc = 5e-6  # 5 ppm as a fraction
    kg_to_g = 1000.0
    eps = 0.02
    mw_o2, mw_so2, mw_so4 = 32.0, 64.0, 96.0
    so4 = fsc * kg_to_g * eps * (mw_so4 / mw_o2)
    so2 = fsc * kg_to_g * (1.0 - eps) * (mw_so2 / mw_o2)

    gse_fuel = co2_wide / fuel.EI_CO2
    expected = SpeciesValues[float](
        {
            Species.CO2: co2_wide,
            Species.NOx: nox_wide,
            Species.HC: hc_wide,
            Species.CO: co_wide,
            Species.H2O: fuel.EI_H2O * gse_fuel,
            Species.NO: nox_wide * no_frac,
            Species.NO2: nox_wide * no2_frac,
            Species.HONO: nox_wide * hono_frac,
            Species.SO4: so4,
            Species.SO2: so2,
            Species.nvPM: pm10_wide - so4,
        }
    )

    result = get_GSE_emissions(AircraftClass.WIDE, fuel).emissions
    for species, value in expected.items():
        assert result[species] == pytest.approx(value)
    if Species.nvPM_N in result:
        assert result[Species.nvPM_N] == pytest.approx(0.0)
