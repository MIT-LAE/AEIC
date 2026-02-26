from __future__ import annotations

import numpy as np
import pytest

from AEIC.config import config
from AEIC.config.emissions import ClimbDescentMode, PMnvolMethod, PMvolMethod
from AEIC.emissions import compute_emissions
from AEIC.emissions.ei.hcco import EI_HCCO
from AEIC.emissions.ei.nox import BFFM2_EINOx, NOx_speciation
from AEIC.emissions.ei.pmnvol import calculate_PMnvolEI_scope11
from AEIC.emissions.ei.pmvol import EI_PMvol_FOA3, EI_PMvol_FuelFlow
from AEIC.emissions.gse import get_GSE_emissions
from AEIC.emissions.trajectory import (
    _calculate_EI_PMnvol,
    _calculate_EI_PMvol,
    _thrust_percentages_from_categories,
    _trajectory_slice,
)
from AEIC.emissions.types import AtmosphericState
from AEIC.emissions.utils import (
    get_SLS_equivalent_fuel_flow,
    get_thrust_cat_cruise,
    scope11_profile,
)
from AEIC.missions import Mission
from AEIC.missions.mission import iso_to_timestamp
from AEIC.performance.apu import APU
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import (
    LTOPerformance,
    ThrustMode,
    ThrustModeArray,
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

        # TODO: These aren't in the engine database. They are calculated in the
        # old Matlab code, but that code doesn't seem to have been carried over
        # to the new AEIC. Where should they come from?
        #
        #     'PMnvolEI_best_ICAOthrust':
        #         np.array([0.05, 0.07, 0.09, 0.12], dtype=float),
        #     'PMnvolEI_new_ICAOthrust':
        #         np.array([0.04, 0.06, 0.08, 0.11], dtype=float),
        #     'PMnvolEIN_best_ICAOthrust': np.array(
        #         [1.1e13, 1.2e13, 1.3e13, 1.4e13], dtype=float
        #     ),
        #     'EImass_max_alt': 0.85,
        # }
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
    assert len(emissions.species) == len(Species)


def _expected_scope11_mapping(perf_model, thrust_categories):
    edb = perf_model.edb
    mass_modes = calculate_PMnvolEI_scope11(
        edb.SN_matrix, edb.engine_type, edb.BP_Ratio
    )
    # TODO: This doesn't exist. Where is this supposed to be coming from?
    # number_modes = edb.PMnvolEIN_best_ICAOthrust
    number_modes = None
    mass = mass_modes.broadcast(thrust_categories)
    number = (
        number_modes.broadcast(thrust_categories) if number_modes is not None else None
    )
    return mass, number


def _expected_trajectory_indices(perf_model, trajectory):
    idx_slice = _trajectory_slice(trajectory)
    lto_inputs = perf_model.lto

    fuel_flow = trajectory.fuel_flow[idx_slice]
    thrust_categories = get_thrust_cat_cruise(fuel_flow, lto_inputs.fuel_flow)

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

    if config.emissions.pmvol_method is PMvolMethod.FUEL_FLOW:
        pmvol, ocic = EI_PMvol_FuelFlow(fuel_flow, thrust_categories)
    else:
        thrust_pct = _thrust_percentages_from_categories(thrust_categories)
        pmvol, ocic = EI_PMvol_FOA3(thrust_pct, expected[Species.HC])
    expected[Species.PMvol] = pmvol
    expected[Species.OCic] = ocic

    if config.emissions.pmnvol_method is PMnvolMethod.SCOPE11:
        mass, number = _expected_scope11_mapping(perf_model, thrust_categories)
        expected[Species.PMnvol] = mass
        expected[Species.PMnvolGMD] = np.zeros_like(fuel_flow)
        if Species.PMnvolN in config.emissions.enabled_species and number is not None:
            expected[Species.PMnvolN] = number
    else:
        expected[Species.PMnvolGMD] = np.zeros_like(fuel_flow)

    return expected, (no_prop, no2_prop, hono_prop)


@pytest.mark.config_updates(emissions__pmnvol_method='scope11')
def test_emit_matches_expected_indices_and_pointwise(perf_model, trajectory, emissions):
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
    assert emissions.lifecycle_co2 is not None


@pytest.mark.config_updates(
    emissions__pmnvol_method='scope11', emissions__lifecycle_enabled=False
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
    output = compute_emissions(perf_model, fuel, trajectory)
    for m in [ThrustMode.APPROACH, ThrustMode.CLIMB]:
        assert all(
            np.isclose(output.lto_emissions[species][m], 0.0)
            for species in output.lto_emissions
        )


@pytest.mark.config_updates(emissions__climb_descent_usage=ClimbDescentMode.LTO)
def test_lto_respects_traj_flag_false(perf_model, fuel, trajectory):
    output = compute_emissions(perf_model, fuel, trajectory)
    for m in [ThrustMode.APPROACH, ThrustMode.CLIMB, ThrustMode.TAKEOFF]:
        assert any(
            np.isclose(output.lto_emissions[species][m], 0.0)
            for species in output.lto_emissions
        )


def test_lto_nox_split_matches_speciation(emissions):
    speciation = NOx_speciation()

    def check(species: Species, prop: ThrustModeValues):
        v1 = emissions.lto_indices[species]
        v2 = emissions.lto_indices[Species.NOx] * prop
        np.testing.assert_allclose(v1.as_array(), v2.as_array())

    check(Species.NO, speciation.no)
    check(Species.NO2, speciation.no2)
    check(Species.HONO, speciation.hono)


@pytest.mark.config_updates(emissions__pmvol_method='foa3')
def test_calculate_pmvol_requires_hc_for_foa3(trajectory):
    thrust_modes = ThrustModeArray(np.array([ThrustMode.APPROACH] * len(trajectory)))
    with pytest.raises(RuntimeError):
        _ = _calculate_EI_PMvol(thrust_modes, trajectory.fuel_flow, None)


@pytest.mark.config_updates(emissions__pmnvol_method='scope11')
def test_calculate_pmnvol_scope11_populates_fields(perf_model, trajectory):
    thrust_modes = ThrustModeArray(np.array([ThrustMode.CLIMB] * len(trajectory)))
    result = _calculate_EI_PMnvol(
        perf_model,
        thrust_modes,
        trajectory.altitude,
    )
    traj_len = len(trajectory)
    expected_mass = np.full(traj_len, 0.07311027)
    expected_number = np.full(traj_len, 1.3e13)
    np.testing.assert_allclose(result[Species.PMnvol], expected_mass)
    np.testing.assert_allclose(
        result[Species.PMnvolGMD], np.zeros(traj_len, dtype=float)
    )
    if Species.PMnvolN in result:
        np.testing.assert_allclose(result[Species.PMnvolN], expected_number)


@pytest.mark.config_updates(emissions__pmnvol_method='meem')
def test_calculate_pmnvol_meem_populates_fields(perf_model, trajectory):
    atmos = AtmosphericState(trajectory.altitude, trajectory.true_airspeed)
    thrust_modes = ThrustModeArray(np.array([ThrustMode.APPROACH] * len(trajectory)))
    result = _calculate_EI_PMnvol(perf_model, thrust_modes, trajectory.altitude, atmos)
    expected_gmd = np.array(
        [40.0, 38.4671063303, 35.9228905331, 30.6483151751, 20.0, 20.0]
    )
    expected_mass = np.array(
        [
            0.008296638,
            0.0073855157,
            0.0060141937,
            0.0048486474,
            0.0031337246,
            0.0057335216,
        ]
    )
    expected_number = np.array(
        [
            2.9357334337e14,
            2.6122814630e14,
            2.0133216668e14,
            1.4705704051e14,
            1.2534898598e14,
            2.2714835290e14,
        ]
    )
    np.testing.assert_allclose(result[Species.PMnvolGMD], expected_gmd)
    np.testing.assert_allclose(result[Species.PMnvol], expected_mass)
    if Species.PMnvolN in result:
        np.testing.assert_allclose(result[Species.PMnvolN], expected_number)


def test_atmospheric_state_and_sls_flow_shapes(perf_model, trajectory):
    atmos = AtmosphericState(trajectory.altitude, trajectory.true_airspeed)
    expected_temp = np.array([288.15, 278.4, 249.15, 216.65, 229.65, 275.15])
    expected_pressure = np.array(
        [
            101325.0,
            84555.9940737564,
            47181.0021852292,
            22632.0400950078,
            30742.4326120969,
            79495.201934051,
        ]
    )
    expected_mach = np.array(
        [
            0.3526362622,
            0.4484475741,
            0.6004518548,
            0.7116967515,
            0.5925081326,
            0.4210157206,
        ]
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
    result = get_GSE_emissions(AircraftClass.WIDE, fuel).emissions
    expected = SpeciesValues[float](
        {
            Species.CO2: 58_000.0,
            Species.NOx: 900.0,
            Species.HC: 70.0,
            Species.CO: 300.0,
            Species.H2O: 22669.67201166181,
            Species.NO: 810.0,
            Species.NO2: 81.0,
            Species.HONO: 9.0,
            Species.SO4: 0.0003,
            Species.SO2: 0.0098,
            Species.PMvol: 27.49985,
            Species.PMnvol: 27.49985,
            Species.PMnvolGMD: 0.0,
            Species.OCic: 0.0,
        }
    )
    for species, value in expected.items():
        assert result[species] == pytest.approx(value)
    if Species.PMnvolN in result:
        assert result[Species.PMnvolN] == pytest.approx(0.0)
