from __future__ import annotations

import numpy as np
import pytest

from AEIC.config import config
from AEIC.emissions.EI_HCCO import EI_HCCO
from AEIC.emissions.EI_NOx import BFFM2_EINOx, NOx_speciation
from AEIC.emissions.EI_PMnvol import calculate_PMnvolEI_scope11
from AEIC.emissions.emission import (
    AtmosphericState,
    EI_PMvol_FOA3,
    EI_PMvol_FuelFlow,
    Emission,
    PMnvolMethod,
    PMvolMethod,
)
from AEIC.missions import Mission
from AEIC.performance_model import PerformanceModel
from AEIC.utils.helpers import iso_to_timestamp
from AEIC.utils.standard_fuel import get_thrust_cat


@pytest.fixture
def performance_model_file():
    # Path to a real fuel TOML file in your repo
    return config.file_location("performance/sample_performance_model.toml")


@pytest.fixture
def performance_model(performance_model_file):
    # Path to a real fuel TOML file in your repo
    return PerformanceModel(performance_model_file)


sample_mission = Mission(
    origin="BOS",
    destination="LAX",
    aircraft_type="738",
    departure=iso_to_timestamp('2019-01-01 12:00:00'),
    arrival=iso_to_timestamp('2019-01-01 18:00:00'),
    load_factor=1.0,
)


class DummyPerformanceModel:
    def __init__(self, edb_overrides=None, lto_settings=None):
        base_edb = {
            'fuelflow_KGperS': np.array([0.25, 0.5, 0.9, 1.2], dtype=float),
            'NOX_EI_matrix': np.array([8.0, 12.0, 26.0, 32.0], dtype=float),
            'HC_EI_matrix': np.array([4.0, 3.0, 2.0, 1.0], dtype=float),
            'CO_EI_matrix': np.array([20.0, 15.0, 10.0, 5.0], dtype=float),
            'PMnvolEI_best_ICAOthrust': np.array([0.05, 0.07, 0.09, 0.12], dtype=float),
            'PMnvolEI_new_ICAOthrust': np.array([0.04, 0.06, 0.08, 0.11], dtype=float),
            'PMnvolEIN_best_ICAOthrust': np.array(
                [1.1e13, 1.2e13, 1.3e13, 1.4e13], dtype=float
            ),
            'SN_matrix': np.array([6.0, 8.0, 11.0, 13.0], dtype=float),
            'PR': np.array([22.0, 22.0, 22.0, 22.0], dtype=float),
            'ENGINE_TYPE': 'TF',
            'BP_Ratio': np.array([5.0, 5.0, 5.0, 5.0], dtype=float),
            'nvPM_mass_matrix': np.array([5.0, 5.5, 6.0, 6.5], dtype=float),
            'nvPM_num_matrix': np.array([2.0e14, 2.1e14, 2.2e14, 2.3e14], dtype=float),
            'EImass_max': 8.0,
            'EImass_max_thrust': 0.575,
            'EInum_max': 2.4e14,
            'EInum_max_thrust': 0.575,
            'EImass_max_alt': 0.85,
        }
        if edb_overrides:
            base_edb.update(edb_overrides)
        self.EDB_data = base_edb
        default_lto_settings = {
            'takeoff': {
                'FUEL_KGs': 1.2,
                'EI_NOx': 40.0,
                'EI_HC': 1.0,
                'EI_CO': 2.0,
                'THRUST_FRAC': 1.0,
            },
            'climb': {
                'FUEL_KGs': 0.9,
                'EI_NOx': 32.0,
                'EI_HC': 1.5,
                'EI_CO': 3.0,
                'THRUST_FRAC': 0.85,
            },
            'approach': {
                'FUEL_KGs': 0.5,
                'EI_NOx': 12.0,
                'EI_HC': 3.0,
                'EI_CO': 10.0,
                'THRUST_FRAC': 0.30,
            },
            'idle': {
                'FUEL_KGs': 0.25,
                'EI_NOx': 8.0,
                'EI_HC': 4.0,
                'EI_CO': 20.0,
                'THRUST_FRAC': 0.07,
            },
        }
        self.LTO_data = {
            'thrust_settings': lto_settings
            if lto_settings is not None
            else default_lto_settings
        }
        self.APU_data = {
            'fuel_kg_per_s': 0.03,
            'PM10_g_per_kg': 0.4,
            'NOx_g_per_kg': 0.05,
            'HC_g_per_kg': 0.02,
            'CO_g_per_kg': 0.03,
        }
        self.model_info = {
            'General_Information': {'aircraft_class': 'wide', 'n_eng': 2},
        }


class DummyTrajectory:
    def __init__(self):
        self.n_climb = 2
        self.n_cruise = 2
        self.n_descent = 2
        self.X_npoints = self.n_climb + self.n_cruise + self.n_descent
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


@pytest.fixture
def trajectory():
    return DummyTrajectory()


@pytest.fixture
def sample_perf_model():
    def _factory(edb_overrides=None, lto_settings=None):
        return DummyPerformanceModel(edb_overrides, lto_settings)

    return _factory


@pytest.fixture
def emission_with_run(sample_perf_model, trajectory):
    perf = sample_perf_model()
    emission = Emission(perf)
    output = emission.emit(trajectory)
    return emission, output, trajectory


def _map_modes_to_categories(mode_values: np.ndarray, thrust_categories: np.ndarray):
    values = np.asarray(mode_values, dtype=float).ravel()
    mapped = np.full(thrust_categories.shape, values[-1], dtype=float)
    mapped[thrust_categories == 2] = values[0]
    if values.size > 1:
        mapped[thrust_categories == 3] = values[1]
    if values.size > 2:
        mapped[thrust_categories == 1] = values[2]
    return mapped


def _expected_scope11_mapping(performance_model, thrust_categories):
    edb = performance_model.EDB_data
    mass_modes = calculate_PMnvolEI_scope11(
        np.array(edb['SN_matrix']),
        np.array(edb['PR']),
        edb['ENGINE_TYPE'],
        np.array(edb['BP_Ratio']),
    )
    number_modes = edb.get('PMnvolEIN_best_ICAOthrust')
    mass = _map_modes_to_categories(mass_modes, thrust_categories)
    number = (
        _map_modes_to_categories(np.array(number_modes), thrust_categories)
        if number_modes is not None
        else None
    )
    return mass, number


def _expected_trajectory_indices(emission, trajectory):
    idx_slice = emission._trajectory_slice()
    lto_inputs = emission._extract_lto_inputs()

    fuel_flow = trajectory.fuel_flow[idx_slice]
    thrust_categories = get_thrust_cat(
        fuel_flow, lto_inputs['fuel_flow'], cruiseCalc=True
    )

    altitudes = trajectory.altitude[idx_slice]
    tas = trajectory.true_airspeed[idx_slice]
    atmos = emission._atmospheric_state(altitudes, tas, True)
    sls_flow = emission._sls_equivalent_fuel_flow(True, fuel_flow, atmos)

    expected = {
        'CO2': np.full_like(fuel_flow, 3155.6),
        'H2O': np.full_like(fuel_flow, 1233.3865),
        'SO2': np.full_like(fuel_flow, 1.176),
        'SO4': np.full_like(fuel_flow, 0.036),
    }

    nox_result = BFFM2_EINOx(
        sls_equiv_fuel_flow=sls_flow,
        NOX_EI_matrix=lto_inputs['nox_ei'],
        fuelflow_performance=lto_inputs['fuel_flow'],
        Tamb=atmos.temperature,
        Pamb=atmos.pressure,
    )
    expected['NOx'] = nox_result.NOxEI
    expected['NO'] = nox_result.NOEI
    expected['NO2'] = nox_result.NO2EI
    expected['HONO'] = nox_result.HONOEI
    no_prop = nox_result.noProp
    no2_prop = nox_result.no2Prop
    hono_prop = nox_result.honoProp

    expected['HC'] = EI_HCCO(
        sls_flow,
        lto_inputs['hc_ei'],
        lto_inputs['fuel_flow'],
        Tamb=atmos.temperature,
        Pamb=atmos.pressure,
        cruiseCalc=True,
    )
    expected['CO'] = EI_HCCO(
        sls_flow,
        lto_inputs['co_ei'],
        lto_inputs['fuel_flow'],
        Tamb=atmos.temperature,
        Pamb=atmos.pressure,
        cruiseCalc=True,
    )

    if config.emissions.pmvol_method is PMvolMethod.FUEL_FLOW:
        thrust_labels = np.full(thrust_categories.shape, 'H', dtype='<U1')
        thrust_labels[thrust_categories == 2] = 'L'
        pmvol, ocic = EI_PMvol_FuelFlow(fuel_flow, thrust_labels)
    else:
        thrust_pct = emission._thrust_percentages_from_categories(thrust_categories)
        pmvol, ocic = EI_PMvol_FOA3(thrust_pct, expected['HC'])
    expected['PMvol'] = pmvol
    expected['OCic'] = ocic

    if config.emissions.pmnvol_method is PMnvolMethod.SCOPE11:
        mass, number = _expected_scope11_mapping(
            emission.performance_model, thrust_categories
        )
        expected['PMnvol'] = mass
        expected['PMnvolGMD'] = np.zeros_like(fuel_flow)
        if emission._include_pmnvol_number and number is not None:
            expected['PMnvolN'] = number
    else:
        expected['PMnvolGMD'] = np.zeros_like(fuel_flow)

    return expected, (no_prop, no2_prop, hono_prop)


def _expected_lto_nox_split(emission):
    lto_inputs = emission._extract_lto_inputs()
    thrust_categories = get_thrust_cat(lto_inputs['fuel_flow'], None, cruiseCalc=False)
    return NOx_speciation(thrust_categories)


@pytest.mark.config_updates(emissions__pmnvol_method='scope11')
def test_emit_matches_expected_indices_and_pointwise(emission_with_run):
    emission, output, trajectory = emission_with_run
    expected, (no_prop, no2_prop, hono_prop) = _expected_trajectory_indices(
        emission, trajectory
    )
    idx_slice = emission._trajectory_slice()
    for field, expected_values in expected.items():
        if field not in emission.emission_indices.dtype.names:
            continue
        np.testing.assert_allclose(
            emission.emission_indices[field][idx_slice], expected_values
        )
    fuel_burn = emission.fuel_burn_per_segment[idx_slice]
    for field in emission._active_fields:
        np.testing.assert_allclose(
            emission.pointwise_emissions_g[field][idx_slice],
            emission.emission_indices[field][idx_slice] * fuel_burn,
        )
    np.testing.assert_allclose(
        emission.emission_indices['NO'][idx_slice],
        emission.emission_indices['NOx'][idx_slice] * no_prop,
    )
    np.testing.assert_allclose(
        emission.emission_indices['NO2'][idx_slice],
        emission.emission_indices['NOx'][idx_slice] * no2_prop,
    )
    np.testing.assert_allclose(
        emission.emission_indices['HONO'][idx_slice],
        emission.emission_indices['NOx'][idx_slice] * hono_prop,
    )
    assert output.trajectory.total_fuel_burn == pytest.approx(emission.total_fuel_burn)
    assert output.lifecycle_co2_g is not None


@pytest.mark.config_updates(
    emissions__pmnvol_method='scope11', emissions__lifecycle_enabled=False
)
def test_sum_total_emissions_matches_components(sample_perf_model, trajectory):
    perf = sample_perf_model()
    emission = Emission(perf)
    emission.emit(trajectory)
    for field in emission.summed_emission_g.dtype.names:
        expected = (
            np.sum(emission.pointwise_emissions_g[field])
            + np.sum(emission.LTO_emissions_g[field])
            + np.sum(emission.APU_emissions_g[field])
            + np.sum(emission.GSE_emissions_g[field])
        )
        assert emission.summed_emission_g[field] == pytest.approx(expected)


def test_lifecycle_emissions_require_total_fuel_mass(sample_perf_model):
    emission = Emission(sample_perf_model())
    bad_traj = DummyTrajectory()
    bad_traj.fuel_mass = None
    with pytest.raises(TypeError):
        emission.emit(bad_traj)


def test_scope11_profile_caching(sample_perf_model):
    emission = Emission(sample_perf_model())
    profile_first = emission._scope11_profile(emission.performance_model)
    profile_second = emission._scope11_profile(emission.performance_model)
    assert profile_first is profile_second
    assert (
        profile_first['mass'].shape
        == emission.performance_model.EDB_data['SN_matrix'].shape
    )


@pytest.mark.config_updates(emissions__climb_descent_usage=True)
def test_get_lto_tims_respects_traj_flag_true(sample_perf_model):
    emission_all = Emission(sample_perf_model())
    assert np.allclose(emission_all._get_LTO_TIMs()[1:3], 0.0)


@pytest.mark.config_updates(emissions__climb_descent_usage=False)
def test_get_lto_tims_respects_traj_flag_false(sample_perf_model):
    emission_partial = Emission(sample_perf_model())
    assert np.all(emission_partial._get_LTO_TIMs()[1:3] > 0.0)


def test_lto_nox_split_matches_speciation(emission_with_run):
    emission, _, _ = emission_with_run
    no_prop, no2_prop, hono_prop = _expected_lto_nox_split(emission)
    np.testing.assert_allclose(
        emission.LTO_emission_indices['NO'],
        emission.LTO_emission_indices['NOx'] * no_prop,
    )
    np.testing.assert_allclose(
        emission.LTO_emission_indices['NO2'],
        emission.LTO_emission_indices['NOx'] * no2_prop,
    )
    np.testing.assert_allclose(
        emission.LTO_emission_indices['HONO'],
        emission.LTO_emission_indices['NOx'] * hono_prop,
    )


@pytest.mark.config_updates(lto_input_mode='performance_model')
def test_extract_lto_inputs_orders_performance_modes(sample_perf_model, trajectory):
    lto_settings = {
        'TakeOff': {
            'FUEL_KGs': 1.7,
            'EI_NOx': 45.0,
            'EI_HC': 1.1,
            'EI_CO': 2.2,
            'THRUST_FRAC': 1.0,
        },
        'Climb': {
            'FUEL_KGs': 0.95,
            'EI_NOx': 33.0,
            'EI_HC': 1.4,
            'EI_CO': 2.8,
            'THRUST_FRAC': 0.85,
        },
        'Approach': {
            'FUEL_KGs': 0.55,
            'EI_NOx': 14.0,
            'EI_HC': 2.8,
            'EI_CO': 9.5,
            'THRUST_FRAC': 0.30,
        },
        'Idle': {
            'FUEL_KGs': 0.28,
            'EI_NOx': 9.0,
            'EI_HC': 3.5,
            'EI_CO': 18.0,
            'THRUST_FRAC': 0.07,
        },
    }
    perf = sample_perf_model(lto_settings=lto_settings)
    emission = Emission(perf)
    emission._prepare_run_state(trajectory)
    lto_inputs = emission._extract_lto_inputs()
    assert np.allclose(lto_inputs['fuel_flow'], [0.28, 0.55, 0.95, 1.7])
    assert np.allclose(lto_inputs['nox_ei'], [9.0, 14.0, 33.0, 45.0])
    assert np.allclose(lto_inputs['hc_ei'], [3.5, 2.8, 1.4, 1.1])
    assert np.allclose(lto_inputs['co_ei'], [18.0, 9.5, 2.8, 2.2])
    assert np.allclose(lto_inputs['thrust_pct'], [7.0, 30.0, 85.0, 100.0])


@pytest.mark.config_updates(emissions__pmvol_method='foa3')
def test_pmvol_foa3_uses_thrust_percentages(monkeypatch, sample_perf_model, trajectory):
    print(config.emissions)
    perf = sample_perf_model({'EI_PMvol_method': 'foa3'})
    emission = Emission(perf)
    emission._prepare_run_state(trajectory)
    idx_slice = emission._trajectory_slice()
    thrust_categories = np.array([1, 2, 3, 1, 2, 3])
    hc_ei = np.linspace(1.0, 2.5, thrust_categories.size)
    captured = {}

    def fake_foa3(thrusts, hc_values):
        captured['thrusts'] = thrusts.copy()
        captured['hc'] = hc_values.copy()
        return np.zeros_like(hc_values), np.zeros_like(hc_values)

    monkeypatch.setattr('AEIC.emissions.emission.EI_PMvol_FOA3', fake_foa3)
    emission._calculate_EI_PMvol(
        idx_slice,
        thrust_categories,
        trajectory.fuel_flow[idx_slice],
        hc_ei,
    )
    expected = emission._thrust_percentages_from_categories(thrust_categories)
    np.testing.assert_allclose(captured['thrusts'], expected)
    np.testing.assert_allclose(captured['hc'], hc_ei)


def test_wnsf_index_mapping_and_errors(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model())
    emission._prepare_run_state(trajectory)
    assert emission._wnsf_index('wide') == 0
    assert emission._wnsf_index('FREIGHT') == 3
    with pytest.raises(ValueError):
        emission._wnsf_index('unknown')


@pytest.mark.config_updates(emissions__pmvol_method='foa3')
def test_calculate_pmvol_requires_hc_for_foa3(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model({'EI_PMvol_method': 'foa3'}))
    emission._prepare_run_state(trajectory)
    idx_slice = emission._trajectory_slice()
    fuel_flow = trajectory.fuel_flow[idx_slice]
    thrust_categories = np.ones_like(fuel_flow, dtype=int)
    with pytest.raises(RuntimeError):
        emission._calculate_EI_PMvol(idx_slice, thrust_categories, fuel_flow, None)


@pytest.mark.config_updates(emissions__pmnvol_method='scope11')
def test_calculate_pmnvol_scope11_populates_fields(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model({'EI_PMnvol_method': 'scope11'}))
    emission._prepare_run_state(trajectory)
    idx_slice = emission._trajectory_slice()
    thrust_categories = np.ones_like(trajectory.fuel_flow[idx_slice], dtype=int)
    emission._calculate_EI_PMnvol(
        idx_slice,
        thrust_categories,
        trajectory.altitude[idx_slice],
        AtmosphericState(None, None, None),
        emission.performance_model,
    )
    traj_len = idx_slice.stop - idx_slice.start
    expected_mass = np.full(traj_len, 0.07311027)
    expected_number = np.full(traj_len, 1.3e13)
    np.testing.assert_allclose(
        emission.emission_indices['PMnvol'][idx_slice], expected_mass
    )
    np.testing.assert_allclose(
        emission.emission_indices['PMnvolGMD'][idx_slice],
        np.zeros(traj_len, dtype=float),
    )
    if 'PMnvolN' in emission.emission_indices.dtype.names:
        np.testing.assert_allclose(
            emission.emission_indices['PMnvolN'][idx_slice], expected_number
        )


@pytest.mark.config_updates(emissions__pmnvol_method='meem')
def test_calculate_pmnvol_meem_populates_fields(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model({'EI_PMnvol_method': 'meem'}))
    emission._prepare_run_state(trajectory)
    idx_slice = emission._trajectory_slice()
    altitudes = trajectory.altitude[idx_slice]
    tas = trajectory.true_airspeed[idx_slice]
    atmos = emission._atmospheric_state(altitudes, tas, True)
    thrust_categories = np.ones_like(trajectory.fuel_flow[idx_slice], dtype=int)
    emission._calculate_EI_PMnvol(
        idx_slice,
        thrust_categories,
        altitudes,
        atmos,
        emission.performance_model,
    )
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
    np.testing.assert_allclose(
        emission.emission_indices['PMnvolGMD'][idx_slice], expected_gmd
    )
    np.testing.assert_allclose(
        emission.emission_indices['PMnvol'][idx_slice], expected_mass
    )
    if 'PMnvolN' in emission.emission_indices.dtype.names:
        np.testing.assert_allclose(
            emission.emission_indices['PMnvolN'][idx_slice], expected_number
        )


def test_compute_ei_nox_requires_inputs(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model())
    emission._prepare_run_state(trajectory)
    idx_slice = emission._trajectory_slice()
    lto_inputs = emission._extract_lto_inputs()
    with pytest.raises(RuntimeError):
        emission.compute_EI_NOx(
            idx_slice, lto_inputs, AtmosphericState(None, None, None), None
        )


def test_atmospheric_state_and_sls_flow_shapes(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model())
    idx_slice = slice(0, trajectory.X_npoints)
    altitudes = trajectory.altitude[idx_slice]
    tas = trajectory.true_airspeed[idx_slice]
    atmos = emission._atmospheric_state(altitudes, tas, True)
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

    fuel_flow = trajectory.fuel_flow[idx_slice]
    sls_flow = emission._sls_equivalent_fuel_flow(True, fuel_flow, atmos)
    expected_sls = np.array(
        [
            0.153777,
            0.203788,
            0.474551,
            0.910231,
            0.561445,
            0.192661,
        ]
    )
    np.testing.assert_allclose(sls_flow, expected_sls, atol=1e-5)
    assert emission._sls_equivalent_fuel_flow(False, fuel_flow, atmos) is None


def test_get_gse_emissions_matches_reference_profile(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model())
    emission._prepare_run_state(trajectory)
    emission.get_GSE_emissions('wide')
    expected = {
        'CO2': 58_000.0,
        'NOx': 900.0,
        'HC': 70.0,
        'CO': 300.0,
        'H2O': 22669.67201166181,
        'NO': 810.0,
        'NO2': 81.0,
        'HONO': 9.0,
        'SO4': 0.0003,
        'SO2': 0.0098,
        'PMvol': 27.49985,
        'PMnvol': 27.49985,
        'PMnvolGMD': 0.0,
        'OCic': 0.0,
    }
    for field, value in expected.items():
        assert emission.GSE_emissions_g[field] == pytest.approx(value)
    if 'PMnvolN' in emission.GSE_emissions_g.dtype.names:
        assert emission.GSE_emissions_g['PMnvolN'] == pytest.approx(0.0)


def test_get_gse_emissions_invalid_code(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model())
    emission._prepare_run_state(trajectory)
    with pytest.raises(ValueError):
        emission.get_GSE_emissions('bad')


def test_emission_dtype_consistency(sample_perf_model, trajectory):
    emission = Emission(sample_perf_model())
    emission._prepare_run_state(trajectory)
    dtype_names = set(emission.emission_indices.dtype.names)
    assert set(emission.pointwise_emissions_g.dtype.names) == dtype_names
    assert set(emission.LTO_emissions_g.dtype.names) == dtype_names
    assert set(emission.APU_emissions_g.dtype.names) == dtype_names
    assert set(emission.GSE_emissions_g.dtype.names) == dtype_names
    assert set(emission.summed_emission_g.dtype.names) == dtype_names
