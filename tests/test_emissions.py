import numpy as np
import pytest

from emissions.emission import AtmosphericState, Emission


def _scalar(value: np.ndarray | float) -> float:
    """Convert 0-D/length-1 numpy structures into native floats safely."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise AssertionError("Expected scalar-like value")
    return float(arr.reshape(-1)[0])


class DummyPerformanceModel:
    """Lightweight stand-in for PerformanceModel with deterministic data."""

    def __init__(self, config_overrides=None, edb_overrides=None):
        base_config = {
            'Fuel': 'conventional_jetA',
            'LTO_input_mode': 'EDB',
            'EI_NOx_method': 'BFFM2',
            'EI_HC_method': 'BFFM2',
            'EI_CO_method': 'BFFM2',
            'EI_PMvol_method': 'fuel_flow',
            'EI_PMnvol_method': 'scope11',
            'CO2_calculation': True,
            'H2O_calculation': True,
            'SOx_calculation': True,
            'APU_calculation': True,
            'GSE_calculation': True,
            'LC_calculation': True,
            'climb_descent_usage': True,
        }
        if config_overrides:
            base_config.update(config_overrides)
        self.config = base_config

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
    """Minimal trajectory profile with monotonic fuel depletion."""

    def __init__(self):
        self.NClm = 2
        self.NCrz = 2
        self.NDes = 2
        self.Ntot = self.NClm + self.NCrz + self.NDes
        fuel_mass = np.array([2000.0, 1994.0, 1987.5, 1975.0, 1960.0, 1945.0])
        self.traj_data = {
            'fuelMass': fuel_mass,
            'fuelFlow': np.array([0.3, 0.35, 0.55, 0.65, 0.5, 0.32]),
            'altitude': np.array([0.0, 1500.0, 6000.0, 11000.0, 9000.0, 2000.0]),
            'tas': np.array([120.0, 150.0, 190.0, 210.0, 180.0, 140.0]),
        }
        self.fuel_mass = float(fuel_mass[0])


@pytest.fixture
def trajectory():
    return DummyTrajectory()


@pytest.fixture
def perf_factory():
    def _factory(config_overrides=None, edb_overrides=None):
        return DummyPerformanceModel(config_overrides, edb_overrides)

    return _factory


@pytest.fixture
def emission(perf_factory, trajectory):
    perf = perf_factory()
    em = Emission(perf, trajectory)
    em.compute_emissions()
    return em


def test_emission_settings_parse_string_flags(perf_factory, trajectory):
    perf = perf_factory(
        {
            'APU_calculation': 'no',
            'GSE_calculation': 'Yes',
            'LC_calculation': 'n',
            'EI_PMvol_method': 'FOA3',
        }
    )
    em = Emission(perf, trajectory)
    assert em.apu_enabled is False
    assert em.gse_enabled is True
    assert em.lifecycle_enabled is False
    assert em.pmvol_method == 'foa3'


def test_compute_emissions_populates_active_fields(emission):
    FILL_VALUE = -1.0
    for field in emission._active_fields:
        indices = emission.emission_indices[field]
        assert np.all(indices != FILL_VALUE)
        assert np.all(indices >= 0.0)
        assert np.all(emission.pointwise_emissions_g[field] >= 0.0)


def test_apu_and_gse_outputs_positive(emission):
    for store in (emission.APU_emissions_g, emission.GSE_emissions_g):
        for field in store.dtype.names:
            assert _scalar(store[field]) >= 0.0


def test_total_fuel_burn_positive(emission):
    assert emission.total_fuel_burn > 0.0


def test_sum_total_emissions_matches_components(perf_factory, trajectory):
    perf = perf_factory({'LC_calculation': False})
    em = Emission(perf, trajectory)
    em.compute_emissions()

    for field in em.summed_emission_g.dtype.names:
        expected = (
            np.sum(em.pointwise_emissions_g[field])
            + np.sum(em.LTO_emissions_g[field])
            + _scalar(em.APU_emissions_g[field])
            + _scalar(em.GSE_emissions_g[field])
        )
        assert _scalar(em.summed_emission_g[field]) == pytest.approx(expected, rel=1e-6)


def test_lifecycle_emissions_override_total_co2(emission):
    expected = emission.fuel['LC_CO2'] * (
        emission.trajectory.fuel_mass * emission.fuel['Energy_MJ_per_kg']
    )
    assert _scalar(emission.summed_emission_g['CO2']) == pytest.approx(expected)


def test_scope11_profile_caching(emission):
    profile_first = emission._scope11_profile(emission.performance_model)
    profile_second = emission._scope11_profile(emission.performance_model)
    assert profile_first is profile_second
    assert (
        profile_first['mass'].shape
        == emission.performance_model.EDB_data['SN_matrix'].shape
    )


def test_get_lto_tims_respects_traj_flag(perf_factory, trajectory):
    em_all = Emission(perf_factory({'climb_descent_usage': True}), trajectory)
    durations_all = em_all._get_LTO_TIMs()
    assert np.allclose(durations_all[1:3], 0.0)

    em_partial = Emission(perf_factory({'climb_descent_usage': False}), trajectory)
    durations_partial = em_partial._get_LTO_TIMs()
    assert np.all(durations_partial[1:3] > 0.0)


def test_wnsf_index_mapping_and_errors(perf_factory, trajectory):
    em = Emission(perf_factory(), trajectory)
    assert em._wnsf_index('wide') == 0
    assert em._wnsf_index('FREIGHT') == 3
    with pytest.raises(ValueError):
        em._wnsf_index('unknown')


def test_calculate_pmvol_requires_hc_for_foa3(perf_factory, trajectory):
    em = Emission(perf_factory({'EI_PMvol_method': 'foa3'}), trajectory)
    idx_slice = em._trajectory_slice()
    fuel_flow = em.trajectory.traj_data['fuelFlow'][idx_slice]
    thrust_categories = np.ones_like(fuel_flow, dtype=int)
    with pytest.raises(RuntimeError):
        em._calculate_EI_PMvol(
            idx_slice,
            thrust_categories,
            fuel_flow,
            None,
        )


def test_calculate_pmnvol_scope11_populates_fields(perf_factory, trajectory):
    em = Emission(perf_factory({'EI_PMnvol_method': 'scope11'}), trajectory)
    idx_slice = em._trajectory_slice()
    thrust_categories = np.ones_like(
        em.trajectory.traj_data['fuelFlow'][idx_slice], dtype=int
    )
    em._calculate_EI_PMnvol(
        idx_slice,
        thrust_categories,
        em.trajectory.traj_data['altitude'][idx_slice],
        AtmosphericState(None, None, None),
        em.performance_model,
    )
    assert np.all(em.emission_indices['PMnvol'][idx_slice] >= 0.0)
    assert np.all(em.emission_indices['PMnvolGMD'][idx_slice] == 0.0)


def test_calculate_pmnvol_meem_populates_fields(perf_factory, trajectory):
    em = Emission(perf_factory({'EI_PMnvol_method': 'meem'}), trajectory)
    idx_slice = em._trajectory_slice()
    altitudes = em.trajectory.traj_data['altitude'][idx_slice]
    tas = em.trajectory.traj_data['tas'][idx_slice]
    atmos = em._atmospheric_state(altitudes, tas, True)
    thrust_categories = np.ones_like(
        em.trajectory.traj_data['fuelFlow'][idx_slice], dtype=int
    )
    em._calculate_EI_PMnvol(
        idx_slice, thrust_categories, altitudes, atmos, em.performance_model
    )
    assert np.all(em.emission_indices['PMnvol'][idx_slice] >= 0.0)
    assert np.all(em.emission_indices['PMnvolGMD'][idx_slice] >= 0.0)
    if 'PMnvolN' in em.emission_indices.dtype.names:
        assert np.all(em.emission_indices['PMnvolN'][idx_slice] >= 0.0)


def test_compute_ei_nox_requires_inputs(perf_factory, trajectory):
    em = Emission(perf_factory(), trajectory)
    idx_slice = em._trajectory_slice()
    lto_inputs = em._extract_lto_inputs()
    with pytest.raises(RuntimeError):
        em.compute_EI_NOx(
            idx_slice, lto_inputs, AtmosphericState(None, None, None), None
        )


def test_atmospheric_state_and_sls_flow_shapes(perf_factory, trajectory):
    em = Emission(perf_factory(), trajectory)
    idx_slice = em._trajectory_slice()
    altitudes = em.trajectory.traj_data['altitude'][idx_slice]
    tas = em.trajectory.traj_data['tas'][idx_slice]
    atmos = em._atmospheric_state(altitudes, tas, True)
    assert atmos.temperature.shape == altitudes.shape
    assert atmos.pressure.shape == altitudes.shape
    assert atmos.mach.shape == altitudes.shape

    fuel_flow = em.trajectory.traj_data['fuelFlow'][idx_slice]
    sls_flow = em._sls_equivalent_fuel_flow(True, fuel_flow, atmos)
    assert sls_flow.shape == fuel_flow.shape
    assert em._sls_equivalent_fuel_flow(False, fuel_flow, atmos) is None


def test_get_gse_emissions_assigns_all_species(perf_factory, trajectory):
    em = Emission(perf_factory(), trajectory)
    em.get_GSE_emissions('wide')
    for field in ('CO2', 'NOx', 'HC', 'CO', 'PMvol', 'PMnvol', 'H2O', 'SO2', 'SO4'):
        assert _scalar(em.GSE_emissions_g[field]) >= 0.0


def test_get_gse_emissions_invalid_code(perf_factory, trajectory):
    em = Emission(perf_factory(), trajectory)
    with pytest.raises(ValueError):
        em.get_GSE_emissions('bad')


def test_emission_dtype_consistency(emission):
    dtype_names = set(emission.emission_indices.dtype.names)
    assert set(emission.pointwise_emissions_g.dtype.names) == dtype_names
    assert set(emission.LTO_emissions_g.dtype.names) == dtype_names
    assert set(emission.APU_emissions_g.dtype.names) == dtype_names
    assert set(emission.GSE_emissions_g.dtype.names) == dtype_names
    assert set(emission.summed_emission_g.dtype.names) == dtype_names
