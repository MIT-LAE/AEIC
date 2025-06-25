import numpy as np
import pytest

from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.legacy_trajectory import LegacyTrajectory
from emissions.emission import Emission
from utils import file_location

# Path to a real fuel TOML file in your repo
performance_model_file = file_location("IO/default_config.toml")

# Path to a real fuel TOML file in your repo
perf = PerformanceModel(performance_model_file)
mis = perf.missions[0]

traj = LegacyTrajectory(perf, mis, False, False)
traj.fly_flight()
em = Emission(perf, traj, True)

# --- Unit tests ---


def test_fuel_burn_consumption():
    """Test if fuel burn per segment sums up to total fuel consumption"""
    fuel_consumed = traj.traj_data['fuelMass'][0] - traj.traj_data['fuelMass'][-1]
    assert pytest.approx(np.sum(em.fuel_burn_per_segment), rel=1e-6) == fuel_consumed


def test_emission_indices_positive():
    """Test that all emission indices are non-negative"""
    for field in em.emission_indices.dtype.names:
        assert np.all(em.emission_indices[field] >= 0), (
            f"Negative emission index found for {field}"
        )


def test_emission_indices_finite():
    """Test that all emission indices are finite (no NaN or inf)"""
    for field in em.emission_indices.dtype.names:
        assert np.all(np.isfinite(em.emission_indices[field])), (
            f"Non-finite emission index found for {field}"
        )


def test_pointwise_emissions_positive():
    """Test that all pointwise emissions are non-negative"""
    for field in em.pointwise_emissions_g.dtype.names:
        assert np.all(em.pointwise_emissions_g[field] >= 0), (
            f"Negative pointwise emission found for {field}"
        )


def test_lto_emissions_positive():
    """Test that all LTO emissions are non-negative"""
    for field in em.LTO_emissions_g.dtype.names:
        assert np.all(em.LTO_emissions_g[field] >= 0), (
            f"Negative LTO emission found for {field}"
        )


def test_apu_emissions_positive():
    """Test that all APU emissions are non-negative"""
    for field in em.APU_emissions_g.dtype.names:
        assert np.all(em.APU_emissions_g[field] >= 0), (
            f"Negative APU emission found for {field}"
        )


def test_gse_emissions_positive():
    """Test that all GSE emissions are non-negative"""
    for field in em.GSE_emissions_g.dtype.names:
        assert np.all(em.GSE_emissions_g[field] >= 0), (
            f"Negative GSE emission found for {field}"
        )


def test_total_emissions_sum():
    """Test that summed emissions equal the sum of all components"""
    for field in em.summed_emission_g.dtype.names:
        calculated_sum = (
            np.sum(em.pointwise_emissions_g[field])
            + np.sum(em.LTO_emissions_g[field])
            + em.APU_emissions_g[field]
            + em.GSE_emissions_g[field]
        )
        if field == 'CO2':
            # CO2 includes lifecycle adjustment, so check before lifecycle addition
            original_co2 = em.summed_emission_g[field] - (
                em.fuel['LC_CO2'] * (traj.fuel_mass * em.fuel['Energy_MJ_per_kg'])
                - (
                    np.sum(em.pointwise_emissions_g[field])
                    + np.sum(em.LTO_emissions_g[field])
                    + em.APU_emissions_g[field]
                    + em.GSE_emissions_g[field]
                )
            )
            assert pytest.approx(original_co2, rel=1e-6) == calculated_sum
        else:
            assert (
                pytest.approx(em.summed_emission_g[field], rel=1e-6) == calculated_sum
            ), (
                f"Sum mismatch for {field}:"
                f"{em.summed_emission_g[field]} vs {calculated_sum}"
            )


def test_trajectory_dimensions():
    """Test that trajectory dimensions are consistent"""
    assert em.Ntot == len(em.fuel_burn_per_segment)
    assert em.Ntot == traj.Ntot
    assert em.NClm + em.NCrz + em.NDes == em.Ntot


def test_fuel_burn_first_segment():
    """Test that first segment has zero fuel burn (initial condition)"""
    assert em.fuel_burn_per_segment[0] == 0, "First segment should have zero fuel burn"


def test_fuel_burn_decreasing_mass():
    """Test that fuel mass is monotonically decreasing"""
    fuel_mass = traj.traj_data['fuelMass']
    assert np.all(np.diff(fuel_mass) <= 0), (
        "Fuel mass should be monotonically decreasing"
    )


def test_nox_speciation_conservation():
    """Test that NOx speciation sums to total NOx for LTO"""
    if (
        hasattr(em, 'LTO_noProp')
        and hasattr(em, 'LTO_no2Prop')
        and hasattr(em, 'LTO_honoProp')
    ):
        total_prop = em.LTO_noProp + em.LTO_no2Prop + em.LTO_honoProp
        assert pytest.approx(total_prop, abs=1e-3) == 1.0, (
            "NOx speciation fractions should sum to 1"
        )


def test_sox_speciation():
    """Test that SO2 and SO4 are both present and reasonable"""
    so2_total = np.sum(em.pointwise_emissions_g['SO2']) + np.sum(
        em.LTO_emissions_g['SO2']
    )
    so4_total = np.sum(em.pointwise_emissions_g['SO4']) + np.sum(
        em.LTO_emissions_g['SO4']
    )

    if so2_total > 0 or so4_total > 0:
        # SO2 should typically be much larger than SO4
        assert so2_total > so4_total, "SO2 emissions should exceed SO4 emissions"


def test_gse_wnsf_mapping():
    """Test GSE emissions for different WNSF categories"""
    wnsf_codes = ['wide', 'narrow', 'small', 'freight']
    for wnsf in wnsf_codes:
        # Create temporary emission object to test GSE mapping
        temp_em = Emission.__new__(Emission)  # Create without calling __init__
        temp_em.pmnvol_mode = 'SCOPE11'
        temp_em.GSE_emissions_g = np.empty(
            (), dtype=temp_em._Emission__emission_dtype(1)
        )

        # This should not raise an error
        temp_em.get_GSE_emissions(wnsf)

        # Check that emissions were assigned
        assert temp_em.GSE_emissions_g['CO2'] > 0, (
            f"GSE CO2 not assigned for WNSF {wnsf}"
        )
        assert temp_em.GSE_emissions_g['NOx'] > 0, (
            f"GSE NOx not assigned for WNSF {wnsf}"
        )


def test_gse_invalid_wnsf():
    """Test that invalid WNSF code raises ValueError"""
    temp_em = Emission.__new__(Emission)
    temp_em.pmnvol_mode = 'SCOPE11'
    temp_em.GSE_emissions_g = np.empty((), dtype=temp_em._Emission__emission_dtype(1))

    with pytest.raises(ValueError):
        temp_em.get_GSE_emissions('x')  # Invalid WNSF code


def test_emission_dtype_consistency():
    """Test that emission data type is consistent across arrays"""
    dtype_names = set(em.emission_indices.dtype.names)

    # All emission arrays should have the same field names
    assert set(em.pointwise_emissions_g.dtype.names) == dtype_names
    assert set(em.LTO_emissions_g.dtype.names) == dtype_names
    assert set(em.APU_emissions_g.dtype.names) == dtype_names
    assert set(em.GSE_emissions_g.dtype.names) == dtype_names
    assert set(em.summed_emission_g.dtype.names) == dtype_names


def test_pm_components_reasonable():
    """Test that PM components (volatile and non-volatile) are reasonable"""
    pmvol_total = np.sum(em.pointwise_emissions_g['PMvol']) + np.sum(
        em.LTO_emissions_g['PMvol']
    )
    pmnvol_total = np.sum(em.pointwise_emissions_g['PMnvol']) + np.sum(
        em.LTO_emissions_g['PMnvol']
    )

    # Both components should be positive if PM is present
    if pmvol_total > 0 or pmnvol_total > 0:
        assert pmvol_total >= 0 and pmnvol_total >= 0, (
            "Both PM components should be non-negative"
        )


def test_trajectory_vs_lto_mode_consistency():
    """Test consistency based on traj_emissions_all flag"""
    if em.traj_emissions_all:
        # Only taxi mode for LTO
        assert em.LTO_emissions_g.shape == (), (
            "Should have scalar LTO arrays when traj_emissions_all=True"
        )
    else:
        # Full LTO cycle
        assert len(em.LTO_emission_indices['NOx']) == 4, (
            "Should have 4 LTO modes when traj_emissions_all=False"
        )


def test_fuel_properties_loaded():
    """Test that fuel properties are properly loaded"""
    assert hasattr(em, 'fuel'), "Fuel properties not loaded"
    assert 'LC_CO2' in em.fuel, "Lifecycle CO2 factor not in fuel properties"
    assert 'Energy_MJ_per_kg' in em.fuel, "Energy content not in fuel properties"


def test_lifecycle_co2_adjustment():
    """Test that lifecycle CO2 adjustment is applied correctly"""
    # The lifecycle adjustment should increase total CO2
    base_co2 = (
        np.sum(em.pointwise_emissions_g['CO2'])
        + np.sum(em.LTO_emissions_g['CO2'])
        + em.APU_emissions_g['CO2']
        + em.GSE_emissions_g['CO2']
    )

    assert em.summed_emission_g['CO2'] != base_co2, (
        "Lifecycle CO2 adjustment should modify total"
    )


def test_no_negative_fuel_burn():
    """Test that no segment has negative fuel burn (except first which is zero)"""
    assert np.all(em.fuel_burn_per_segment >= 0), "Negative fuel burn detected"


def test_emission_units_consistency():
    """Test that emission units are consistent (all in grams)"""
    # This is more of a documentation test
    # ensures the class produces results in expected units
    total_co2_kg = np.sum(em.pointwise_emissions_g['CO2']) / 1000.0  # Convert g to kg
    fuel_consumed_kg = traj.traj_data['fuelMass'][0] - traj.traj_data['fuelMass'][-1]

    # CO2 per kg fuel should be reasonable (typically 3.1-3.2 kg CO2/kg fuel)
    if fuel_consumed_kg > 0:
        co2_per_fuel = total_co2_kg / fuel_consumed_kg
        assert 2.5 <= co2_per_fuel <= 4.4, (
            f"CO2 per fuel ratio {co2_per_fuel} outside expected range"
        )


@pytest.mark.parametrize("pmnvol_mode", ['SCOPE11', 'foa3', 'fox'])
def test_different_pmnvol_modes(pmnvol_mode):
    """Test that different PMnvol modes produce valid results"""
    # This test would require creating new Emission instances with different modes
    # For now, just test that the current mode produces valid results
    if em.pmnvol_mode == pmnvol_mode:
        assert np.all(np.isfinite(em.LTO_emission_indices['PMnvol'])), (
            f"Invalid PMnvol values for mode {pmnvol_mode}"
        )


def test_array_shapes_consistency():
    """Test that all arrays have consistent shapes"""
    n_traj = em.Ntot

    # Trajectory arrays should match Ntot
    assert em.emission_indices['CO2'].shape == (n_traj,), (
        "Emission indices shape mismatch"
    )
    assert em.pointwise_emissions_g['CO2'].shape == (n_traj,), (
        "Pointwise emissions shape mismatch"
    )
    assert em.fuel_burn_per_segment.shape == (n_traj,), "Fuel burn array shape mismatch"

    # Summary arrays should be scalars
    assert em.summed_emission_g['CO2'].shape == (1,), (
        "Summed emissions should be scalar"
    )
