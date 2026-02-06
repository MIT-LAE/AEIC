import tomllib
from unittest.mock import patch

import numpy as np
import pytest

from AEIC.config import config
from AEIC.emissions.apu import get_APU_emissions
from AEIC.emissions.ei.hcco import EI_HCCO
from AEIC.emissions.ei.nox import BFFM2_EINOx, BFFM2EINOxResult, NOx_speciation
from AEIC.emissions.ei.pmnvol import PMnvol_MEEM, calculate_PMnvolEI_scope11
from AEIC.emissions.ei.pmvol import EI_PMvol_FOA3, EI_PMvol_FuelFlow
from AEIC.emissions.ei.sox import EI_SOx, SOxEmissionResult
from AEIC.performance.apu import APU
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import ThrustMode, ThrustModeArray, ThrustModeValues
from AEIC.types import Fuel, Species, SpeciesValues


@pytest.fixture
def fuel_jetA():
    with open(config.file_location('fuels/conventional_jetA.toml'), 'rb') as f:
        return Fuel.model_validate(tomllib.load(f))


@pytest.fixture
def fuel_SAF():
    with open(config.file_location("fuels/SAF.toml"), 'rb') as f:
        return Fuel.model_validate(tomllib.load(f))


class TestEI_HCCO:
    """Tests for EI_HCCO function"""

    def setup_method(self):
        """Set up test data"""
        self.fuelflow_evaluate = np.array([0.1, 0.5, 1.0, 2.0])
        self.x_EI_matrix = ThrustModeValues(100.0, 50.0, 10.0, 8.0)
        self.fuelflow_calibrate = ThrustModeValues(0.2, 0.6, 1.5, 2.0)
        # Standard atmosphere conditions.
        self.Tamb = 288.15
        self.Pamb = 101325.0

    def test_basic_functionality(self):
        """Test basic HC+CO emissions calculation"""
        result = EI_HCCO(
            self.fuelflow_evaluate,
            self.x_EI_matrix,
            self.fuelflow_calibrate,
            self.Tamb,
            self.Pamb,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == self.fuelflow_evaluate.shape
        assert len(result) == len(self.fuelflow_evaluate)

    def test_non_negativity(self):
        """Test that outputs are non-negative"""
        result = EI_HCCO(
            self.fuelflow_evaluate,
            self.x_EI_matrix,
            self.fuelflow_calibrate,
            self.Tamb,
            self.Pamb,
        )
        assert np.all(result >= 0)

    def test_finiteness(self):
        """Test that outputs are finite"""
        result = EI_HCCO(
            self.fuelflow_evaluate,
            self.x_EI_matrix,
            self.fuelflow_calibrate,
            self.Tamb,
            self.Pamb,
        )
        assert np.all(np.isfinite(result))

    def test_zero_and_negative_fuel_flows(self):
        """Test how zero and negative fuel flows are handled"""
        test_flow = np.array([0.0, -0.01, 0.1, 1.0])
        result = EI_HCCO(
            test_flow, self.x_EI_matrix, self.fuelflow_calibrate, self.Tamb, self.Pamb
        )
        assert result.shape == test_flow.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_duplicate_calibration_flows_flatten_slanted_segment(self):
        """Duplicate calibration flows should force a flat lower segment"""
        fuelflow_eval = np.array([0.15, 0.25, 0.3, 0.5])
        x_EI_matrix = ThrustModeValues(10.0, 10.0, 5.0, 3.0)
        fuelflow_calibrate = ThrustModeValues(0.3, 0.3, 0.7, 1.4)

        result = EI_HCCO(
            fuelflow_eval, x_EI_matrix, fuelflow_calibrate, self.Tamb, self.Pamb
        )
        expected_upper = np.sqrt(
            x_EI_matrix[ThrustMode.CLIMB] * x_EI_matrix[ThrustMode.TAKEOFF]
        )
        expected = np.full_like(fuelflow_eval, expected_upper)

        low_thrust_mask = fuelflow_eval < fuelflow_calibrate[ThrustMode.IDLE]
        expected[low_thrust_mask] *= 1 - 52.0 * (
            fuelflow_eval[low_thrust_mask] - fuelflow_calibrate[ThrustMode.IDLE]
        )

        assert np.allclose(result, expected)

    def test_intercept_adjustment_uses_second_mode_value(self):
        """When intercept drifts low, the second mode should set the ceiling"""
        fuelflow_eval = np.array([0.2, 0.5, 0.9])
        x_EI_matrix = ThrustModeValues(
            38.33753758, 2.4406048, 106.49710981, 13.57427593
        )
        fuelflow_calibrate = ThrustModeValues(
            0.10569869, 0.40041291, 0.81271722, 0.86727924
        )

        result = EI_HCCO(
            fuelflow_eval, x_EI_matrix, fuelflow_calibrate, self.Tamb, self.Pamb
        )
        high_mask = fuelflow_eval >= fuelflow_calibrate[ThrustMode.APPROACH]

        assert np.allclose(result[high_mask], x_EI_matrix[ThrustMode.APPROACH])
        assert np.all(result >= 0.0)

    def test_positive_slope_forces_horizontal_segment(self):
        """Non-negative slopes should collapse to the upper horizontal level"""
        fuelflow_eval = np.array([0.2, 0.35, 0.5])
        x_EI_matrix = ThrustModeValues(10.0, 11.0, 2.0, 1.0)
        fuelflow_calibrate = ThrustModeValues(0.2, 0.3, 0.4, 0.5)

        result = EI_HCCO(
            fuelflow_eval, x_EI_matrix, fuelflow_calibrate, self.Tamb, self.Pamb
        )
        expected_value = np.sqrt(
            x_EI_matrix[ThrustMode.CLIMB] * x_EI_matrix[ThrustMode.TAKEOFF]
        )

        assert np.allclose(result, expected_value)


class TestBFFM2_EINOx:
    """Tests for BFFM2_EINOx function"""

    def setup_method(self):
        """Set up test data"""
        self.fuelflow_trajectory = np.array([0.5, 1.0, 1.5, 2.0])
        self.EI_NOx_matrix = ThrustModeValues(30.0, 25.0, 20.0, 18.0)
        self.fuelflow_performance = ThrustModeValues(0.4, 0.8, 1.2, 1.8)
        self.Tamb = np.array([288.15, 250.0, 220.0, 280.0])
        self.Pamb = np.array([101325.0, 25000.0, 15000.0, 95000.0])

    def _components(self, result: BFFM2EINOxResult):
        return (
            result.NOxEI,
            result.NOEI,
            result.NO2EI,
            result.HONOEI,
            result.noProp,
            result.no2Prop,
            result.honoProp,
        )

    def test_basic_functionality(self):
        """Test basic NOₓ emissions calculation"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        assert isinstance(result, BFFM2EINOxResult)

        # Check shapes
        expected_shape = self.fuelflow_trajectory.shape
        for array in self._components(result):
            assert array.shape == expected_shape

    def test_non_negativity(self):
        """Test that all outputs are non-negative"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        for array in self._components(result):
            assert np.all(array >= 0)

    def test_summation_consistency(self):
        """Test that NO + NO₂ + HONO proportions sum to 1"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        NOxEI = result.NOxEI
        NOEI = result.NOEI
        NO2EI = result.NO2EI
        HONOEI = result.HONOEI
        noProp = result.noProp
        no2Prop = result.no2Prop
        honoProp = result.honoProp

        # Proportions should sum to 1
        total_prop = noProp + no2Prop + honoProp
        assert np.allclose(total_prop, 1.0, rtol=1e-10)

        # Component EIs should sum to total NOₓ EI
        total_component_EI = NOEI + NO2EI + HONOEI
        assert np.allclose(total_component_EI, NOxEI, rtol=1e-10)

    def test_finiteness(self):
        """Test that all outputs are finite"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        for array in self._components(result):
            assert np.all(np.isfinite(array))

    @patch('AEIC.emissions.utils.get_thrust_cat_cruise')
    def test_thrust_categorization(self, mock_get_thrust_cat_cruise):
        """Test thrust categorization functionality"""
        # Mock thrust categories
        mock_get_thrust_cat_cruise.return_value = np.array(
            [1, 2, 3, 1]
        )  # High, Low, Approach, High

        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        # Should still return valid results
        for array in self._components(result):
            assert np.all(np.isfinite(array))

    def test_matches_reference_component_values(self):
        """Reference regression to guard against inadvertent logic changes"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )
        expected_arrays = [
            np.array([28.26762038, 15.01898492, 12.50815229, 18.59189322]),
            np.array([3.64440296, 12.0482297, 11.48326556, 17.06851997]),
            np.array([23.3511745, 2.2949009, 0.93107559, 1.38393405]),
            np.array([1.27204292, 0.67585432, 0.09381114, 0.1394392]),
            np.array([0.128925, 0.8022, 0.9180625, 0.9180625]),
            np.array([0.826075, 0.1528, 0.0744375, 0.0744375]),
            np.array([0.045, 0.045, 0.0075, 0.0075]),
        ]
        for array, expected in zip(self._components(result), expected_arrays):
            np.testing.assert_allclose(array, expected, rtol=1e-6, atol=1e-9)


class TestNOxSpeciation:
    """Tests for NOₓ_speciation function"""

    def test_summation_consistency(self):
        """Test that proportions sum to 1 for each thrust category"""
        speciation = NOx_speciation()

        total_prop = speciation.no + speciation.no2 + speciation.hono
        assert np.isclose(total_prop[ThrustMode.IDLE], 1.0, rtol=1.0e-10)
        assert np.isclose(total_prop[ThrustMode.APPROACH], 1.0, rtol=1.0e-10)
        assert np.isclose(total_prop[ThrustMode.CLIMB], 1.0, rtol=1.0e-10)
        assert np.isclose(total_prop[ThrustMode.TAKEOFF], 1.0, rtol=1.0e-10)

    def test_non_negativity(self):
        """Test that all proportions are non-negative"""
        speciation = NOx_speciation()

        for species in [speciation.no, speciation.no2, speciation.hono]:
            assert species[ThrustMode.IDLE] >= 0.0
            assert species[ThrustMode.APPROACH] >= 0.0
            assert species[ThrustMode.CLIMB] >= 0.0
            assert species[ThrustMode.TAKEOFF] >= 0.0


class TestEI_SOx:
    """Tests for EI_SOx function"""

    def test_basic_functionality(self, fuel_jetA):
        """Test basic SOx emissions calculation"""
        result = EI_SOx(fuel_jetA)

        assert isinstance(result, SOxEmissionResult)
        assert isinstance(result.EI_SO2, int | float)
        assert isinstance(result.EI_SO4, int | float)

    def test_non_negativity(self, fuel_jetA):
        """Test that outputs are non-negative"""
        result = EI_SOx(fuel_jetA)

        assert result.EI_SO2 >= 0
        assert result.EI_SO4 >= 0

    def test_mass_balance(self, fuel_jetA):
        """Test that SO2 + SO4 production makes sense relative to sulfur content"""

        result = EI_SOx(fuel_jetA)

        # Calculate total sulfur converted (should be proportional to FSC)
        MW_SO2, MW_SO4, MW_S = 64.0, 96.0, 32.0

        # Back-calculate sulfur content from emissions
        sulfur_as_SO2 = result.EI_SO2 * MW_S / MW_SO2
        sulfur_as_SO4 = result.EI_SO4 * MW_S / MW_SO4
        total_sulfur_converted = sulfur_as_SO2 + sulfur_as_SO4

        # Should be approximately equal to input sulfur (with unit conversions)
        expected_sulfur = (
            fuel_jetA.fuel_sulfur_content_nom / 1000
        )  # Convert ppm to g/kg
        assert np.isclose(total_sulfur_converted, expected_sulfur, rtol=0.01)

    def test_finiteness(self, fuel_jetA):
        """Test that outputs are finite"""

        result = EI_SOx(fuel_jetA)

        assert np.isfinite(result.EI_SO2)
        assert np.isfinite(result.EI_SO4)


class TestGetAPUEmissions:
    """Tests for get_APU_emissions function"""

    def setup_method(self):
        """Set up test data"""

        self.LTO_emission_indices = SpeciesValues[ThrustModeValues](
            {
                Species.SO2: ThrustModeValues(1.2),
                Species.SO4: ThrustModeValues(0.8),
            }
        )

        self.apu = APU(
            name='Test APU',
            defra='00000',
            fuel_kg_per_s=0.1,
            PM10_g_per_kg=0.5,
            NOx_g_per_kg=15.0,
            HC_g_per_kg=2.0,
            CO_g_per_kg=25.0,
        )

    def test_basic_functionality(self, fuel_jetA):
        """Test basic APU emissions calculation"""
        apu = get_APU_emissions(self.LTO_emission_indices, self.apu, fuel_jetA)

        assert apu.indices[Species.SO2] > 0
        assert apu.indices[Species.NOx] > 0
        assert apu.emissions[Species.SO2] > 0

    def test_non_negativity(self, fuel_jetA):
        """Test that all emissions are non-negative"""
        apu = get_APU_emissions(self.LTO_emission_indices, self.apu, fuel_jetA)

        for field in apu.indices.keys():
            assert apu.indices[field] >= 0, f"{field} emission index is negative"
            assert apu.emissions[field] >= 0, f"{field} total emission is negative"

    def test_consistency_between_ei_and_total(self, fuel_jetA):
        """Test consistency between emission indices and total emissions"""
        apu_time = 2854
        apu = get_APU_emissions(
            self.LTO_emission_indices,
            self.apu,
            fuel_jetA,
            apu_time=apu_time,
        )

        fuel_burn = self.apu.fuel_kg_per_s * apu_time
        for field in apu.indices.keys():
            expected = apu.indices[field] * fuel_burn
            assert np.isclose(apu.emissions[field], expected, rtol=1e-10), (
                f"Mismatch in {field}"
            )

    def test_nox_speciation_consistency(self, fuel_jetA):
        """Test NOₓ speciation consistency via PM10_g_per_kg scaling"""
        apu = get_APU_emissions(self.LTO_emission_indices, self.apu, fuel_jetA)
        nox_speciation = NOx_speciation()

        assert (
            apu.indices[Species.NO]
            == self.apu.PM10_g_per_kg * nox_speciation.no[ThrustMode.IDLE]
        )
        assert (
            apu.indices[Species.NO2]
            == self.apu.PM10_g_per_kg * nox_speciation.no2[ThrustMode.IDLE]
        )
        assert (
            apu.indices[Species.HONO]
            == self.apu.PM10_g_per_kg * nox_speciation.hono[ThrustMode.IDLE]
        )

    def test_zero_fuel_flow_handling(self, fuel_jetA):
        """Test handling of zero fuel flow"""
        apu_data_zero = self.apu.model_copy(update={'fuel_kg_per_s': 0.0})

        apu = get_APU_emissions(self.LTO_emission_indices, apu_data_zero, fuel_jetA)

        assert apu.indices[Species.SO2] == 0.0
        assert apu.indices[Species.SO4] == 0.0
        assert apu.indices[Species.CO2] == 0.0
        assert apu.emissions[Species.CO2] == 0.0

    @pytest.mark.config_updates(emissions__pmnvol_method='scope11')
    def test_nvpm_method_enables_number_channel(self, fuel_jetA):
        """PM number index should be emitted when nvpm_method requests it"""
        apu = get_APU_emissions(self.LTO_emission_indices, self.apu, fuel_jetA)

        assert Species.PMnvolN in apu.indices
        assert apu.indices[Species.PMnvolN] == 0.0


def make_edb_lto_values(x0: float, x1: float, x2: float, x3: float) -> ThrustModeValues:
    """Helper to create ModeDict from array for tests"""
    return ThrustModeValues(
        {
            ThrustMode.IDLE: x0,
            ThrustMode.APPROACH: x1,
            ThrustMode.CLIMB: x2,
            ThrustMode.TAKEOFF: x3,
        }
    )


class TestPMnvolMEEM:
    """Tests for the PMnvol_MEEM cruise methodology"""

    def test_reconstructs_missing_mode_data_and_interpolates(self):
        """Negative mode inputs should be rebuilt and yield finite cruise profiles"""
        EDB_data = EDBEntry(
            engine='Test',
            uid='TEST000',
            engine_type='MTF',
            BP_Ratio=5.0,
            rated_thrust=100.0,
            fuel_flow=make_edb_lto_values(0, 0, 0, 0),
            CO_EI_matrix=make_edb_lto_values(0, 0, 0, 0),
            HC_EI_matrix=make_edb_lto_values(0, 0, 0, 0),
            EI_NOx_matrix=make_edb_lto_values(0, 0, 0, 0),
            SN_matrix=make_edb_lto_values(10, 20, 25, 30),
            nvPM_mass_matrix=make_edb_lto_values(-1, -1, -1, -1),
            nvPM_num_matrix=make_edb_lto_values(-1, -1, -1, -1),
            PR=make_edb_lto_values(25, 25, 25, 25),
            EImass_max=50.0,
            EImass_max_thrust=0.575,
            EInum_max=4.5e15,
            EInum_max_thrust=0.925,
        )
        altitudes = np.array([0.0, 6000.0, 12000.0])
        Tamb = np.array([288.15, 250.0, 220.0])
        Pamb = np.array([101325.0, 54000.0, 26500.0])
        mach = np.array([0.0, 0.7, 0.8])

        gmd, mass, num = PMnvol_MEEM(EDB_data, altitudes, Tamb, Pamb, mach)

        assert gmd.shape == altitudes.shape
        assert mass.shape == altitudes.shape
        assert num.shape == altitudes.shape
        assert np.all(gmd > 0.0)
        assert np.all(mass > 0.0)
        assert np.all(num > 0.0)
        assert np.all(np.isfinite(mass))

    def test_invalid_smoke_numbers_zero_results(self):
        """All-negative smoke numbers should zero out the trajectory"""
        EDB_data = EDBEntry(
            engine='Test',
            uid='TEST000',
            engine_type='TF',
            BP_Ratio=0.0,
            rated_thrust=100.0,
            fuel_flow=make_edb_lto_values(0, 0, 0, 0),
            CO_EI_matrix=make_edb_lto_values(0, 0, 0, 0),
            HC_EI_matrix=make_edb_lto_values(0, 0, 0, 0),
            EI_NOx_matrix=make_edb_lto_values(0, 0, 0, 0),
            SN_matrix=make_edb_lto_values(-5, -5, -5, -5),
            nvPM_mass_matrix=make_edb_lto_values(1, 2, 3, 4),
            nvPM_num_matrix=make_edb_lto_values(1, 2, 3, 4),
            PR=make_edb_lto_values(20, 20, 20, 20),
            EImass_max=10.0,
            EImass_max_thrust=float('nan'),
            EInum_max=1.0e12,
            EInum_max_thrust=float('nan'),
        )
        altitudes = np.array([3000.0, 3500.0])
        Tamb = np.array([260.0, 250.0])
        Pamb = np.array([70000.0, 65000.0])
        mach = np.array([0.3, 0.4])

        gmd, mass, num = PMnvol_MEEM(EDB_data, altitudes, Tamb, Pamb, mach)

        assert np.all(gmd == 0.0)
        assert np.all(mass == 0.0)
        assert np.all(num == 0.0)


class TestCalculatePMnvolScope11:
    """Tests for calculate_PMnvolEI_scope11"""

    def test_engine_type_scaling_and_invalid_smoke_numbers(self):
        SN_matrix = ThrustModeValues(5.0, 50.0, -1.0, 0.0)
        BP_Ratio = 2.0

        mtf = calculate_PMnvolEI_scope11(SN_matrix, 'MTF', BP_Ratio)
        tf = calculate_PMnvolEI_scope11(SN_matrix, 'TF', BP_Ratio)

        SN0 = min(SN_matrix[ThrustMode.IDLE], 40.0)
        CBC0 = 0.6484 * np.exp(0.0766 * SN0) / (1 + np.exp(-1.098 * (SN0 - 3.064)))
        AFR = ThrustModeValues(106, 83, 51, 45)

        bypass = 1 + BP_Ratio
        kslm_mtf = np.log(
            (3.219 * CBC0 * bypass * 1000 + 312.5) / (CBC0 * bypass * 1000 + 42.6)
        )
        Q_mtf = 0.776 * AFR[ThrustMode.IDLE] * bypass + 0.767
        expected_mtf = (kslm_mtf * CBC0 * Q_mtf) / 1000.0
        assert np.isclose(mtf[ThrustMode.IDLE], expected_mtf)

        kslm_tf = np.log((3.219 * CBC0 * 1000 + 312.5) / (CBC0 * 1000 + 42.6))
        Q_tf = 0.776 * AFR[ThrustMode.IDLE] + 0.767
        expected_tf = (kslm_tf * CBC0 * Q_tf) / 1000.0
        assert np.isclose(tf[ThrustMode.IDLE], expected_tf)

        assert mtf[ThrustMode.IDLE] > tf[ThrustMode.IDLE]
        assert mtf[ThrustMode.CLIMB] == 0.0
        assert tf[ThrustMode.TAKEOFF] == 0.0


class TestEI_PMvol:
    """Tests for EI_PMvol helper functions"""

    def test_fuel_flow_path_uses_lube_contributions(self):
        fuelflow = np.ones((2, 4))
        thrustModes = ThrustModeArray(np.array([ThrustMode.IDLE, ThrustMode.APPROACH]))

        pmvol, ocic = EI_PMvol_FuelFlow(fuelflow, thrustModes)

        assert pmvol.shape == thrustModes.shape
        assert ocic.shape == fuelflow.shape
        assert np.isclose(pmvol[0], 0.02 / (1 - 0.15))
        assert np.isclose(pmvol[1], 0.02 / (1 - 0.50))
        assert np.allclose(ocic, 0.02)

    def test_foa3_interpolation_matches_reference_curve(self):
        thrusts = np.array([[7.0, 30.0, 85.0, 100.0], [50.0, 70.0, 90.0, 100.0]])
        HCEI = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.75, 1.0, 1.5]])

        pmvol, ocic = EI_PMvol_FOA3(thrusts, HCEI)

        ICAO_thrust = np.array([7.0, 30.0, 85.0, 100.0])
        delta = np.array([6.17, 56.25, 76.0, 115.0])
        expected_delta = np.interp(thrusts, ICAO_thrust, delta)
        expected_pmvol = expected_delta * HCEI / 1000.0

        assert np.allclose(pmvol, expected_pmvol)
        assert np.allclose(ocic, expected_pmvol)


# Integration tests
class TestIntegration:
    """Integration tests to check function interactions"""

    def test_nox_emissions_consistency(self):
        """Test NOₓ emissions consistency across functions"""
        fuelflow_trajectory = np.array([1.0, 1.5, 2.0])
        EI_NOx_matrix = ThrustModeValues(30.0, 25.0, 20.0, 18.0)
        fuelflow_performance = ThrustModeValues(0.8, 1.2, 1.6, 2.0)
        Tamb = np.array([288.15, 250.0, 220.0])
        Pamb = np.array([101325.0, 25000.0, 15000.0])

        result = BFFM2_EINOx(
            fuelflow_trajectory, EI_NOx_matrix, fuelflow_performance, Tamb, Pamb
        )

        assert np.allclose(
            result.NOxEI, np.array([27.11460822, 14.28251747, 11.92937893])
        )
