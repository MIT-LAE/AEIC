import tomllib
from unittest.mock import patch

import numpy as np
import pytest

from AEIC.emissions.APU_emissions import get_APU_emissions
from AEIC.emissions.EI_CO2 import EI_CO2, CO2EmissionResult
from AEIC.emissions.EI_H2O import EI_H2O
from AEIC.emissions.EI_HCCO import EI_HCCO
from AEIC.emissions.EI_NOx import BFFM2_EINOx, BFFM2EINOxResult, NOx_speciation
from AEIC.emissions.EI_PMnvol import PMnvol_MEEM, calculate_PMnvolEI_scope11
from AEIC.emissions.EI_PMvol import EI_PMvol_FOA3, EI_PMvol_FuelFlow
from AEIC.emissions.EI_SOx import EI_SOx, SOxEmissionResult
from AEIC.emissions.lifecycle_CO2 import lifecycle_CO2
from AEIC.utils.files import file_location


class TestEI_CO2:
    """Tests for EI_CO2 function"""

    def test_returns_documented_jet_a_values(self):
        """Jet-A reference EI and carbon fraction should match documentation"""
        with open(file_location("fuels/conventional_jetA.toml"), 'rb') as f:
            fuel = tomllib.load(f)
        result = EI_CO2(fuel)

        assert isinstance(result, CO2EmissionResult)
        assert result.EI_CO2 == pytest.approx(3155.6)
        assert result.nvolCarbCont == pytest.approx(0.95)

    def test_distinguishes_saf_inputs(self):
        """Different fuels propagate their specific EI metadata."""
        with open(file_location("fuels/SAF.toml"), 'rb') as f:
            fuel = tomllib.load(f)
        result = EI_CO2(fuel)

        assert result.EI_CO2 == pytest.approx(fuel['EI_CO2'])
        assert result.nvolCarbCont == pytest.approx(fuel['nvolCarbCont'])

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(KeyError):
            EI_CO2({})

        fuel = {'EI_CO2': -100, 'nvolCarbCont': -0.5}
        result = EI_CO2(fuel)
        assert result.EI_CO2 == -100
        assert result.nvolCarbCont == -0.5


class TestEI_H2O:
    """Tests for EI_H2O function"""

    def test_returns_documented_jet_a_values(self):
        """Jet-A water EI should match the nominal property sheet."""
        with open(file_location("fuels/conventional_jetA.toml"), 'rb') as f:
            fuel = tomllib.load(f)
        assert EI_H2O(fuel) == pytest.approx(1233.3865)

    def test_distinguishes_saf_inputs(self):
        """SAF water EI is different and should be propagated verbatim."""
        with open(file_location("fuels/SAF.toml"), 'rb') as f:
            fuel = tomllib.load(f)
        assert EI_H2O(fuel) == pytest.approx(1356.72515)

    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(KeyError):
            EI_H2O({})


class TestEI_HCCO:
    """Tests for EI_HCCO function"""

    def setup_method(self):
        """Set up test data"""
        self.fuelflow_evaluate = np.array([0.1, 0.5, 1.0, 2.0])
        self.x_EI_matrix = np.array([100.0, 50.0, 10.0, 8.0])
        self.fuelflow_calibrate = np.array([0.2, 0.6, 1.5, 2.0])

    def test_basic_functionality(self):
        """Test basic HC+CO emissions calculation"""
        result = EI_HCCO(
            self.fuelflow_evaluate, self.x_EI_matrix, self.fuelflow_calibrate
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == self.fuelflow_evaluate.shape
        assert len(result) == len(self.fuelflow_evaluate)

    def test_non_negativity(self):
        """Test that outputs are non-negative"""
        result = EI_HCCO(
            self.fuelflow_evaluate, self.x_EI_matrix, self.fuelflow_calibrate
        )
        assert np.all(result >= 0)

    def test_finiteness(self):
        """Test that outputs are finite"""
        result = EI_HCCO(
            self.fuelflow_evaluate, self.x_EI_matrix, self.fuelflow_calibrate
        )
        assert np.all(np.isfinite(result))

    def test_shape_consistency(self):
        """Test shape consistency for fuel flow input"""
        fuelflow_2d = np.array([[0.1, 0.5], [1.0, 2.0]])
        with pytest.raises(ValueError, match="fuelflow_evaluate must be a 1D array"):
            EI_HCCO(fuelflow_2d, self.x_EI_matrix, self.fuelflow_calibrate)

    def test_input_validation(self):
        """Test input validation for EI and calibration vectors"""
        with pytest.raises(
            ValueError, match="x_EI_matrix must be a 1D array of length 4"
        ):
            EI_HCCO(
                self.fuelflow_evaluate, np.array([1, 2, 3]), self.fuelflow_calibrate
            )

        with pytest.raises(
            ValueError, match="fuelflow_calibrate must be a 1D array of length 4"
        ):
            EI_HCCO(self.fuelflow_evaluate, self.x_EI_matrix, np.array([0.1, 0.2, 0.3]))

    def test_cruise_correction(self):
        """Test cruise correction modifies output as expected"""
        result_no_cruise = EI_HCCO(
            self.fuelflow_evaluate,
            self.x_EI_matrix,
            self.fuelflow_calibrate,
            cruiseCalc=False,
        )
        result_with_cruise = EI_HCCO(
            self.fuelflow_evaluate,
            self.x_EI_matrix,
            self.fuelflow_calibrate,
            cruiseCalc=True,
            Tamb=250.0,
            Pamb=25000.0,
        )

        assert not np.allclose(result_no_cruise, result_with_cruise)
        assert np.all(np.isfinite(result_with_cruise))

    def test_zero_and_negative_fuel_flows(self):
        """Test how zero and negative fuel flows are handled"""
        test_flow = np.array([0.0, -0.01, 0.1, 1.0])
        result = EI_HCCO(test_flow, self.x_EI_matrix, self.fuelflow_calibrate)
        assert result.shape == test_flow.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_duplicate_calibration_flows_flatten_slanted_segment(self):
        """Duplicate calibration flows should force a flat lower segment"""
        fuelflow_eval = np.array([0.15, 0.25, 0.3, 0.5])
        x_EI_matrix = np.array([10.0, 10.0, 5.0, 3.0])
        fuelflow_calibrate = np.array([0.3, 0.3, 0.7, 1.4])

        result = EI_HCCO(fuelflow_eval, x_EI_matrix, fuelflow_calibrate)
        expected_upper = np.sqrt(x_EI_matrix[2] * x_EI_matrix[3])
        expected = np.full_like(fuelflow_eval, expected_upper)

        low_thrust_mask = fuelflow_eval < fuelflow_calibrate[0]
        expected[low_thrust_mask] *= 1 - 52.0 * (
            fuelflow_eval[low_thrust_mask] - fuelflow_calibrate[0]
        )

        assert np.allclose(result, expected)

    def test_intercept_adjustment_uses_second_mode_value(self):
        """When intercept drifts low, the second mode should set the ceiling"""
        fuelflow_eval = np.array([0.2, 0.5, 0.9])
        x_EI_matrix = np.array([38.33753758, 2.4406048, 106.49710981, 13.57427593])
        fuelflow_calibrate = np.array([0.10569869, 0.40041291, 0.81271722, 0.86727924])

        result = EI_HCCO(fuelflow_eval, x_EI_matrix, fuelflow_calibrate)
        high_mask = fuelflow_eval >= fuelflow_calibrate[1]

        assert np.allclose(result[high_mask], x_EI_matrix[1])
        assert np.all(result >= 0.0)

    def test_positive_slope_forces_horizontal_segment(self):
        """Non-negative slopes should collapse to the upper horizontal level"""
        fuelflow_eval = np.array([0.2, 0.35, 0.5])
        x_EI_matrix = np.array([10.0, 11.0, 2.0, 1.0])
        fuelflow_calibrate = np.array([0.2, 0.3, 0.4, 0.5])

        result = EI_HCCO(fuelflow_eval, x_EI_matrix, fuelflow_calibrate)
        expected_value = np.sqrt(x_EI_matrix[2] * x_EI_matrix[3])

        assert np.allclose(result, expected_value)


class TestBFFM2_EINOx:
    """Tests for BFFM2_EINOx function"""

    def setup_method(self):
        """Set up test data"""
        self.fuelflow_trajectory = np.array([0.5, 1.0, 1.5, 2.0])
        self.NOX_EI_matrix = np.array([30.0, 25.0, 20.0, 18.0])
        self.fuelflow_performance = np.array([0.4, 0.8, 1.2, 1.8])
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
        """Test basic NOx emissions calculation"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
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
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        for array in self._components(result):
            assert np.all(array >= 0)

    def test_summation_consistency(self):
        """Test that NO + NO2 + HONO proportions sum to 1"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
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

        # Component EIs should sum to total NOx EI
        total_component_EI = NOEI + NO2EI + HONOEI
        assert np.allclose(total_component_EI, NOxEI, rtol=1e-10)

    def test_finiteness(self):
        """Test that all outputs are finite"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        for array in self._components(result):
            assert np.all(np.isfinite(array))

    def test_cruise_correction_effect(self):
        """Test that cruise correction has an effect"""
        results_no_cruise = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
            cruiseCalc=False,
        )

        results_with_cruise = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
            cruiseCalc=True,
        )

        # NOx EI should be different
        assert not np.allclose(
            results_no_cruise.NOxEI,
            results_with_cruise.NOxEI,
        )

    @patch('AEIC.utils.standard_fuel.get_thrust_cat')
    def test_thrust_categorization(self, mock_get_thrust_cat):
        """Test thrust categorization functionality"""
        # Mock thrust categories
        mock_get_thrust_cat.return_value = np.array(
            [1, 2, 3, 1]
        )  # High, Low, Approach, High

        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
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
            self.NOX_EI_matrix,
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
    """Tests for NOx_speciation function"""

    def test_basic_functionality(self):
        """Test basic NOx speciation"""
        thrustCat = np.array([1, 2, 3, 1, 2])  # High, Low, Approach, High, Low
        noProp, no2Prop, honoProp = NOx_speciation(thrustCat)

        assert len(noProp) == len(thrustCat)
        assert len(no2Prop) == len(thrustCat)
        assert len(honoProp) == len(thrustCat)

    def test_summation_consistency(self):
        """Test that proportions sum to 1 for each thrust category"""
        thrustCat = np.array([1, 2, 3])
        noProp, no2Prop, honoProp = NOx_speciation(thrustCat)

        total_prop = noProp + no2Prop + honoProp
        assert np.allclose(total_prop, 1.0, rtol=1e-10)

    def test_non_negativity(self):
        """Test that all proportions are non-negative"""
        thrustCat = np.array([1, 2, 3, 1, 2, 3])
        noProp, no2Prop, honoProp = NOx_speciation(thrustCat)

        assert np.all(noProp >= 0)
        assert np.all(no2Prop >= 0)
        assert np.all(honoProp >= 0)

    def test_thrust_category_consistency(self):
        """Test that same thrust categories give same results"""
        thrustCat = np.array([1, 1, 2, 2, 3, 3])
        noProp, no2Prop, honoProp = NOx_speciation(thrustCat)

        # Same categories should have same proportions
        assert noProp[0] == noProp[1]
        assert noProp[2] == noProp[3]
        assert noProp[4] == noProp[5]


class TestEI_SOx:
    """Tests for EI_SOx function"""

    def test_basic_functionality(self):
        """Test basic SOx emissions calculation"""
        fuel = {
            'FSCnom': 600.0,  # ppm
            'Epsnom': 0.02,  # fraction
        }

        result = EI_SOx(fuel)

        assert isinstance(result, SOxEmissionResult)
        assert isinstance(result.EI_SO2, int | float)
        assert isinstance(result.EI_SO4, int | float)

    def test_non_negativity(self):
        """Test that outputs are non-negative"""
        fuel = {'FSCnom': 600.0, 'Epsnom': 0.02}

        result = EI_SOx(fuel)

        assert result.EI_SO2 >= 0
        assert result.EI_SO4 >= 0

    def test_mass_balance(self):
        """Test that SO2 + SO4 production makes sense relative to sulfur content"""
        fuel = {
            'FSCnom': 600.0,  # 600 ppm sulfur
            'Epsnom': 0.02,  # 2% converted to SO4
        }

        result = EI_SOx(fuel)

        # Calculate total sulfur converted (should be proportional to FSC)
        MW_SO2, MW_SO4, MW_S = 64.0, 96.0, 32.0

        # Back-calculate sulfur content from emissions
        sulfur_as_SO2 = result.EI_SO2 * MW_S / MW_SO2
        sulfur_as_SO4 = result.EI_SO4 * MW_S / MW_SO4
        total_sulfur_converted = sulfur_as_SO2 + sulfur_as_SO4

        # Should be approximately equal to input sulfur (with unit conversions)
        expected_sulfur = fuel['FSCnom'] / 1000  # Convert ppm to g/kg
        assert np.isclose(total_sulfur_converted, expected_sulfur, rtol=0.01)

    def test_finiteness(self):
        """Test that outputs are finite"""
        fuel = {'FSCnom': 600.0, 'Epsnom': 0.02}

        result = EI_SOx(fuel)

        assert np.isfinite(result.EI_SO2)
        assert np.isfinite(result.EI_SO4)

    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(KeyError):
            EI_SOx({})

        # Test with zero values
        fuel = {'FSCnom': 0.0, 'Epsnom': 0.0}
        result = EI_SOx(fuel)
        assert result.EI_SO2 == 0.0
        assert result.EI_SO4 == 0.0


class TestGetAPUEmissions:
    """Tests for get_APU_emissions function"""

    def setup_method(self):
        """Set up test data"""
        dtype = [
            ('SO2', 'f8'),
            ('SO4', 'f8'),
            ('H2O', 'f8'),
            ('PMnvol', 'f8'),
            ('PMvol', 'f8'),
            ('PMnvolGMD', 'f8'),
            ('PMnvolN', 'f8'),
            ('OCic', 'f8'),
            ('NO', 'f8'),
            ('NO2', 'f8'),
            ('HONO', 'f8'),
            ('NOx', 'f8'),
            ('HC', 'f8'),
            ('CO', 'f8'),
            ('CO2', 'f8'),
        ]
        self.APU_emission_indices = np.zeros(1, dtype=dtype)[0]
        self.APU_emissions_g = np.zeros(1, dtype=dtype)[0]

        self.LTO_emission_indices = {'SO2': np.array([1.2]), 'SO4': np.array([0.8])}

        self.APU_data = {
            'fuel_kg_per_s': 0.1,
            'PM10_g_per_kg': 0.5,
            'NOx_g_per_kg': 15.0,
            'HC_g_per_kg': 2.0,
            'CO_g_per_kg': 25.0,
        }

        self.LTO_noProp = np.array([0.85])
        self.LTO_no2Prop = np.array([0.10])
        self.LTO_honoProp = np.array([0.05])

    def test_basic_functionality(self):
        """Test basic APU emissions calculation"""
        apu_ei, apu_g, _ = get_APU_emissions(
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.LTO_emission_indices,
            self.APU_data,
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
            EI_H2O=1233.3865,
        )

        assert apu_ei['SO2'] > 0
        assert apu_ei['NOx'] > 0
        assert apu_g['SO2'] > 0

    def test_non_negativity(self):
        """Test that all emissions are non-negative"""
        apu_ei, apu_g, _ = get_APU_emissions(
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.LTO_emission_indices,
            self.APU_data,
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
            EI_H2O=1233.3865,
        )

        for field in apu_ei.dtype.names:
            assert apu_ei[field] >= 0, f"{field} emission index is negative"
            assert apu_g[field] >= 0, f"{field} total emission is negative"

    def test_consistency_between_ei_and_total(self):
        """Test consistency between emission indices and total emissions"""
        apu_tim = 2854
        apu_ei, apu_g, _ = get_APU_emissions(
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.LTO_emission_indices,
            self.APU_data,
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
            EI_H2O=1233.3865,
            apu_tim=apu_tim,
        )

        fuel_burn = self.APU_data['fuel_kg_per_s'] * apu_tim
        for field in apu_ei.dtype.names:
            expected = apu_ei[field] * fuel_burn
            assert np.isclose(apu_g[field], expected, rtol=1e-10), (
                f"Mismatch in {field}"
            )

    def test_nox_speciation_consistency(self):
        """Test NOx speciation consistency via PM10_g_per_kg scaling"""
        apu_ei, _, _ = get_APU_emissions(
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.LTO_emission_indices,
            self.APU_data,
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
            EI_H2O=1233.3865,
        )

        assert apu_ei['NO'] == self.APU_data['PM10_g_per_kg'] * self.LTO_noProp[0]
        assert apu_ei['NO2'] == self.APU_data['PM10_g_per_kg'] * self.LTO_no2Prop[0]
        assert apu_ei['HONO'] == self.APU_data['PM10_g_per_kg'] * self.LTO_honoProp[0]

    def test_zero_fuel_flow_handling(self):
        """Test handling of zero fuel flow"""
        apu_data_zero = self.APU_data.copy()
        apu_data_zero['fuel_kg_per_s'] = 0.0

        apu_ei, apu_g, _ = get_APU_emissions(
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.LTO_emission_indices,
            apu_data_zero,
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
            EI_H2O=1233.3865,
        )

        assert apu_ei['SO2'] == 0.0
        assert apu_ei['SO4'] == 0.0
        assert apu_ei['CO2'] == 0.0
        assert apu_g['CO2'] == 0.0

    def test_nvpm_method_enables_number_channel(self):
        """PM number index should be emitted when nvpm_method requests it"""
        apu_ei, _, _ = get_APU_emissions(
            self.APU_emission_indices,
            self.APU_emissions_g,
            self.LTO_emission_indices,
            self.APU_data,
            self.LTO_noProp,
            self.LTO_no2Prop,
            self.LTO_honoProp,
            EI_H2O=1233.3865,
            nvpm_method='scope11',
        )

        assert 'PMnvolN' in apu_ei.dtype.names
        assert apu_ei['PMnvolN'] == 0.0


class TestPMnvolMEEM:
    """Tests for the PMnvol_MEEM cruise methodology"""

    def test_reconstructs_missing_mode_data_and_interpolates(self):
        """Negative mode inputs should be rebuilt and yield finite cruise profiles"""
        EDB_data = {
            'ENGINE_TYPE': 'MTF',
            'BP_Ratio': 5.0,
            'SN_matrix': np.array([10.0, 20.0, 25.0, 30.0]),
            'nvPM_mass_matrix': np.full(4, -1.0),
            'nvPM_num_matrix': np.full(4, -1.0),
            'PR': np.array([25.0]),
            'EImass_max': 50.0,
            'EImass_max_thrust': 0.575,
            'EInum_max': 4.5e15,
            'EInum_max_thrust': 0.925,
        }
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
        EDB_data = {
            'ENGINE_TYPE': 'TF',
            'BP_Ratio': 0.0,
            'SN_matrix': np.full(4, -5.0),
            'nvPM_mass_matrix': np.linspace(1.0, 4.0, 4),
            'nvPM_num_matrix': np.linspace(1.0, 4.0, 4),
            'PR': np.array([20.0]),
            'EImass_max': 10.0,
            'EImass_max_thrust': float('nan'),
            'EInum_max': 1.0e12,
            'EInum_max_thrust': float('nan'),
        }
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
        SN_matrix = np.array([5.0, 50.0, -1.0, 0.0])
        PR = np.array([20.0, 20.0, 20.0, 20.0])
        BP_Ratio = np.array([2.0, 1.0, 0.0, 0.0])

        mtf = calculate_PMnvolEI_scope11(SN_matrix, PR, 'MTF', BP_Ratio)
        tf = calculate_PMnvolEI_scope11(SN_matrix, PR, 'TF', BP_Ratio)

        assert mtf.shape == SN_matrix.shape
        assert tf.shape == SN_matrix.shape

        SN0 = min(SN_matrix[0], 40.0)
        CBC0 = 0.6484 * np.exp(0.0766 * SN0) / (1 + np.exp(-1.098 * (SN0 - 3.064)))
        AFR = np.array([106, 83, 51, 45], dtype=float)

        bypass = 1 + BP_Ratio[0]
        kslm_mtf = np.log(
            (3.219 * CBC0 * bypass * 1000 + 312.5) / (CBC0 * bypass * 1000 + 42.6)
        )
        Q_mtf = 0.776 * AFR[0] * bypass + 0.767
        expected_mtf = (kslm_mtf * CBC0 * Q_mtf) / 1000.0
        assert np.isclose(mtf[0], expected_mtf)

        kslm_tf = np.log((3.219 * CBC0 * 1000 + 312.5) / (CBC0 * 1000 + 42.6))
        Q_tf = 0.776 * AFR[0] + 0.767
        expected_tf = (kslm_tf * CBC0 * Q_tf) / 1000.0
        assert np.isclose(tf[0], expected_tf)

        assert mtf[0] > tf[0]
        assert mtf[2] == 0.0
        assert tf[3] == 0.0


class TestEI_PMvol:
    """Tests for EI_PMvol helper functions"""

    def test_fuel_flow_path_uses_lube_contributions(self):
        fuelflow = np.ones((2, 4))
        thrustCat = np.array(['L', 'H'])

        pmvol, ocic = EI_PMvol_FuelFlow(fuelflow, thrustCat)

        assert pmvol.shape == thrustCat.shape
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


class TestLifecycleCO2:
    """Tests for lifecycle_CO2"""

    def test_lifecycle_offset_applies(self):
        fuel = {'LC_CO2': 4500.0, 'EI_CO2': 3150.0}

        result = lifecycle_CO2(fuel, fuel_burn=10.0)

        assert result == pytest.approx(10.0 * (fuel['LC_CO2'] - fuel['EI_CO2']))
        assert result > 0.0


# Integration tests
class TestIntegration:
    """Integration tests to check function interactions"""

    def test_nox_emissions_consistency(self):
        """Test NOx emissions consistency across functions"""
        fuelflow_trajectory = np.array([1.0, 1.5, 2.0])
        NOX_EI_matrix = np.array([30.0, 25.0, 20.0, 18.0])
        fuelflow_performance = np.array([0.8, 1.2, 1.6, 2.0])
        Tamb = np.array([288.15, 250.0, 220.0])
        Pamb = np.array([101325.0, 25000.0, 15000.0])

        result = BFFM2_EINOx(
            fuelflow_trajectory, NOX_EI_matrix, fuelflow_performance, Tamb, Pamb
        )

        assert np.allclose(
            result.NOxEI, np.array([27.11460822, 14.28251747, 11.92937893])
        )
