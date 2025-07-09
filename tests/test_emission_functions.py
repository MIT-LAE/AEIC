from unittest.mock import patch

import numpy as np
import pytest

# Add the src directory to the path to import modules
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.legacy_trajectory import LegacyTrajectory
from emissions.APU_emissions import get_APU_emissions
from emissions.EI_CO2 import EI_CO2
from emissions.EI_H2O import EI_H2O
from emissions.EI_HCCO import EI_HCCO
from emissions.EI_NOx import BFFM2_EINOx, NOx_speciation
from emissions.EI_SOx import EI_SOx
from emissions.emission import Emission
from utils import file_location

# Path to a real fuel TOML file in your repo
performance_model_file = file_location("IO/default_config.toml")

perf = PerformanceModel(performance_model_file)
mis = perf.missions[0]

traj = LegacyTrajectory(perf, mis, False, False)
traj.fly_flight()
em = Emission(perf, traj, True)


class TestEI_CO2:
    """Tests for EI_CO2 function"""

    def test_basic_functionality(self):
        """Test basic CO2 emissions calculation"""
        fuel = {'EI_CO2': 3160.0, 'nvolCarbCont': 0.95}
        co2_ei, nvol_carb = EI_CO2(fuel)

        assert co2_ei == 3160.0
        assert nvol_carb == 0.95
        assert isinstance(co2_ei, (int, float))
        assert isinstance(nvol_carb, (int, float))

    def test_non_negativity(self):
        """Test that outputs are non-negative"""
        fuel = {'EI_CO2': 3160.0, 'nvolCarbCont': 0.95}
        co2_ei, nvol_carb = EI_CO2(fuel)

        assert co2_ei >= 0
        assert nvol_carb >= 0

    def test_finiteness(self):
        """Test that outputs are finite"""
        fuel = {'EI_CO2': 3160.0, 'nvolCarbCont': 0.95}
        co2_ei, nvol_carb = EI_CO2(fuel)

        assert np.isfinite(co2_ei)
        assert np.isfinite(nvol_carb)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Missing keys
        with pytest.raises(KeyError):
            EI_CO2({})

        # Invalid data types - should work but test for reasonable values
        fuel = {'EI_CO2': -100, 'nvolCarbCont': -0.5}
        co2_ei, nvol_carb = EI_CO2(fuel)
        # Function doesn't validate inputs, but we can check they're returned as-is
        assert co2_ei == -100
        assert nvol_carb == -0.5


class TestEI_H2O:
    """Tests for EI_H2O function"""

    def test_basic_functionality(self):
        """Test basic H2O emissions calculation"""
        fuel = {'EI_H2O': 1230.0}
        h2o_ei = EI_H2O(fuel)

        assert h2o_ei == 1230.0
        assert isinstance(h2o_ei, (int, float))

    def test_non_negativity(self):
        """Test reasonable values are non-negative"""
        fuel = {'EI_H2O': 1230.0}
        h2o_ei = EI_H2O(fuel)
        assert h2o_ei >= 0

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


class TestBFFM2_EINOx:
    """Tests for BFFM2_EINOx function"""

    def setup_method(self):
        """Set up test data"""
        self.fuelflow_trajectory = np.array([0.5, 1.0, 1.5, 2.0])
        self.NOX_EI_matrix = np.array([30.0, 25.0, 20.0, 18.0])
        self.fuelflow_performance = np.array([0.4, 0.8, 1.2, 1.8])
        self.Tamb = np.array([288.15, 250.0, 220.0, 280.0])
        self.Pamb = np.array([101325.0, 25000.0, 15000.0, 95000.0])

    def test_basic_functionality(self):
        """Test basic NOx emissions calculation"""
        results = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        # Should return 7 arrays
        assert len(results) == 7
        NOxEI, NOEI, NO2EI, HONOEI, noProp, no2Prop, honoProp = results

        # Check shapes
        expected_shape = self.fuelflow_trajectory.shape
        for result in results:
            assert result.shape == expected_shape

    def test_non_negativity(self):
        """Test that all outputs are non-negative"""
        results = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        for result in results:
            assert np.all(result >= 0)

    def test_summation_consistency(self):
        """Test that NO + NO2 + HONO proportions sum to 1"""
        results = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        NOxEI, NOEI, NO2EI, HONOEI, noProp, no2Prop, honoProp = results

        # Proportions should sum to 1
        total_prop = noProp + no2Prop + honoProp
        assert np.allclose(total_prop, 1.0, rtol=1e-10)

        # Component EIs should sum to total NOx EI
        total_component_EI = NOEI + NO2EI + HONOEI
        assert np.allclose(total_component_EI, NOxEI, rtol=1e-10)

    def test_finiteness(self):
        """Test that all outputs are finite"""
        results = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        for result in results:
            assert np.all(np.isfinite(result))

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
        assert not np.allclose(results_no_cruise[0], results_with_cruise[0])

    @patch('utils.standard_fuel.get_thrust_cat')
    def test_thrust_categorization(self, mock_get_thrust_cat):
        """Test thrust categorization functionality"""
        # Mock thrust categories
        mock_get_thrust_cat.return_value = np.array(
            [1, 2, 3, 1]
        )  # High, Low, Approach, High

        results = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.NOX_EI_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )

        # Should still return valid results
        assert len(results) == 7
        for result in results:
            assert np.all(np.isfinite(result))


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

        SO2EI, SO4EI = EI_SOx(fuel)

        assert isinstance(SO2EI, (int, float))
        assert isinstance(SO4EI, (int, float))

    def test_non_negativity(self):
        """Test that outputs are non-negative"""
        fuel = {'FSCnom': 600.0, 'Epsnom': 0.02}

        SO2EI, SO4EI = EI_SOx(fuel)

        assert SO2EI >= 0
        assert SO4EI >= 0

    def test_mass_balance(self):
        """Test that SO2 + SO4 production makes sense relative to sulfur content"""
        fuel = {
            'FSCnom': 600.0,  # 600 ppm sulfur
            'Epsnom': 0.02,  # 2% converted to SO4
        }

        SO2EI, SO4EI = EI_SOx(fuel)

        # Calculate total sulfur converted (should be proportional to FSC)
        MW_SO2, MW_SO4, MW_S = 64.0, 96.0, 32.0

        # Back-calculate sulfur content from emissions
        sulfur_as_SO2 = SO2EI * MW_S / MW_SO2
        sulfur_as_SO4 = SO4EI * MW_S / MW_SO4
        total_sulfur_converted = sulfur_as_SO2 + sulfur_as_SO4

        # Should be approximately equal to input sulfur (with unit conversions)
        expected_sulfur = fuel['FSCnom'] / 1000  # Convert ppm to g/kg
        assert np.isclose(total_sulfur_converted, expected_sulfur, rtol=0.01)

    def test_finiteness(self):
        """Test that outputs are finite"""
        fuel = {'FSCnom': 600.0, 'Epsnom': 0.02}

        SO2EI, SO4EI = EI_SOx(fuel)

        assert np.isfinite(SO2EI)
        assert np.isfinite(SO4EI)

    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(KeyError):
            EI_SOx({})

        # Test with zero values
        fuel = {'FSCnom': 0.0, 'Epsnom': 0.0}
        SO2EI, SO4EI = EI_SOx(fuel)
        assert SO2EI == 0.0
        assert SO4EI == 0.0


class TestGetAPUEmissions:
    """Tests for get_APU_emissions function"""

    def setup_method(self):
        """Set up test data"""
        dtype = [
            ('SO2', 'f8'),
            ('SO4', 'f8'),
            ('PMnvol', 'f8'),
            ('PMvol', 'f8'),
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
        )

        assert apu_ei['SO2'] == 0.0
        assert apu_ei['SO4'] == 0.0
        assert apu_ei['CO2'] == 0.0
        assert apu_g['CO2'] == 0.0


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

        results = BFFM2_EINOx(
            fuelflow_trajectory, NOX_EI_matrix, fuelflow_performance, Tamb, Pamb
        )

        NOxEI, *_ = results
        assert np.allclose(NOxEI, np.array([27.11460822, 14.28251747, 11.92937893]))


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
