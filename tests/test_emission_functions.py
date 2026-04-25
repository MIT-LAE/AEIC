import tomllib

import numpy as np
import pytest

from AEIC.config import config
from AEIC.emissions.apu import get_APU_emissions
from AEIC.emissions.ei.hcco import EI_HCCO
from AEIC.emissions.ei.nox import BFFM2_EINOx, BFFM2EINOxResult, NOx_speciation
from AEIC.emissions.ei.nvpm import calculate_nvPM_scope11_LTO, nvPM_MEEM
from AEIC.emissions.ei.sox import EI_SOx, SOxEmissionResult
from AEIC.emissions.types import AtmosphericState
from AEIC.emissions.utils import get_thrust_cat_cruise
from AEIC.performance.apu import APU
from AEIC.performance.edb import EDBEntry
from AEIC.performance.types import ThrustMode, ThrustModeValues
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
        self.fuelflow_evaluate = np.array([0.15, 0.175, 0.275, 0.325, 0.25, 0.16])
        self.x_EI_matrix = ThrustModeValues(1.54, 0.05, 0.02, 0.03)
        self.fuelflow_calibrate = ThrustModeValues(0.4, 0.8, 1.2, 1.8)
        # Standard atmosphere conditions.
        self.Tamb = np.array([288.15, 278.4, 249.15, 216.65, 229.65, 275.15])
        self.Pamb = np.array(
            [
                101325.0,
                84555.9940737564,
                47181.0021852292,
                22632.0400950078,
                30742.4326120969,
                79495.201934051,
            ]
        )

    def test_HC_outputs(self):
        result = EI_HCCO(
            self.fuelflow_evaluate,
            self.x_EI_matrix,
            self.fuelflow_calibrate,
            self.Tamb,
            self.Pamb,
        )
        out_result = np.array(
            [
                196.73236113,
                98.54714941,
                13.25415652,
                7.73938389,
                25.11776059,
                157.25298236,
            ]
        )
        np.testing.assert_allclose(result, out_result)

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
        test_flow = np.array([0.0, -0.01, 0.1, 1.0, 0.001, 10.1])
        result = EI_HCCO(
            test_flow, self.x_EI_matrix, self.fuelflow_calibrate, self.Tamb, self.Pamb
        )
        assert result.shape == test_flow.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_intercept_adjustment_uses_second_mode_value(self):
        """When intercept drifts low, the second mode should set the ceiling"""
        x_EI_matrix = ThrustModeValues(
            38.33753758, 2.4406048, 106.49710981, 13.57427593
        )
        fuelflow_calibrate = ThrustModeValues(
            0.10569869, 0.40041291, 0.81271722, 0.86727924
        )

        result = EI_HCCO(
            self.fuelflow_evaluate,
            x_EI_matrix,
            fuelflow_calibrate,
            self.Tamb,
            self.Pamb,
        )
        high_mask = self.fuelflow_evaluate >= fuelflow_calibrate[ThrustMode.APPROACH]

        assert np.allclose(result[high_mask], x_EI_matrix[ThrustMode.APPROACH])
        assert np.all(result >= 0.0)

    def test_branches_split_at_intercept(self):
        """Pin the lower (slanted-line) and upper (horizontal-line) branches
        of `EI_HCCO` (`hcco.py:115–128`). The previous test only ever
        landed in the upper segment because its calibration parameters
        forced the intercept-adjustment branch (b). Use a calibration set
        where (a) slope is non-zero and negative (so the slanted formula
        actually runs in the lower mask) and (b) the intercept lands
        squarely between the IDLE and CLIMB calibration flows; pick
        evaluation flows on each side, plus a mixed array, and verify
        each lands in the right segment.

        Constants verified by direct call to the SUT once and pinned
        here — the test is not a tautological copy of the formula, but a
        regression guard against either branch's expression drifting.
        """
        # Strictly-decreasing-with-flow EI to guarantee a finite negative
        # log-log slope and an intercept inside the calibration range.
        x_EI = ThrustModeValues(100.0, 50.0, 10.0, 5.0)
        ff_cal = ThrustModeValues(0.1, 0.3, 1.0, 2.0)
        # ISA ground conditions so the cruise correction is the identity
        # factor (theta=delta=1.0).
        Tamb_ground = np.array([288.15])
        Pamb_ground = np.array([101325.0])

        # Lower segment — well below intercept.
        low_only = EI_HCCO(np.array([0.05]), x_EI, ff_cal, Tamb_ground, Pamb_ground)
        # Upper segment — well above intercept.
        high_only = EI_HCCO(np.array([1.5]), x_EI, ff_cal, Tamb_ground, Pamb_ground)
        # Horizontal-line value is 10**(0.5*(log10(xEI[CLIMB])+log10(xEI[TAKEOFF]))),
        # which is the geometric mean of CLIMB and TAKEOFF EI values.
        expected_high = np.sqrt(x_EI[ThrustMode.CLIMB] * x_EI[ThrustMode.TAKEOFF])
        assert high_only[0] == pytest.approx(expected_high)

        # Lower-segment values must be strictly above the upper segment
        # (since EI is decreasing with flow on a negative slope), and
        # well above zero.
        assert low_only[0] > expected_high
        assert low_only[0] > 0

        # Mixed array: half low, half high. The split should match the
        # individual-call results.
        mixed_ff = np.array([0.05, 1.5])
        mixed_T = np.array([288.15, 288.15])
        mixed_P = np.array([101325.0, 101325.0])
        mixed = EI_HCCO(mixed_ff, x_EI, ff_cal, mixed_T, mixed_P)
        assert mixed[0] == pytest.approx(low_only[0])
        assert mixed[1] == pytest.approx(high_only[0])


class TestBFFM2_EINOx:
    """Tests for BFFM2_EINOx function"""

    def setup_method(self):
        """Set up test data"""
        self.fuelflow_trajectory = np.array(
            [0.15377735, 0.19154479, 0.36525745, 0.54475802, 0.37317567, 0.17729855]
        )
        self.EI_NOx_matrix = ThrustModeValues(30.0, 25.0, 20.0, 18.0)
        self.fuelflow_performance = ThrustModeValues(0.4, 0.8, 1.2, 1.8)
        self.Tamb = np.array([288.15, 278.4, 249.15, 216.65, 229.65, 275.15])
        self.Pamb = np.array(
            [
                101325.0,
                84555.9940737564,
                47181.0021852292,
                22632.0400950078,
                30742.4326120969,
                79495.201934051,
            ]
        )

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

    def test_thrust_categorization(self):
        """The speciation proportions returned by `BFFM2_EINOx` must equal
        the per-thrust-category values from `NOx_speciation()` once each
        evaluation point has been mapped through `get_thrust_cat_cruise`.

        Previously this `@patch`'d `'AEIC.emissions.utils.get_thrust_cat_cruise'`,
        but `ei/nox.py` imports the name into its own module, so the
        patch missed the bound reference and the SUT ran unmocked — the
        test was effectively a finiteness smoke duplicating
        `test_finiteness`. Drop the mock and pin the actual
        category-to-speciation contract.

        Construct an evaluation-flow array that spans all three thrust
        categories so a regression that collapsed the categorization to
        a single bucket would change `noProp` / `no2Prop` / `honoProp`
        and trip this test.
        """
        # Spans IDLE (≤0.6), APPROACH ((0.6, 1.0]), and CLIMB/TO (>1.0)
        # given `fuelflow_performance = (0.4, 0.8, 1.2, 1.8)`.
        ff_eval = np.array([0.2, 0.5, 0.9, 1.5])
        Tamb = np.full_like(ff_eval, 288.15)
        Pamb = np.full_like(ff_eval, 101325.0)

        result = BFFM2_EINOx(
            ff_eval,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            Tamb,
            Pamb,
        )

        cats = get_thrust_cat_cruise(ff_eval, self.fuelflow_performance)
        # Sanity check that we actually exercised more than one bucket —
        # without this, a regression that made `get_thrust_cat_cruise`
        # return a constant could still satisfy the per-point identity.
        assert len(set(cats.data.tolist())) >= 2

        spec = NOx_speciation()
        expected_no = np.array([spec.no[cat] for cat in cats])
        expected_no2 = np.array([spec.no2[cat] for cat in cats])
        expected_hono = np.array([spec.hono[cat] for cat in cats])
        np.testing.assert_allclose(result.noProp, expected_no)
        np.testing.assert_allclose(result.no2Prop, expected_no2)
        np.testing.assert_allclose(result.honoProp, expected_hono)

    def test_matches_reference_component_values(self):
        """Reference regression to guard against inadvertent logic changes"""
        result = BFFM2_EINOx(
            self.fuelflow_trajectory,
            self.EI_NOx_matrix,
            self.fuelflow_performance,
            self.Tamb,
            self.Pamb,
        )
        # The results below were generated in test-cases.ipynb
        expected_arrays = [
            np.array(
                [
                    42.65302497,
                    39.87840171,
                    30.13039678,
                    22.9420127,
                    27.77833904,
                    40.95955377,
                ]
            ),
            np.array(
                [5.49904124, 5.14132294, 3.88456141, 2.95779899, 3.58132236, 5.28071047]
            ),
            np.array(
                [
                    35.2345976,
                    32.94255069,
                    24.88996752,
                    18.95182314,
                    22.94699142,
                    33.83566338,
                ]
            ),
            np.array(
                [1.91938612, 1.79452808, 1.35586786, 1.03239057, 1.25002526, 1.84317992]
            ),
            np.array([0.128925, 0.128925, 0.128925, 0.128925, 0.128925, 0.128925]),
            np.array([0.826075, 0.826075, 0.826075, 0.826075, 0.826075, 0.826075]),
            np.array([0.045, 0.045, 0.045, 0.045, 0.045, 0.045]),
        ]
        for array, expected in zip(self._components(result), expected_arrays):
            np.testing.assert_allclose(array, expected, rtol=1e-6, atol=1e-9)


class TestNOxSpeciation:
    """Tests for NOₓ_speciation function"""

    def test_NOx_speciation_results(self):
        """Test that speciation results are the same as AEIC v2 matlab implementation"""
        hono_prop = np.array([0.045, 0.045, 0.0075, 0.0075])
        no2_prop = np.array([0.826075, 0.1528, 0.0744375, 0.0744375])
        no_prop = np.array([0.128925, 0.8022, 0.9180625, 0.9180625])

        speciation = NOx_speciation()

        for idx, mode in enumerate(ThrustMode):
            assert np.isclose(speciation.hono[mode], hono_prop[idx], rtol=1.0e-6)
            assert np.isclose(speciation.no2[mode], no2_prop[idx], rtol=1.0e-6)
            assert np.isclose(speciation.no[mode], no_prop[idx], rtol=1.0e-6)

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
    """Tests for get_APU_emissions function.

    NOTE: The numeric APU parameters below (`fuel_kg_per_s=0.1`,
    `NOx_g_per_kg=15.0`, `apu_time=2854`, etc.) are **synthetic test
    inputs**, not values drawn from a published APU dataset. They are
    chosen to produce numerically distinguishable outputs across modes,
    not to match any real airframe's APU. Treat them accordingly when
    porting this fixture or interpreting result magnitudes.
    """

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

    def test_zero_fuel_flow_handling(self, fuel_jetA):
        """Test handling of zero fuel flow"""
        apu_data_zero = self.apu.model_copy(update={'fuel_kg_per_s': 0.0})

        apu = get_APU_emissions(self.LTO_emission_indices, apu_data_zero, fuel_jetA)

        assert apu.indices[Species.SO2] == 0.0
        assert apu.indices[Species.SO4] == 0.0
        assert apu.indices[Species.CO2] == 0.0
        assert apu.emissions[Species.CO2] == 0.0

    @pytest.mark.config_updates(emissions__nvpm_method='meem')
    def test_nvpm_method_enables_number_channel(self, fuel_jetA):
        """PM number index should be emitted when nvpm_method requests it"""
        apu = get_APU_emissions(self.LTO_emission_indices, self.apu, fuel_jetA)

        assert Species.nvPM_N in apu.indices
        assert apu.indices[Species.nvPM_N] == 0.0


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


class TestNvPMMEEM:
    """Tests for the nvPM_MEEM cruise methodology."""

    def test_MEEM_using_test_cases_data(self):
        edb_data = EDBEntry(
            engine='Test',
            uid='TEST000',
            engine_type='TF',
            BP_Ratio=5.1,
            rated_thrust=120.0,
            fuel_flow=make_edb_lto_values(0, 0, 0, 0),
            CO_EI_matrix=make_edb_lto_values(0, 0, 0, 0),
            HC_EI_matrix=make_edb_lto_values(0, 0, 0, 0),
            EI_NOx_matrix=make_edb_lto_values(0, 0, 0, 0),
            SN_matrix=make_edb_lto_values(10, 20, 25, 30),
            nvPM_mass_matrix=make_edb_lto_values(1.1, 2.7, 53.2, 82.1),
            nvPM_num_matrix=make_edb_lto_values(26.6e12, 71e12, 433e12, 402e12),
            PR=make_edb_lto_values(25, 25, 25, 25),
            EImass_max=50.0,
            EImass_max_thrust=0.575,
            EInum_max=4.5e15,
            EInum_max_thrust=0.925,
        )
        altitudes = np.array([6000.0, 33000.0, 12000.0]) / 3.28084
        TAS = np.array([200.0, 240.0, 190.0])
        rocd = np.array([1.0, 0.0, -1.0])

        atmos_state = AtmosphericState(altitudes, TAS)

        nvPM_profile = nvPM_MEEM(edb_data, altitudes, rocd, atmos_state)
        EI_mass = nvPM_profile.mass
        EI_num = nvPM_profile.number

        # These numbers come from MEEM test cases generated in
        # notebooks/test-cases.ipynb
        ref_EI_mass = np.array([87.80422697, 19.50901914, 1.40599907]) * 1e-3
        ref_EI_num = np.array([4.72211990e14, 1.74290262e14, 3.50345395e13])

        # Match the BFFM2 tolerance scheme: numpy default atol=1e-8 is
        # negligible against EI_num values of order 1e13–1e14, so the
        # implicit check would degenerate to relative-only with rtol=1e-5.
        np.testing.assert_allclose(EI_mass, ref_EI_mass, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(EI_num, ref_EI_num, rtol=1e-6, atol=1e-9)


class Test_nvPMScope11:
    """Tests for calculate_nvPM_scope11_LTO"""

    def test_SCOPE11_unit_test(self):
        SN_matrix = ThrustModeValues(2.1, 2.1, 11.2, 13.4)
        BP_ratio = 5.1
        profile = calculate_nvPM_scope11_LTO(SN_matrix, "TF", BP_ratio)
        # These numbers come from SCOPE11 test cases generated in
        # notebooks/test-cases.ipynb
        ref_mass = (
            np.array(
                [
                    22.374162153861967,
                    17.564241455716534,
                    74.21034231130484,
                    77.37542370682733,
                ]
            )
            * 1e-3
        )
        ref_num = np.array([1.12837170e15, 8.85798223e14, 4.67821152e14, 4.87773789e14])
        for i, mode in enumerate(ThrustMode):
            # `np.allclose` defaults of atol=1e-8 are negligible against
            # number values of order 1e14–1e15; pin the BFFM2-style
            # rtol=1e-6, atol=1e-9 scheme explicitly so a regression in
            # either channel surfaces, and use `assert_allclose` so the
            # failure reports the diff.
            np.testing.assert_allclose(
                profile.mass[mode], ref_mass[i], rtol=1e-6, atol=1e-9
            )
            np.testing.assert_allclose(
                profile.number[mode], ref_num[i], rtol=1e-6, atol=1e-9
            )

    def test_scope11_engine_type_scaling(self):
        """`calculate_nvPM_scope11_LTO` applies a bypass-ratio correction to
        the MTF (mixed turbofan) path that the TF path does not. The full
        numeric contract for both engine types at every mode is pinned by
        `test_SCOPE11_unit_test` against the notebook reference values;
        here, just pin the qualitative invariant — at IDLE on a finite
        positive SN, MTF > TF — without inline-computing the
        SUT formula. Inline-computation would be a tautological copy of
        `nvpm.py` and a regression that broke both implementations the
        same way would still pass.
        """
        SN_matrix = ThrustModeValues(5.0, 50.0, 30.0, 40.0)
        BP_Ratio = 2.0

        mtf = calculate_nvPM_scope11_LTO(SN_matrix, 'MTF', BP_Ratio)
        tf = calculate_nvPM_scope11_LTO(SN_matrix, 'TF', BP_Ratio)

        # Bypass correction makes MTF strictly larger than TF at IDLE.
        assert mtf.mass[ThrustMode.IDLE] > tf.mass[ThrustMode.IDLE] > 0
        # Number-channel parity check: both engine types yield non-empty
        # number profiles.
        assert mtf.number[ThrustMode.IDLE] > 0
        assert tf.number[ThrustMode.IDLE] > 0

    def test_scope11_invalid_smoke_numbers_return_zero(self):
        """Per `emissions/ei/nvpm.py`, smoke numbers ≤ 0 are treated as
        invalid and the SUT must emit zero in both mass and number for
        the corresponding modes. Bug guard: a future refactor that
        propagated -1.0 / 0.0 through the CBC0 expression would either
        produce NaN or a negative emission — both of which would slip
        through if only "non-zero" was asserted.
        """
        # IDLE has a finite positive SN so the rest of the modes can be
        # invalid without the SUT raising on a degenerate input set.
        SN_matrix = ThrustModeValues(5.0, 50.0, -1.0, 0.0)
        for engine_type in ('MTF', 'TF'):
            profile = calculate_nvPM_scope11_LTO(SN_matrix, engine_type, BP_Ratio=2.0)
            assert profile.mass[ThrustMode.CLIMB] == 0.0
            assert profile.mass[ThrustMode.TAKEOFF] == 0.0
            assert profile.number[ThrustMode.CLIMB] == 0.0
            assert profile.number[ThrustMode.TAKEOFF] == 0.0


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
            result.NOxEI, np.array([26.75988671, 14.4120521, 11.92014638])
        )
