import tomllib
from pathlib import Path

import numpy as np

from AEIC.BADA.aircraft_parameters import Bada3AircraftParameters
from AEIC.BADA.model import Bada3JetEngineModel
from AEIC.config import LTOInputMode, PerformanceInputMode, config
from AEIC.parsers.LTO_reader import parseLTO
from AEIC.parsers.OPF_reader import parse_OPF
from AEIC.utils.edb import get_EDB_data_for_engine


class PerformanceModel:
    '''Performance model for an aircraft. Contains
    fuel flow, airspeed, ROC/ROD, LTO emissions,
    and OAG schedule'''

    """Loads the configuration and mission set, ingests the
    requested performance input (OPF or TOML input), and prepares performance model
    table, LTO, and APU artefacts needed for downstream analysis.

    Parameters:
        config_file: Path resolved via `file_location` that points to the master TOML
            configuration describing missions, performance data, and emissions settings.
    """

    def __init__(
        self,
        input_file: Path,
        mode: PerformanceInputMode = PerformanceInputMode.PERFORMANCE_MODEL,
    ):
        '''Initializes the performance model by reading the configuration,
        loading mission data, and setting up performance and engine models.'''
        # Process input performance data
        self.mode = mode
        self.input_file = input_file

        self.ac_params = Bada3AircraftParameters()
        match self.mode:
            # If OPF data input
            case PerformanceInputMode.OPF:
                opf_params = parse_OPF(self.input_file)
                for key in opf_params:
                    setattr(self.ac_params, key, opf_params[key])
            # If fuel flow function input
            case PerformanceInputMode.PERFORMANCE_MODEL:
                self.read_performance_data()
                ac_params_input = {
                    "cas_cruise_lo": self.model_info["speeds"]['cruise']['cas_lo'],
                    "cas_cruise_hi": self.model_info["speeds"]['cruise']['cas_hi'],
                    "cas_cruise_mach": self.model_info["speeds"]['cruise']['mach'],
                }
                for key in ac_params_input:
                    setattr(self.ac_params, key, ac_params_input[key])

        # Initialize BADA engine model
        self.engine_model = Bada3JetEngineModel(self.ac_params)

        if config.lto_input_mode == LTOInputMode.INPUT_FILE:
            # Load LTO data
            self.LTO_data = parseLTO(config.lto_input_file)

    def read_performance_data(self):
        '''Parses the TOML input file containing flight and LTO performance data.
        Separates model metadata and prepares the data for table generation.'''

        # Read and load TOML data
        with open(self.input_file, "rb") as f:
            data = tomllib.load(f)

        self.LTO_data = data['LTO_performance']
        match config.lto_input_mode:
            case LTOInputMode.EDB:
                # Read UID
                UID = data['LTO_performance']['ICAO_UID']
                # Read EDB file and get engine
                engine_info = get_EDB_data_for_engine(UID)
                if engine_info is not None:
                    self.EDB_data = engine_info
                else:
                    ValueError(f"No engine with UID={UID} found.")
            case LTOInputMode.PERFORMANCE_MODEL:
                self.EDB_data = data['LTO_performance']

        # Read APU data
        apu_name = data['General_Information']['APU_name']
        with open(config.file_location("engines/APU_data.toml"), "rb") as f:
            APU_data = tomllib.load(f)

        for apu in APU_data.get("APU", []):
            if apu["name"] == apu_name:
                self.APU_data = {
                    "fuel_kg_per_s": apu["fuel_kg_per_s"],
                    "PM10_g_per_kg": apu["PM10_g_per_kg"],
                    "NOx_g_per_kg": apu["NOx_g_per_kg"],
                    "CO_g_per_kg": apu["CO_g_per_kg"],
                    "HC_g_per_kg": apu["HC_g_per_kg"],
                }
                break
            else:
                self.APU_data = {
                    "fuel_kg_per_s": 0.0,
                    "PM10_g_per_kg": 0.0,
                    "NOx_g_per_kg": 0.0,
                    "CO_g_per_kg": 0.0,
                    "HC_g_per_kg": 0.0,
                }

        self.create_performance_table(data['flight_performance'])
        del data["LTO_performance"]
        del data["flight_performance"]
        self.model_info = data

    def create_performance_table(self, data_dict):
        """
        Dynamically creates a multidimensional performance table
        where fuel flow is a function of input variables such as
        flight level, true airspeed, rate of climb/descent, and mass.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing keys 'cols' and 'data' from the input TOML.
        """

        cols = data_dict["cols"]
        data = data_dict["data"]

        # Identify output column (we assume it's the first column or
        # explicitly labeled as fuel flow)
        try:
            output_col_idx = cols.index("FUEL_FLOW")  # Output is fuel flow
        except ValueError:
            raise ValueError("FUEL_FLOW column not found in performance data.")

        input_col_names = [c for i, c in enumerate(cols) if i != output_col_idx]

        # Extract and sort unique values for each input dimension
        input_values = {
            col: sorted(set(row[cols.index(col)] for row in data))
            for col in input_col_names
        }
        input_indices = {
            col: {val: idx for idx, val in enumerate(input_values[col])}
            for col in input_col_names
        }

        # Prepare multidimensional shape and index arrays
        shape = tuple(len(input_values[col]) for col in input_col_names)
        fuel_flow_array = np.zeros(shape)

        # Get index arrays for each input variable
        index_arrays = [
            np.array([input_indices[col][row[cols.index(col)]] for row in data])
            for col in input_col_names
        ]
        index_arrays = tuple(index_arrays)

        # Extract output (fuel flow) values
        fuel_flow = np.array([row[output_col_idx] for row in data])

        # Assign to multidimensional array using advanced indexing
        fuel_flow_array[index_arrays] = fuel_flow

        # Save results
        self.performance_table = fuel_flow_array
        self.performance_table_cols = [input_values[col] for col in input_col_names]
        self.performance_table_colnames = input_col_names  # Save for external reference

    def search_mass_ind(self, mass: float) -> list[int]:
        """Searches the valid mass values in the performance model for the indices
        bounding a known mass value.

        Args:
            mass (float): Mass value of interest.

        Returns:
            (list[int]) List containing the indices of the mass values in performance
                data that bound the given mass.
        """

        mass_ind_high = np.searchsorted(self.performance_table_cols[-1], mass)

        if mass_ind_high == 0:
            raise ValueError('Aircraft is trying to fly below minimum mass')
        if mass_ind_high == len(self.performance_table_cols[0]):
            raise ValueError('Aircraft is trying to fly above maximum mass')

        return [int(mass_ind_high - 1), int(mass_ind_high)]

    def search_flight_levels_ind(self, FL: float) -> list[int]:
        """Searches the valid flight levels in the performance model for the indices
        bounding a known FL value.

        Args:
            FL (float): Flight level of interest.

        Returns:
            (list[int]) List containing the indices of the FLs in the performance data
                that bound the given FL.
        """

        FL_ind_high = np.searchsorted(self.performance_table_cols[0], FL)

        if FL_ind_high == 0:
            raise ValueError(
                f"Aircraft is trying to fly below minimum cruise altitude(FL {FL:.2f})"
            )
        if FL_ind_high == len(self.performance_table_cols[0]):
            raise ValueError(
                f"Aircraft is trying to fly above maximum cruise altitude(FL {FL:.2f})"
            )

        return [int(FL_ind_high - 1), int(FL_ind_high)]
