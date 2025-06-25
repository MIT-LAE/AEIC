import gc
import os

import numpy as np
import tomllib

from BADA.aircraft_parameters import Bada3AircraftParameters
from BADA.model import Bada3JetEngineModel
from parsers.LTO_reader import parseLTO
from parsers.OPF_reader import parse_OPF

# from src.missions.OAG_filter import filter_OAG_schedule
from utils import file_location


class PerformanceModel:
    '''Performance model for an aircraft. Contains
    fuel flow, airspeed, ROC/ROD, LTO emissions,
    and OAG schedule'''

    def __init__(self, config_file="IO/default_config.toml"):
        '''Initializes the performance model by reading the configuration,
        loading mission data, and setting up performance and engine models.'''
        config_file_loc = file_location(config_file)
        self.config = {}
        with open(config_file_loc, 'rb') as f:
            config_data = tomllib.load(f)
            self.config = {
                k: v for subdict in config_data.values() for k, v in subdict.items()
            }

        # Get mission data
        # self.filter_OAG_schedule = filter_OAG_schedule
        mission_file = file_location(
            os.path.join(
                self.config['missions_folder'], self.config['missions_in_file']
            )
        )
        with open(mission_file, 'rb') as f:
            all_missions = tomllib.load(f)
            self.missions = all_missions['flight']
        # self.schedule = filter_OAG_schedule()

        # Process input performance data
        self.initialize_performance()

    def initialize_performance(self):
        '''Initializes aircraft performance characteristics from TOML sourcee.
        Also loads LTO/EDB data and sets up the engine model using BADA3 parameters.'''

        self.ac_params = Bada3AircraftParameters()
        # If OPF data input
        if self.config["performance_model_input"] == "OPF":
            opf_params = parse_OPF(
                file_location(self.config["performance_model_input_file"])
            )
            for key in opf_params:
                setattr(self.ac_params, key, opf_params[key])
        # If fuel flow function input
        elif self.config["performance_model_input"] == "PerformanceModel":
            self.read_performance_data()
            ac_params_input = {
                "cas_cruise_lo": self.model_info["speeds"]['cruise']['cas_lo'],
                "cas_cruise_hi": self.model_info["speeds"]['cruise']['cas_hi'],
                "cas_cruise_mach": self.model_info["speeds"]['cruise']['mach'],
            }
            for key in ac_params_input:
                setattr(self.ac_params, key, ac_params_input[key])
        else:
            print("Invalid performance model input provided!")

        # Initialize BADA engine model
        self.engine_model = Bada3JetEngineModel(self.ac_params)

        if self.config["LTO_input_mode"] == "input_file":
            # Load LTO data
            self.LTO_data = parseLTO(self.config['LTO_input_file'])

    def read_performance_data(self):
        '''Parses the TOML input file containing flight and LTO performance data.
        Separates model metadata and prepares the data for table generation.'''

        # Read and load TOML data
        with open(
            file_location(self.config["performance_model_input_file"]), "rb"
        ) as f:
            data = tomllib.load(f)

        self.LTO_data = data['LTO_performance']
        if self.config["LTO_input_mode"] == "EDB":
            # Read UID
            UID = data['LTO_performance']['ICAO_UID']
            # Read EDB file and get engine
            engine_info = self.get_engine_by_uid(UID, self.config["edb_engine_file"])
            if engine_info is not None:
                self.EDB_data = engine_info
            else:
                ValueError(f"No engine with UID={UID} found.")

        # Read APU data
        apu_name = data['General_Information']['APU_name']
        with open(file_location("engines/APU_data.toml"), "rb") as f:
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
        fuel_flow_array = np.empty(shape)

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

    def get_engine_by_uid(self, uid: str, toml_path: str) -> dict:
        """
        Reads a TOML file containing multiple [[engine]] tables, finds and returns
        the engine dict whose 'UID' field matches the given uid. After locating
        the matching table, the entire TOML parse tree is deleted to free memory.

        Parameters
        ----------
        uid : str
            The UID string to search for (e.g. "1RR021").
        toml_path : str
            Path to the TOML file to read.

        Returns
        -------
        dict or None
            The dict corresponding to the matching [[engine]] table if found;
            otherwise, None.
        """
        # Open and parse the TOML file
        edb_file_loc = file_location(toml_path)
        with open(edb_file_loc, 'rb') as f:
            data = tomllib.load(f)

        # data["engine"] is a list of dicts (one per [[engine]] table)
        engines = data.get("engine", [])

        # Search for the matching UID
        match = None
        for engine in engines:
            if engine.get("UID") == uid:
                match = engine
                break

        # Remove the parsed data from memory
        del data
        del engines
        gc.collect()

        return match
