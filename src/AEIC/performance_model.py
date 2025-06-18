import numpy as np
import tomllib
import json
import os
import gc
from parsers.PTF_reader import parse_PTF
from parsers.OPF_reader import parse_OPF
from parsers.LTO_reader import parseLTO
from BADA.aircraft_parameters import Bada3AircraftParameters
from BADA.model import Bada3JetEngineModel
from utils import file_location

class PerformanceModel:
    '''Performance model for an aircraft. Contains
        fuel flow, airspeed, ROC/ROD, LTO emissions,
        and OAG schedule'''

    def __init__(self, config_file="IO/default_config.toml"):
        # Read config file and store all variables in self.config
        config_file_loc = file_location(config_file)
        self.config = {}
        with open(config_file_loc, 'rb') as f:
            config_data = tomllib.load(f)
            self.config = {k: v for subdict in config_data.values() for k, v in subdict.items()}

        # Get mission data
        # self.filter_OAG_schedule = filter_OAG_schedule
        mission_file = file_location(
            os.path.join(self.config['missions_folder'], self.config['missions_in_file'])
        )
        with open(mission_file, 'rb') as f:
            all_missions = tomllib.load(f)
            self.missions = all_missions['flight']
        # self.schedule = filter_OAG_schedule()

        # Process input performance data
        self.initialize_performance()

    def initialize_performance(self):
        '''Takes input data given on aircraft performance
            and creates the state variable array'''
        
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
        
    def read_performance_data(self):
        '''Parses input json data of aircraft performance'''
        
        # Read and load TOML data 
        with open(file_location(self.config["performance_model_input_file"]), "rb") as f:
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
        

        self.create_performance_table(data['flight_performance']['data'])

        del data["LTO_performance"]
        del data["flight_performance"]
        self.model_info = data

    def create_performance_table(self,data):
        # Extract unique values for each dimension

        # TODO: need to somehow do this dynamically since not certain that col order is the same/more values
        self.fuel_flow_values = np.array([row[0] for row in data])
        self.EI_NOx_values = np.array([row[5] for row in data])
        fl_values = sorted(set(row[1] for row in data))
        tas_values = sorted(set(row[2] for row in data))
        rocd_values = sorted(set(row[3] for row in data))
        mass_values = sorted(set(row[4] for row in data))
        
        
        # Create mapping dictionaries for fast lookups
        fl_indices = {val: idx for idx, val in enumerate(fl_values)}
        tas_indices = {val: idx for idx, val in enumerate(tas_values)}
        rocd_indices = {val: idx for idx, val in enumerate(rocd_values)}
        mass_indices = {val: idx for idx, val in enumerate(mass_values)}
        
        shape = (len(fl_values), len(tas_values), len(rocd_values), len(mass_values))
        fuel_flow_array = np.empty(shape)
        
        # Populate the array using vectorized approach
        fl_idx = np.array([fl_indices[row[1]] for row in data])
        tas_idx = np.array([tas_indices[row[2]] for row in data])
        rocd_idx = np.array([rocd_indices[row[3]] for row in data])
        mass_idx = np.array([mass_indices[row[4]] for row in data])
        fuel_flow = np.array([row[0] for row in data])
        
        # Use advanced indexing to assign values
        fuel_flow_array[fl_idx, tas_idx, rocd_idx, mass_idx] = fuel_flow
        
        self.performance_table = fuel_flow_array
        self.performance_table_cols = [fl_values, tas_values, rocd_values, mass_values]

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


