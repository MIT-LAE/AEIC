import numpy as np
import tomllib
import json
import os
from src.parsers.PTF_reader import parse_PTF
from src.parsers.OPF_reader import parse_OPF
from src.parsers.LTO_reader import parseLTO
from src.BADA.aircraft_parameters import Bada3AircraftParameters
from src.BADA.model import Bada3JetEngineModel
# from src.missions.OAG_filter import filter_OAG_schedule

class PerformanceModel:
    '''Performance model for an aircraft. Contains
        fuel flow, airspeed, ROC/ROD, LTO emissions,
        and OAG schedule'''

    def __init__(self, config_file_loc="./IO/default_config.toml"):
        # Read config file and store all variables in self.config
        self.config = {}
        with open(config_file_loc, 'rb') as f:
            config_data = tomllib.load(f)
            self.config = {k: v for subdict in config_data.values() for k, v in subdict.items()}

        # Get mission data
        # self.filter_OAG_schedule = filter_OAG_schedule
        mission_file = os.path.join(self.config['missions_folder'], self.config['missions_in_file'])
        with open(mission_file, 'r') as f:
            self.missions = json.load(f)
        # self.schedule = filter_OAG_schedule()

        # Process input performance data
        self.initialize_performance()

    def initialize_performance(self):
        '''Takes input data given on aircraft performance
            and creates the state variable array'''
        
        self.ac_params = Bada3AircraftParameters()
        # If OPF data input
        if self.config["performance_model_input"] == "OPF":
            opf_params = parse_OPF(self.config["performance_model_input_file"])
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
        '''Parses input json data of aircraft performance'''
        
        # Read and load TOML data 
        with open(self.config["performance_model_input_file"], "rb") as f:
            data = tomllib.load(f)

        self.LTO_data = data['LTO_performance']
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


