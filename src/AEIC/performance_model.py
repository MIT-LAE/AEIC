import gc
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from AEIC.BADA.aircraft_parameters import Bada3AircraftParameters
from AEIC.BADA.model import Bada3JetEngineModel
from AEIC.missions import Mission
from AEIC.parsers.LTO_reader import parseLTO
from AEIC.parsers.OPF_reader import parse_OPF
from AEIC.utils.files import file_location
from AEIC.utils.inspect_inputs import require_str


class PerformanceInputMode(Enum):
    """Config for selecting input modes Performance Model"""

    # INPUT OPTIONS
    OPF = "opf"
    PERFORMANCE_MODEL = "performancemodel"

    @classmethod
    def from_value(cls, value: str | None) -> "PerformanceInputMode":
        normalized = (value or cls.PERFORMANCE_MODEL.value).strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(
            f"performance_model_input '{value}' is invalid. "
            f"Valid options: {[m.value for m in cls]}"
        )


@dataclass(frozen=True)
class PerformanceConfig:
    """Immutable, validated view of the performance configuration consumed by
    `PerformanceModel`. Has convenience accessors for emission-specific options.

    Attributes:
        missions_folder: Directory that holds the mission definition TOML files.
        missions_in_file: Name of the missions list to load from `missions_folder`.
        performance_model_input: Selected `PerformanceInputMode` (OPF vs table data).
        performance_model_input_file: Path (relative or absolute) to the
                                        performance input.
        emissions: Raw mapping of emission-related configuration used by helpers like
            `lto_input_mode`/`edb_input_file`.
    """

    missions_folder: str
    missions_in_file: str
    performance_model_input: PerformanceInputMode
    performance_model_input_file: str
    emissions: Mapping[str, Any]
    use_weather: bool
    weather_data_dir: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PerformanceConfig":
        missions = mapping.get('Missions', {})
        general = mapping.get('General Information', {})
        emissions = mapping.get('Emissions', {})
        weather = mapping.get('Weather', {})
        if not missions:
            raise ValueError("Missing [Missions] section in configuration file.")
        if not general:
            raise ValueError(
                "Missing [General Information] section in configuration file."
            )
        if not emissions:
            raise ValueError("Missing [Emissions] section in configuration file.")
        return cls(
            missions_folder=require_str(missions, 'missions_folder'),
            missions_in_file=require_str(missions, 'missions_in_file'),
            performance_model_input=PerformanceInputMode.from_value(
                general.get('performance_model_input')
            ),
            performance_model_input_file=require_str(
                general, 'performance_model_input_file'
            ),
            emissions=emissions,
            use_weather=bool(weather.get('use_weather', '') or False),
            weather_data_dir=str(weather.get('weather_data_dir', '') or ''),
        )

    def emission_option(self, key: str, default: Any = None) -> Any:
        return self.emissions.get(key, default)

    @property
    def lto_input_mode(self) -> str:
        raw = self.emission_option('LTO_input_mode', 'EDB')
        return str(raw or 'EDB')

    @property
    def lto_input_file(self) -> str | None:
        value = self.emission_option('LTO_input_file')
        return None if value in (None, '') else str(value)

    @property
    def edb_input_file(self) -> str:
        raw = self.emission_option('EDB_input_file')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("EDB_input_file must be provided in [Emissions].")
        return raw


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

    def __init__(self, config_file="IO/default_config.toml"):
        '''Initializes the performance model by reading the configuration,
        loading mission data, and setting up performance and engine models.'''
        config_file_loc = file_location(config_file)
        with open(config_file_loc, 'rb') as f:
            config_data = tomllib.load(f)
        self.config = PerformanceConfig.from_mapping(config_data)

        # Get mission data
        # self.filter_OAG_schedule = filter_OAG_schedule
        mission_file = file_location(
            os.path.join(self.config.missions_folder, self.config.missions_in_file)
        )
        with open(mission_file, 'rb') as f:
            self.missions = Mission.from_toml(tomllib.load(f))
        # self.schedule = filter_OAG_schedule()

        # Process input performance data
        self.initialize_performance()

    def initialize_performance(self):
        '''Initializes aircraft performance characteristics from TOML sourcee.
        Also loads LTO/EDB data and sets up the engine model using BADA3 parameters.'''

        self.ac_params = Bada3AircraftParameters()
        input_mode = self.config.performance_model_input
        # If OPF data input
        if input_mode is PerformanceInputMode.OPF:
            opf_params = parse_OPF(
                file_location(self.config.performance_model_input_file)
            )
            for key in opf_params:
                setattr(self.ac_params, key, opf_params[key])
        # If fuel flow function input
        elif input_mode is PerformanceInputMode.PERFORMANCE_MODEL:
            self.read_performance_data()
            ac_params_input = {
                "cas_cruise_lo": self.model_info["speeds"]['cruise']['cas_lo'],
                "cas_cruise_hi": self.model_info["speeds"]['cruise']['cas_hi'],
                "cas_cruise_mach": self.model_info["speeds"]['cruise']['mach'],
            }
            for key in ac_params_input:
                setattr(self.ac_params, key, ac_params_input[key])
        else:
            raise ValueError("Invalid performance model input provided!")

        # Initialize BADA engine model
        self.engine_model = Bada3JetEngineModel(self.ac_params)

        if self.config.lto_input_mode.strip().lower() == "input_file":
            # Load LTO data
            lto_input_file = self.config.lto_input_file
            if not lto_input_file:
                raise ValueError(
                    "LTO_input_file must be provided when"
                    "using LTO_input_mode='input_file'."
                )
            self.LTO_data = parseLTO(file_location(lto_input_file))

    def read_performance_data(self):
        '''Parses the TOML input file containing flight and LTO performance data.
        Separates model metadata and prepares the data for table generation.'''

        # Read and load TOML data
        with open(file_location(self.config.performance_model_input_file), "rb") as f:
            data = tomllib.load(f)

        self.LTO_data = data['LTO_performance']
        if self.config.lto_input_mode.strip().lower() == "edb":
            # Read UID
            UID = data['LTO_performance']['ICAO_UID']
            # Read EDB file and get engine
            engine_info = self.get_engine_by_uid(UID, self.config.edb_input_file)
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
