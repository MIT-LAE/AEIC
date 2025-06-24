import os
from .aircraft_parameters import Bada3AircraftParameters
from .model import Bada3FuelBurnModel
from typing import Dict

def read_synonym_file_to_dict(folder_path) -> Dict[str, str]:
    """
    Returns a dictionary of synonyms, where the key is the aircraft type and the value is the corresponding aircraft type that has a direct OPF file available.

    Parameters:
    -----------
    folder_path: str
        Path to the folder containing the BADA3 files

    Returns:
    --------
    dict
        Dictionary of synonyms, where the key is the aircraft type and the value is the corresponding aircraft type that has a direct OPF file available.
    """

    file_path = f"{folder_path}/SYNONYM.NEW"

    with open(file_path, "r", encoding="ISO-8859-1") as f:
        lines = f.readlines()

    # lines 17 to -2 define synonyms
    synonym_lines = [line.split() for line in lines[17:-2]]

    # create a dictionary of synonyms
    synonym_dict = {}
    for line in synonym_lines:
        synonym_dict[line[2]] = line[-3].strip("_")
    return synonym_dict

def get_directly_available_aircraft_types(folder_path):
    """
    Returns a list of all directly available aircraft types in the BADA3 database.

    Parameters:
    -----------
    folder_path: str
        Path to the folder containing the BADA3 files

    Returns:
    --------
    list
        List of all directly available aircraft types in the BADA3 database.
    """
    directly_available = []

    for file in os.listdir(folder_path):
        if file.endswith(".OPF"):
            directly_available.append(file.split(".")[0].strip("_"))

    return directly_available


def get_all_available_aircraft_types(folder_path):
    """
    Returns a list of all available aircraft types in the BADA3 database.

    Parameters:
    -----------
    folder_path: str
        Path to the folder containing the BADA3 files

    Returns:
    --------
    list
        List of all available aircraft types in the BADA3 database.
    """

    # can either be directly available (the .opf file has the aircraft type in it) or available via synonym, synonyms are defined in SYNONYM.NEW

    directly_available = get_directly_available_aircraft_types(folder_path)

    synonym_dict = read_synonym_file_to_dict(folder_path)
    available_via_synonym = list(synonym_dict.keys())

    return set(directly_available + available_via_synonym)


def get_aircraft_params_for_all_aircraft_types(
    folder_path,
) -> Dict[str, Bada3AircraftParameters]:
    """
    Returns a dictionary containing the Bada3AircraftParameters for all aircraft types.

    Parameters:
    -----------
    folder_path: str
        Path to the folder containing the BADA3 files

    Returns:
    --------
    dict
        Dictionary containing the Bada3AircraftParameters for all aircraft types.
    """
    aircraft_params_dict = {}

    directly_available_types = get_directly_available_aircraft_types(folder_path)

    for aircraft_type in directly_available_types:
        ap = Bada3AircraftParameters()

        # aircraft type needs to be right filled to 6 characters by _
        ap.assign_opf_parameters_fromfile(
            os.path.join(folder_path, f"{aircraft_type.ljust(6, '_')}.OPF")
        )
        # ap.assign_ptf_parameters_fromfile(
        #     os.path.join(folder_path, f"{aircraft_type.ljust(6, '_')}.PTF")
        # )

        aircraft_params_dict[aircraft_type] = ap

    synonym_dict = read_synonym_file_to_dict(folder_path)

    for aircraft_type in synonym_dict.keys():
        aircraft_params_dict[aircraft_type] = aircraft_params_dict[
            synonym_dict[aircraft_type]
        ]

    return aircraft_params_dict


def get_models_for_all_implemented_aircraft_types(
    folder_path,
) -> Dict[str, Bada3FuelBurnModel]:
    """
    Returns a dictionary containing the Bada3FuelBurnModel for all aircraft types.

    Parameters:
    -----------
    folder_path: str
        Path to the folder containing the BADA3 files

    Returns:
    --------
    dict
        Dictionary containing the Bada3FuelBurnModel for all aircraft types.
    """
    aircraft_params_dict = get_aircraft_params_for_all_aircraft_types(folder_path)

    model_dict = {}

    for aircraft_type in aircraft_params_dict.keys():
        ap = aircraft_params_dict[aircraft_type]
        if ap.engine_type not in ["Jet", "Turboprop", "Piston"]:
            continue
        model_dict[aircraft_type] = Bada3FuelBurnModel(
            aircraft_params_dict[aircraft_type]
        )

    return model_dict
