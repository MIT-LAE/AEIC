import json
import re
import numpy as np
import pandas as pd

import interpolate as interp
import parser as pars
import distance as dist





def get_fuel_consumption(climb_data, flight_level):
    """
    Returns the fuel consumption in kg/min for a given flight level (FL) at nominal conditions.
    """
    flight_level_int = int(flight_level)  # Ensure key lookup uses string type
    if flight_level_int in climb_data:
        return climb_data[flight_level_int]["Fuel_kg/min"]
    else:
        return "Flight level not found."
    
    
def get_flight_levels(climb_data):
    """
    Returns an array of flight levels from the parsed climb data.
    """
    return list(climb_data.keys())


def get_OAG_flight_data(csv_file_path, row_index):
    """
    Reads the CSV file and returns dep_lat, arr_lat, dep_elev_ft, and arr_elev_ft for a given row index.
    """
    df = pd.read_csv(csv_file_path)
    
    if 0 <= row_index < len(df):
        row = df.iloc[row_index]
        return {
            "dep_lat": row["dep_lat"],
            "dep_lon": row["dep_lon"],
            "arr_lat": row["arr_lat"],
            "arr_lon": row["arr_lon"],
            "dep_elev_ft": row["dep_elev_ft"],
            "arr_elev_ft": row["arr_elev_ft"]
        }
    else:
        return "Invalid row index."

# Define file path
file_path = "data/B738__.PTF"

OAG_file_path = "data/OAG_2024_SUBSET.csv"

# Run the function to parse the FL climb data
climb_data = pars.parse_climb_data(file_path)

# Get the list of available flight levels from the parsed climb data
flight_levels = get_flight_levels(climb_data)

fl_query = 100

# Example query for flight data
row_index_query = 9
flight_data = get_OAG_flight_data(OAG_file_path, row_index_query)
print(f"Flight data at row {row_index_query}: {flight_data}")


# Shortest distance in nm
distance = dist.get_great_circle_distance(flight_data)

# Departure & arrival airport elevation
elevation_dep_ft = flight_data['dep_elev_ft']
elevation_arr_ft = flight_data['arr_elev_ft']

print("Departure elevation:", elevation_dep_ft)
print("Arrival elevation:", elevation_arr_ft)


#fuel_consumption = get_fuel_consumption(climb_data, fl_query)
#print(f"Fuel consumption at FL {fl_query}: {fuel_consumption} kg/min")


####### INITIALIZE VARIABLES ########
elapsedTime      = 0.
fuelConsumed     = 0
NOxEmitted       = 0
distanceTraveled = 0
currentAltitude  = elevation_dep_ft + 3000  # Start NON-LTO analysis at airport elevation + 3000 ft AGL
currentLat       = flight_data['dep_lat']
currentLong      = flight_data['dep_lon']

endLat           = flight_data['arr_lat']
endLong          = flight_data['arr_lon']


# Assume constant load factor
loadFactor = 0.70

# Get aircraft operationl limits
# Get max altitude and max payload
alt_payload_data = pars.parse_max_alt_payload(file_path)

max_design_alt = alt_payload_data['Max Altitude [ft]']
max_design_payload = alt_payload_data['Max Payload [kg]']

# If current altitude exceeds aircraft operational limits, use airport altitude
if currentAltitude >= max_design_alt:
    # Setting to the current airport elevation
    currentAltitude = elevation_dep_ft
#end

# Setup paramters for climb
altStart = currentAltitude

# Set the altitude step to 1000 feet
altStep = 1000

# Set max altitude based on aircraft operating limits
altEnd = max_design_alt - 7000

####### DETERMINE STARTING MASS #########

# Define the flight level at cruise
FL = altEnd/100

# Parse the cruise data and find the closes flight level in PTF to the FL above
cruise_data = pars.parse_cruise_data(file_path)

# Convert FLs (dict keys) to a NumPy array
FLs_array = np.array(list(cruise_data.keys()), dtype=float)

# Find the index of the closest flight level
FL_index1 = np.argmin(np.abs(FLs_array - FL))

# Get the closest flight level
closest_FL = FLs_array[FL_index1]




    