import json
import re
import numpy as np
import pandas as pd

import interpolate as interp
import parser as pars
import distance as dist
import trajectory as traj





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

print(cruise_data)

# Convert FLs (dict keys) to a NumPy array
Cruise_FLs = np.array(list(cruise_data.keys()), dtype=float)

# Find the index of the cruise FL closest to the aircraft FL
FL_index1 = np.argmin(np.abs(Cruise_FLs - FL))


print("FL index closest to aircraft FL:", FL_index1)

# Get the closest flight level
closest_FL = Cruise_FLs[FL_index1]

print("closest FL: ", closest_FL)

print("First crusie index:", Cruise_FLs[0])
print("Second crusie index:", Cruise_FLs[1])

# See if the aircraft FL is either greater or lesser than than th first
# PTF cruise FL and select the next flight level accordingly
if FL > FL_index1:
    FL_index2 = FL_index1 + 1
elif FL < FL_index1:
    FL_index2 = FL_index1 - 1
elif FL == FL_index1:
    FL_index2 = FL_index1
else:
    raise ValueError("Could not set FL index from PTF")

# Interpoate for cruise fuel flow:
    # Get the Flight levels based on the FL_indices
FL_1 = Cruise_FLs[FL_index1]
FL_2 = Cruise_FLs[FL_index2]

print("First Flight level: ", FL_1)
print("Second flight level: ", FL_2)


# Extract the fuel flow at these flight levels
FF_1 = cruise_data[FL_1]['Fuel_flow_kgm']['Nominal']
FF_2 = cruise_data[FL_2]['Fuel_flow_kgm']['Nominal']

# Extract the TAS at these flight levels
TAS_KTS_1 = cruise_data[FL_1]['TAS_kts']
TAS_KTS_2 = cruise_data[FL_2]['TAS_kts']

# Skip NOX data for now.
if FL_1 == FL_2:
    fuelFlowRate = FF_1
    TAS_kts = TAS_KTS_1
else:
    fuelFlowRate = FF_1 + (FF_2 - FF_1)/ (FL_2 - FL_1) * (FL - FL_1)
    TAS_kts = TAS_KTS_1 + (TAS_KTS_2 - TAS_KTS_1)/ (FL_2 - FL_1) * (FL- FL_1)
    
# Guess the starting weight

# Setting a duy weight for now
emptyWeight = 39.5*1000

payloadWeight = max_design_payload * loadFactor

# Use TOC TAS_kts and straign line distance to get flight duration in mins
approxTimeFlight_min = distance/TAS_kts * 60

# USE TOC fuel flow rate and flight time to get fuelWeightMission
# NOTE: This assumption will overestimate fuel weight.
fuelWeightMission = fuelFlowRate * approxTimeFlight_min

# Add 5 % for reserves
fuelWeightReserves = fuelWeightMission * 0.05


# Determine Diversio Fuel Weight and hold fuel
if approxTimeFlight_min > 180:
    # Long haul: 200 nm diversion + 30 min low alt hold
    fuelWeight_Divert = 200/TAS_kts * 60 * fuelFlowRate
    
    # Time at holding pattern
    HOLD_min = 30
    
    # Holding pattern flight level
    FL_HOLD = Cruise_FLs[0]
    
    # At holding pattern, use the fuel flow at "lo" mass fraction at the first cruise FL from PTF
    fuelWeight_Hold = HOLD_min * cruise_data[FL_HOLD]['Fuel_flow_kgm']['Low']

else:
    # Short haul: 100 nm diversion + 45 min low alt hold
    fuelWeight_Divert = 100/TAS_kts * 60 * fuelFlowRate
    
    # Time at holding pattern
    HOLD_min = 45
    
    # Holding pattern flight level
    FL_HOLD = Cruise_FLs[0]
    
    # At holding pattern, use the fuel flow at "lo" mass fraction at the first cruise FL from PTF
    fuelWeight_Hold = HOLD_min * cruise_data[FL_HOLD]['Fuel_flow_kgm']['Low']
    

# Set starting mass
rv_TOW = 1  # TODO: Change later

startingMass = emptyWeight + payloadWeight + fuelWeightReserves + fuelWeight_Divert * fuelWeight_Hold
startingMass = startingMass * rv_TOW    


# Get the maximum allowable mass from PTF
mass_levels = pars.parse_mass_levels(file_path)
max_mass_kg = mass_levels["High Mass"]

# If starting mass exceeds the max weight in the PTF, clip to max weight
if startingMass > max_mass_kg:
    print("Starting mass exceeds maximum design weight")
    print("Flight distance (nm): ", distance)
    print("Starting mass: ", startingMass)
    print("Maximum allowable mass: ", max_mass_kg)
    
    print("Aircraft too heavy for the mission --> setting starting mass to max mass")
    startingMass = max_mass_kg
    #end
    


######----INITILIZATON COMPLETE----######

#Set current mass to starting mass

currentMass = startingMass


##########################################################
###########---------CLIMB PHASE-------------##############
##########################################################

currentAz = 0

traj.alt_change(altStart, altEnd, altStep, currentMass, fuelConsumed, distanceTraveled, elapsedTime,currentLat, currentLong, currentAz, endLat, endLong)


    