import json
import re
import pandas as pd

import interpolate as interp

def parse_climb_data(file_path):
    """
    Reads a TASOPT performance file and extracts Flight Level (FL) data
    for climb performance into a structured JSON format.
    """
    climb_data = {}
    capture = False

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Start capturing after detecting the FL table header
            if "FL |" in line:
                capture = True
                continue
            
            if capture:
                # Extract FL values and corresponding data
                parts = line.split("|")
                if len(parts) < 3:
                    continue  # Skip lines that don't have enough data
                
                # Extract FL
                try:
                    fl = int(parts[0].strip())
                    
                except ValueError:
                    continue  # Skip non-numeric FL lines
                
                # Extract Climb data from the second column
                climb_match = re.findall(r"\d+\.?\d*", parts[2])
                
                if len(climb_match) >= 5:
                    climb_data[fl] = {
                        "TAS_kts": int(climb_match[0]),
                        "ROCD_fpm": {
                            "Low": int(climb_match[1]),
                            "Nominal": int(climb_match[2]),
                            "High": int(climb_match[3])
                        },
                        "Fuel_kg/min": float(climb_match[4])
                    }
    
    #return json.dumps(climb_data, indent=4)
    return climb_data


def get_fuel_consumption(climb_data, flight_level):
    """
    Returns the fuel consumption in kg/min for a given flight level (FL) at nominal conditions.
    """
    flight_level_int = int(flight_level)  # Ensure key lookup uses string type
    if flight_level_int in climb_data:
        return climb_data[flight_level_int]["Fuel_kg/min"]
    else:
        return "Flight level not found."
    
    
# Updated function to extract Max Altitude and Max Payload correctly
def parse_max_alt_payload(file_path):
    """
    Reads an ASCII performance file and extracts Max Altitude [ft] and Max Payload [kg].
    """
    max_alt = None
    max_payload = None
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Search for Max Altitude
            match_alt = re.search(r"Max Alt\.\s*\[ft\]:\s*([\d,]+)", line)
            if match_alt:
                max_alt = int(match_alt.group(1).replace(",", ""))  # Remove commas if present
            
            # Search for Max Payload
            match_payload = re.search(r"Max Payload\s*\[kg\]:\s*([\d,]+)", line)
            if match_payload:
                max_payload = int(match_payload.group(1).replace(",", ""))  # Remove commas if present

    return {"Max Altitude [ft]": max_alt, "Max Payload [kg]": max_payload}

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
            "arr_lat": row["arr_lat"],
            "dep_elev_ft": row["dep_elev_ft"],
            "arr_elev_ft": row["arr_elev_ft"]
        }
    else:
        return "Invalid row index."

# Define file path
file_path = "data/B738__.PTF"

OAG_file_path = "data/OAG_2024_SUBSET.csv"

# Run the function to parse the FL climb data
climb_data = parse_climb_data(file_path)

# Get the list of available flight levels from the parsed climb data
flight_levels = get_flight_levels(climb_data)

fl_query = 100

# Example query for flight data
row_index_query = 9
flight_data = get_OAG_flight_data(OAG_file_path, row_index_query)
print(f"Flight data at row {row_index_query}: {flight_data}")


fuel_consumption = get_fuel_consumption(climb_data, fl_query)
print(f"Fuel consumption at FL {fl_query}: {fuel_consumption} kg/min")


