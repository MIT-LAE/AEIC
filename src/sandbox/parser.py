import json
import re

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


def parse_cruise_data(file_path):
    """
    Reads a TASOPT performance file and extracts Flight Level (FL) data
    for cruise performance into a structured JSON format.
    """
    cruise_data = {}
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
                cruise_match = re.findall(r"\d+\.?\d*", parts[1])
                if len(cruise_match) >= 4:
                    cruise_data[fl] = {
                        "TAS_kts": int(cruise_match[0]),
                        "Fuel_flow_kgm": {
                            "Low": float(cruise_match[1]),
                            "Nominal": float(cruise_match[2]),
                            "High": float(cruise_match[3])
                        }
                    }
    
    #return json.dumps(climb_data, indent=4)
    return cruise_data



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