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
                
                print(parts[2])
                
                # Extract Climb data from the second column
                climb_match = re.findall(r"\d+\.?\d*", parts[1])
                
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
    
    return json.dumps(climb_data, indent=4)

# Define file path
file_path = "data/B738__.PTF"

# Run the function to parse the FL climb data
json_climb_data = parse_climb_data(file_path)
#print(json_climb_data)
