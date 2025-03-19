import json
import re







# Extracting the full table below the "FL" header
fl_table = []
capture = False


def parse_flight_level_data(fl_table):
    """
    Parses the extracted FL table into a structured JSON format.
    """
    fl_data = {}

    for line in fl_table:
        # Skip separators or empty lines
        if "===" in line or "|" not in line.strip():
            continue

        # Extract FL values and corresponding data
        parts = line.split("|")
        if len(parts) < 3:
            continue  # Skip lines that don't have enough data

        # Extract FL
        try:
            fl = int(parts[0].strip())
        except ValueError:
            continue  # Skip non-numeric FL lines

        # Extract Climb data
        climb_match = re.findall(r"[\d]+", parts[1])
        if len(climb_match) >= 5:
            climb_data = {
                "TAS_kts": int(climb_match[0]),
                "ROCD_fpm": [int(climb_match[1]), int(climb_match[2]), int(climb_match[3])],
                "Fuel_kg/min": float(climb_match[4])
            }
        else:
            climb_data = None

        # Extract Cruise data
        cruise_match = re.findall(r"[\d.]+", parts[0])
        if len(cruise_match) >= 4:
            cruise_data = {
                "TAS_kts": int(cruise_match[0]),
                "Fuel_kg/min": [float(cruise_match[1]), float(cruise_match[2]), float(cruise_match[3])]
            }
        else:
            cruise_data = None

        # Extract Descent data
        descent_match = re.findall(r"[\d]+", parts[2])
        if len(descent_match) >= 3:
            descent_data = {
                "TAS_kts": int(descent_match[0]),
                "ROCD_fpm": int(descent_match[1]),
                "Fuel_kg/min": float(descent_match[2])
            }
        else:
            descent_data = None

        # Store the data
        fl_data[fl] = {
            "Cruise": cruise_data,
            "Climb": climb_data,
            "Descent": descent_data
        }

    return json.dumps(fl_data, indent=4)

file_path = "data/B738__.PTF"

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        if "FL |" in line:
            capture = True  # Start capturing data after finding the FL header
            continue  # Skip the header itself

        if capture:
            if line.strip():  # Ensure it's not an empty line
                fl_table.append(line.strip())
            else:
                break  # Stop capturing when an empty line is encountered
            

# Run the function to parse the FL data
json_fl_data = parse_flight_level_data(fl_table)

print(json_fl_data)

