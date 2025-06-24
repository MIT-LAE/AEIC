import re

def parse_PTF(file_path):
    """
    Reads the TASOPT performance file in a single pass and returns a dict:
      {
        "phases": {
          "climb": {
            "flight_levels_ft": [...],
            "tas": [...],
            "rocd_lo": [...],
            "rocd_hi": [...],
            "fuel_flow_lo": [...],
            "fuel_flow_hi": [...],
            "cas_lo": <int>,
            "cas_hi": <int>,
            "mach": <float>
          },
          "cruise": { ... },
          "descent": { ... }
        },
        "max_alt_ft": ...,
        "max_payload_kg": ...,
        "low_mass_kg": ...,
        "nominal_mass_kg": ...,
        "high_mass_kg": ...
      }
    """
    
    # Prepare data structures for phases with new keys for speeds
    climb_data = {
        "flight_levels_ft": [],
        "tas": [],
        "rocd_lo": [],
        "rocd_hi": [],
        "fuel_flow_lo": [],
        "fuel_flow_hi": [],
        "cas_lo": None,
        "cas_hi": None,
        "mach": None,
    }
    cruise_data = {
        "flight_levels_ft": [],
        "tas": [],
        "rocd_lo": [],
        "rocd_hi": [],
        "fuel_flow_lo": [],
        "fuel_flow_hi": [],
        "cas_lo": None,
        "cas_hi": None,
        "mach": None,
    }
    descent_data = {
        "flight_levels_ft": [],
        "tas": [],
        "rocd_lo": [],
        "rocd_hi": [],
        "fuel_flow_lo": [],
        "fuel_flow_hi": [],
        "cas_lo": None,
        "cas_hi": None,
        "mach": None,
    }
    
    # Prepare storage for top-level info
    max_alt = None
    max_payload = None
    low_mass = None
    nominal_mass = None
    high_mass = None

    capture = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1) Parse top-level lines before/after the table for alt/payload/mass and speeds:
            if not capture:
                # Max altitude
                match_alt = re.search(r"Max Alt\.\s*\[ft\]:\s*([\d,]+)", line)
                if match_alt:
                    max_alt = int(match_alt.group(1).replace(",", ""))
                
                # Max payload
                match_payload = re.search(r"Max Payload\s*\[kg\]:\s*([\d,]+)", line)
                if match_payload:
                    max_payload = int(match_payload.group(1).replace(",", ""))

                # Mass levels
                # low mass
                if 'low' in line and 'climb' in line:
                    match = re.search(r'low\s*-\s*(\d+)', line)
                    if match:
                        low_mass = int(match.group(1))
                # nominal mass
                elif 'nominal' in line and 'cruise' in line:
                    match = re.search(r'nominal\s*-\s*(\d+)', line)
                    if match:
                        nominal_mass = int(match.group(1))
                # high mass
                elif 'high' in line and 'descent' in line:
                    match = re.search(r'high\s*-\s*(\d+)', line)
                    if match:
                        high_mass = int(match.group(1))

                # Parse speeds for each phase if the line starts with climb, cruise, or descent
                stripped = line.lstrip()
                if stripped.startswith("climb"):
                    tokens = stripped.split()
                    if len(tokens) >= 4:
                        try:
                            cas = tokens[2]       # Expecting a format like "250/300"
                            mach = tokens[3]      # e.g., "0.80"
                            cas_lo, cas_hi = cas.split("/")
                            climb_data["cas_lo"] = int(cas_lo)
                            climb_data["cas_hi"] = int(cas_hi)
                            climb_data["mach"] = float(mach)
                        except Exception:
                            pass
                elif stripped.startswith("cruise"):
                    tokens = stripped.split()
                    if len(tokens) >= 4:
                        try:
                            cas = tokens[2]
                            mach = tokens[3]
                            cas_lo, cas_hi = cas.split("/")
                            cruise_data["cas_lo"] = int(cas_lo)
                            cruise_data["cas_hi"] = int(cas_hi)
                            cruise_data["mach"] = float(mach)
                        except Exception:
                            pass
                elif stripped.startswith("descent"):
                    tokens = stripped.split()
                    if len(tokens) >= 4:
                        try:
                            cas = tokens[2]
                            mach = tokens[3]
                            cas_lo, cas_hi = cas.split("/")
                            descent_data["cas_lo"] = int(cas_lo)
                            descent_data["cas_hi"] = int(cas_hi)
                            descent_data["mach"] = float(mach)
                        except Exception:
                            pass

            # 2) Detect the start of the flight-level table
            if "FL |" in line:
                capture = True
                continue

            # 3) Once capturing, parse columns for FL, climb, cruise, descent
            if capture:
                parts = line.split("|")
                # We expect 4 columns: FL, CRUISE, CLIMB, DESCENT
                if len(parts) < 4:
                    continue
                
                # Flight level in parts[0]
                try:
                    fl = int(parts[0].strip())
                except ValueError:
                    # If we can't parse a valid FL, skip
                    continue

                # ============ CRUISE ============ 
                cruise_str = parts[1]
                # Typically we expect [TAS, fuel_low, fuel_nom, fuel_high]
                c_vals = re.findall(r"\d+\.?\d*", cruise_str)
                if len(c_vals) >= 4:
                    # TAS
                    c_tas = float(c_vals[0])
                    # Fuel flows ignoring nominal => Low and High
                    c_fuel_lo = float(c_vals[1])
                    c_fuel_hi = float(c_vals[3])
                else:
                    # If line is blank for cruise, skip
                    c_tas = None

                # ============ CLIMB ============ 
                climb_str = parts[2]
                # Typically we expect [TAS, rocd_lo, rocd_nom, rocd_hi, fuel_nom]
                cl_vals = re.findall(r"\d+\.?\d*", climb_str)
                if len(cl_vals) >= 5:
                    # TAS
                    cl_tas = float(cl_vals[0])
                    # ROCD
                    cl_rocd_lo = float(cl_vals[1])
                    cl_rocd_hi = float(cl_vals[3])    # ignoring nominal
                    # Fuel: replicate nominal in lo & hi
                    cl_fuel_nom = float(cl_vals[4])
                else:
                    # If line is blank for climb, skip
                    cl_tas = None

                # ============ DESCENT ============ 
                descent_str = parts[3]
                # Typically we expect [TAS_nom, rocd_nom, fuel_nom]
                d_vals = re.findall(r"\d+\.?\d*", descent_str)
                if len(d_vals) >= 3:
                    d_tas_nom = float(d_vals[0])
                    # TODO: For now just putting rocd for descent as negative
                    d_rocd_nom = -float(d_vals[1])
                    d_fuel_nom = float(d_vals[2])
                else:
                    d_tas_nom = None

                # Now store them if they exist
                # -- Cruise
                if c_tas is not None:
                    cruise_data["flight_levels_ft"].append(fl)
                    cruise_data["tas"].append(c_tas)
                    # Cruise ROCD is always 0
                    cruise_data["rocd_lo"].append(0.0)
                    cruise_data["rocd_hi"].append(0.0)
                    cruise_data["fuel_flow_lo"].append(c_fuel_lo)
                    cruise_data["fuel_flow_hi"].append(c_fuel_hi)

                # -- Climb
                if cl_tas is not None:
                    climb_data["flight_levels_ft"].append(fl)
                    climb_data["tas"].append(cl_tas)
                    climb_data["rocd_lo"].append(cl_rocd_lo)
                    climb_data["rocd_hi"].append(cl_rocd_hi)
                    climb_data["fuel_flow_lo"].append(cl_fuel_nom)
                    climb_data["fuel_flow_hi"].append(cl_fuel_nom)

                # -- Descent
                if d_tas_nom is not None:
                    descent_data["flight_levels_ft"].append(fl)
                    descent_data["tas"].append(d_tas_nom)
                    # Replicate nominal ROCD in both lo & hi
                    descent_data["rocd_lo"].append(d_rocd_nom)
                    descent_data["rocd_hi"].append(d_rocd_nom)
                    descent_data["fuel_flow_lo"].append(d_fuel_nom)
                    descent_data["fuel_flow_hi"].append(d_fuel_nom)

    # Build final data structure
    data = {
        "phases": {
            "climb": climb_data,
            "cruise": cruise_data,
            "descent": descent_data
        },
        # Top-level info
        "max_alt_ft": max_alt,
        "max_payload_kg": max_payload,
        "low_mass_kg": low_mass,
        "nominal_mass_kg": nominal_mass,
        "high_mass_kg": high_mass
    }
    return data
