import re

def parseLTO(file_path):
    """
    Reads an aircraft LTO characteristics file and returns a dictionary with:
      - "Foo": the numeric value from the line "Foo = <value> [kN]"
      - "eng_data": a list of dictionaries corresponding to the comma-separated
         engine data after the marker line containing "AEIC ENG_EI output".
         
    Each engine data dictionary has the following keys:
      "ENG_NAME", "MODE", "CO_EI", "HC_EI", "NOX_EI",
      "SOX_EI", "SMOKE_NUM", "FUEL_KG/S", "ICAO_UID", "MODE_SN"
    """
    foo = None
    eng_data = {}
    capture_eng = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Look for the Foo value (only once)
            if foo is None:
                m = re.search(r"Foo\s*=\s*([\d\.]+)", line)
                if m:
                    foo = float(m.group(1))
                    continue

            # Check if this line marks the start of engine data
            if "AEIC ENG_EI output" in line:
                capture_eng = True
                continue

            # Once we have encountered the marker, process subsequent lines as engine data
            if capture_eng:
                # Split the line by commas and strip extra whitespace
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 10:
                    entry = {
                        "ENG_NAME": parts[0],
                        "CO_EI": parts[2],
                        "HC_EI": parts[3],
                        "NOX_EI": parts[4],
                        "SOX_EI": parts[5],
                        "SMOKE_NUM": parts[6],
                        "FUEL_KG/S": parts[7],
                        "ICAO_UID": parts[8],
                        "MODE_SN": parts[9]
                    }
                    eng_data[parts[1]] = entry
    return {"Foo": foo, "eng_LTO": eng_data}
