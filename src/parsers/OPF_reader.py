def parse_OPF(file_path):
    """
    Example: Parse OPF into distinct blocks for:
      - Aerodynamics
      - Engine Thrust
      - Fuel Consumption
      - Ground Movement
    
    Relies on the "CC===== <Section> =====" comment lines to find section boundaries.
    """
    
    extracted_data = {
        # Aircraft Type
        "n_eng": None,
        "engine_type": None,
        "wake_cat": None,
        
        # Mass
        "ref_mass": None,
        "min_mass": None,
        "max_mass": None,
        "max_payload": None,
        
        # Flight Envelope
        "V_MO": None,
        "M_MO": None,
        "H_MO": None,
        "h_max": None,
        "G_w": None,
        "G_t": None,
        
        # Aerodynamics
        "S_ref": None,
        "c_d0cr": None,
        "c_d2cr": None,
        "C_D0_AP": None,
        "C_D2_AP": None,
        "C_D0_LD": None,
        "C_D2_LD": None,
        "C_D0_ALDG": None,
        "C_M16": None,
        "V_stall_i": None,
        "C_Lbo_M0": None,
        "K": None,
        
        # Engine Thrust
        "c_tc1": None,
        "c_tc2": None,
        "c_tc3": None,
        "c_tc4": None,
        "c_tc5": None,
        "c_tdes_low": None,
        "c_tdes_high": None,
        "h_p_des": None,
        "c_tdes_app": None,
        "c_tdes_ld": None,
        "V_des_ref": None,
        "cas_cruise_mach": None,
        
        # Fuel Flow
        "c_f1": None,
        "c_f2": None,
        "c_f3": None,
        "c_f4": None,
        "c_fcr": None,
        
        # Ground Movement
        "TOL": None,
        "LDL": None,
        "span": None,
        "length": None
    }
    
    def to_float(s):
        return float(s.strip())
    
    # 1) Read in lines that start with "CD" (ignore "CC"), store them in a list
    raw_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("CD"):
                raw_lines.append(line)
    
    # 2) We also need to locate the "CC===== Section =====" lines
    #    so let's read the entire file again but keep them for splitting:
    lines_with_sections = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            lines_with_sections.append(line.rstrip())
    
    # Identify the index of each major section by searching lines_with_sections
    section_indices = {}
    for i, line in enumerate(lines_with_sections):
        if "Aerodynamics" in line:
            section_indices["aero"] = i
        elif "Engine Thrust" in line:
            section_indices["thrust"] = i
        elif "Fuel Consumption" in line:
            section_indices["fuel"] = i
        elif "Ground" in line:
            section_indices["ground"] = i
    
    # (A) Parse the first known lines (0..4 in raw_lines) for B744__, mass, envelope, wing:
    if len(raw_lines) > 1:
        tokens = raw_lines[1].split()  # "CD B744___ 4 engines Jet H"
        extracted_data["n_eng"] = int(tokens[2])   # 4
        extracted_data["engine_type"] = tokens[4]  # Jet
        wc = tokens[5].upper()  # "H"
        if wc == "H":
            extracted_data["wake_cat"] = "Heavy"
        elif wc == "M":
            extracted_data["wake_cat"] = "Medium"
        elif wc == "L":
            extracted_data["wake_cat"] = "Light"
        else:
            extracted_data["wake_cat"] = wc
    
    if len(raw_lines) > 2:
        parts = raw_lines[2].split()
        extracted_data["ref_mass"]  = to_float(parts[1])
        extracted_data["min_mass"]  = to_float(parts[2])
        extracted_data["max_mass"]  = to_float(parts[3])
        extracted_data["max_payload"] = to_float(parts[4])
        # If the file truly has 5 floats on this line, store the 5th in G_w
        if len(parts) > 5:
            extracted_data["G_w"] = to_float(parts[5])
    
    if len(raw_lines) > 3:
        parts = raw_lines[3].split()
        extracted_data["V_MO"]  = to_float(parts[1])
        extracted_data["M_MO"]  = to_float(parts[2])
        extracted_data["H_MO"]  = to_float(parts[3])
        extracted_data["h_max"] = to_float(parts[4])
        extracted_data["G_t"]   = to_float(parts[5])
    
    if len(raw_lines) > 4:
        parts = raw_lines[4].split()
        extracted_data["S_ref"]        = to_float(parts[2])
        extracted_data["C_Lbo_M0"] = to_float(parts[3])
        extracted_data["K"]        = to_float(parts[4])
        extracted_data["C_M16"]    = to_float(parts[5])
    
    # 3) Now we gather each block by looking at the lines_with_sections
    def get_section_lines(lines_list, start_key, end_key=None):
        """
        Return a list of lines (that start with "CD") between the section 
        named start_key and the section named end_key (exclusive).
        If end_key is None, we go to the end of the file.
        """
        start_idx = section_indices.get(start_key, None)
        end_idx   = section_indices.get(end_key, None)
        
        if start_idx is None:
            return []
        if end_idx is None:
            end_idx = len(lines_list)
        
        collected = []
        for i in range(start_idx+1, end_idx):
            line = lines_list[i]
            if line.startswith("CD"):
                collected.append(line)
        return collected
    
    aero_data   = get_section_lines(lines_with_sections, "aero", "thrust")
    thrust_data = get_section_lines(lines_with_sections, "thrust", "fuel")
    fuel_data   = get_section_lines(lines_with_sections, "fuel", "ground")
    ground_data = get_section_lines(lines_with_sections, "ground", None)
    
    #
    # (1) Aerodynamics block
    #
    for line in aero_data:
        parts = line.split()
        if "CR" in parts:
            extracted_data["V_stall_i"] = to_float(parts[4])
            extracted_data["c_d0cr"]   = to_float(parts[5])
            extracted_data["c_d2cr"]   = to_float(parts[6])
        elif "AP" in parts:
            extracted_data["C_D0_AP"] = to_float(parts[5])
            extracted_data["C_D2_AP"] = to_float(parts[6])
        elif "Lb" in parts:
            extracted_data["C_D0_LD"] = to_float(parts[5])
            extracted_data["C_D2_LD"] = to_float(parts[6])
        elif "DOWN" in parts:
            extracted_data["C_D0_ALDG"] = to_float(parts[3])
    
    #
    # (2) Engine Thrust block
    #
    # We expect 3 lines each with 5 floats. We parse them in order:
    #   1) c_tc1..c_tc5
    #   2) c_tdes_low..c_tdes_ld
    #   3) V_des_ref, cas_cruise_mach (+ 3 unused floats)
    #
    thrust_5floats = []
    for line in thrust_data:
        parts = line.split()
        # "CD" + 5 floats => len=6
        
        if len(parts) == 7:
            # confirm all floats except 'CD'
            try:
                float(parts[1]); float(parts[2]); float(parts[3]); float(parts[4]); float(parts[5])
                thrust_5floats.append(parts)
            except ValueError:
                pass

    if len(thrust_5floats) >= 1:
        # parse c_Tc_1..c_Tc_5
        p = thrust_5floats[0]
        extracted_data["c_tc1"] = to_float(p[1])
        extracted_data["c_tc2"] = to_float(p[2])
        extracted_data["c_tc3"] = to_float(p[3])
        extracted_data["c_tc4"] = to_float(p[4])
        extracted_data["c_tc5"] = to_float(p[5])
    
    if len(thrust_5floats) >= 2:
        # parse c_Tdes_low..c_Tdes_ld
        p = thrust_5floats[1]
        extracted_data["c_tdes_low"]  = to_float(p[1])
        extracted_data["c_tdes_high"] = to_float(p[2])
        extracted_data["h_p_des"]       = to_float(p[3])
        extracted_data["c_tdes_app"]  = to_float(p[4])
        extracted_data["c_tdes_ld"]   = to_float(p[5])
    
    if len(thrust_5floats) >= 3:
        # parse v_des_ref..m_des_ref (the first 2 floats after 'CD')
        # the remaining 3 floats are unused in your model
        p = thrust_5floats[2]
        extracted_data["V_des_ref"] = to_float(p[1])
        extracted_data["cas_cruise_mach"] = to_float(p[2])
    
    #
    # (3) Fuel Flow block
    #
    idx_fuel_2floats_1 = None
    idx_fuel_2floats_2 = None
    idx_fuel_5floats   = None
    
    for i, line in enumerate(fuel_data):
        parts = line.split()
        if len(parts) == 4:
            # "CD" + 2 floats
            if idx_fuel_2floats_1 is None:
                idx_fuel_2floats_1 = i
            elif idx_fuel_2floats_2 is None:
                idx_fuel_2floats_2 = i
        elif len(parts) == 7:
            # "CD" + 5 floats => c_fcr line
            idx_fuel_5floats = i
    
    if idx_fuel_2floats_1 is not None:
        p = fuel_data[idx_fuel_2floats_1].split()
        extracted_data["c_f1"] = to_float(p[1])
        extracted_data["c_f2"] = to_float(p[2])
    
    if idx_fuel_2floats_2 is not None:
        p = fuel_data[idx_fuel_2floats_2].split()
        extracted_data["c_f3"] = to_float(p[1])
        extracted_data["c_f4"] = to_float(p[2])
    
    if idx_fuel_5floats is not None:
        p = fuel_data[idx_fuel_5floats].split()
        extracted_data["c_fcr"] = to_float(p[1])
    
    #
    # (4) Ground block
    #
    for line in ground_data:
        parts = line.split()
        if len(parts) == 7:
            try:
                float(parts[1]); float(parts[2]); float(parts[3]); float(parts[4]); float(parts[5])
                extracted_data["TOL"]    = to_float(parts[1])
                extracted_data["LDL"]    = to_float(parts[2])
                extracted_data["span"]   = to_float(parts[3])
                extracted_data["length"] = to_float(parts[4])
            except ValueError:
                pass
    
    return extracted_data


if __name__ == "__main__":
    # Example usage:
    file_path = "/Users/aditeyashukla/Dropbox/Mac (2)/Documents/LAE/AEIC/src/IO/B744__.OPF"  # <-- Adjust to your actual file path
    data = parse_OPF(file_path)
    
    # Print each key/value
    for k, v in data.items():
        print(f"{k} = {v}")
