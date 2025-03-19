def interpolate_fuel_tas(climb_data, flight_level, mass_level):
    """
    Performs linear interpolation for fuel consumption and TAS at a non-listed flight level
    and mass level using bilinear interpolation.
    """
    sorted_fls = sorted(climb_data.keys())
    mass_levels = ["Low", "Nominal", "High"]
    
    if flight_level in climb_data and mass_level in climb_data[flight_level]["Fuel_kg/min"]:
        return {
            "TAS_kts": climb_data[flight_level]["TAS_kts"],
            "Fuel_kg/min": climb_data[flight_level]["Fuel_kg/min"][mass_level]
        }
    
    # Find bounding flight levels
    lower_fl = max([fl for fl in sorted_fls if fl < flight_level], default=None)
    upper_fl = min([fl for fl in sorted_fls if fl > flight_level], default=None)
    
    if lower_fl is None or upper_fl is None:
        return "Flight level out of interpolation range."
    
    # Find bounding mass levels
    if mass_level not in mass_levels:
        return "Mass level out of interpolation range."
    
    lower_mass = mass_levels[max(0, mass_levels.index(mass_level) - 1)]
    upper_mass = mass_levels[min(len(mass_levels) - 1, mass_levels.index(mass_level) + 1)]
    
    # Bilinear interpolation
    x1, x2 = lower_fl, upper_fl
    y1, y2 = lower_mass, upper_mass
    
    Q11 = climb_data[x1]["Fuel_kg/min"][y1]
    Q12 = climb_data[x1]["Fuel_kg/min"][y2]
    Q21 = climb_data[x2]["Fuel_kg/min"][y1]
    Q22 = climb_data[x2]["Fuel_kg/min"][y2]
    
    # Interpolating fuel consumption
    fuel_interp_mass1 = Q11 + (flight_level - x1) * (Q21 - Q11) / (x2 - x1)
    fuel_interp_mass2 = Q12 + (flight_level - x1) * (Q22 - Q12) / (x2 - x1)
    fuel_interp = fuel_interp_mass1 + (mass_levels.index(mass_level) - mass_levels.index(y1)) * (fuel_interp_mass2 - fuel_interp_mass1) / (mass_levels.index(y2) - mass_levels.index(y1))
    
    # Interpolating TAS
    tas_interp = climb_data[x1]["TAS_kts"] + (flight_level - x1) * (climb_data[x2]["TAS_kts"] - climb_data[x1]["TAS_kts"]) / (x2 - x1)
    
    return {"TAS_kts": tas_interp, "Fuel_kg/min": fuel_interp}