def lifecycle_CO2(fuel, fuel_burn):
    """
    Calculate lifecycle CO2 emissions.

    Parameters
    ----------
    fuel : dictionary
        Fuel information (input from toml file)

    Returns
    -------
    lifecycle_CO2 : float
    """
    return fuel_burn * (fuel['LC_CO2'] - fuel['EI_CO2'])
