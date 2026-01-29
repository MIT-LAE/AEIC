from AEIC.types import Fuel


def lifecycle_CO2(fuel: Fuel, fuel_burn):
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
    if fuel.lifecycle_CO2 is None:
        raise ValueError('Lifecycle CO2 data not available for this fuel.')
    return fuel_burn * (fuel.lifecycle_CO2 * fuel.energy_MJ_per_kg - fuel.EI_CO2)
