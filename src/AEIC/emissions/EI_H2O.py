def EI_H2O(fuel):
    """
    Calculate H2O emissions index (EI).

    Parameters
    ----------
    fuel : dictionary
        Fuel information (input from toml file)

    Returns
    -------
    H2O_EI : float
        H2O emissions index [g/kg fuel]
    """

    return fuel['EI_H2O']
