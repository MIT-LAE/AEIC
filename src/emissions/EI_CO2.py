def EI_CO2(fuel):
    """
    Calculate carbon-balanced CO2 emissions index (EI).

    Parameters
    ----------
    fuel : dictionary
        Fuel information (input from toml file)

    Returns
    -------
    CO2EI : ndarray
        CO2 emissions index [g/kg fuel], same shape as HCEI.
    CO2EInom : float
        Nominal CO2 emissions index (scalar).
    nvolCarbCont : float
        Non-volatile particulate carbon content fraction.
    """

    # if mcs == 1:
    #     CO2EInom = trirnd(3148, 3173, 3160, rv)
    #     nvolCarbCont = 0.9 + (0.98 - 0.9) * np.random.rand()
    # else:
    CO2EInom = fuel['EI_CO2']
    nvolCarbCont = fuel['nvolCarbCont']

    return CO2EInom, nvolCarbCont