def co2_eis_func(HCEI, COEI, OCicEI, PMnvolEI):
    """
    Calculate carbon-balanced CO2 emissions index (EI).

    Parameters
    ----------
    HCEI : ndarray
        Emissions index for hydrocarbons [g/kg fuel].
    COEI : ndarray
        Emissions index for CO [g/kg fuel].
    OCicEI : ndarray
        Emissions index for organic carbon [g/kg fuel].
    PMnvolEI : ndarray
        Emissions index for non-volatile particulate matter [g/kg fuel].

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
    CO2EInom = 3160.0
    nvolCarbCont = 0.95

    # Compute CO2EI with broadcast over array shapes
    CO2EI = (
        CO2EInom
        - (44/28)   * COEI
        - (44/(82/5)) * HCEI
        - (44/(55/4)) * OCicEI
        - (44/12)   * nvolCarbCont * PMnvolEI
    )

    return CO2EI, CO2EInom, nvolCarbCont