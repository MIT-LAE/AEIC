import numpy as np

def get_APU_emissions(APU_emission_indices, APU_emissions_g, 
                  LTO_emission_indices, APU_data, 
                  LTO_noProp, LTO_no2Prop, LTO_honoProp, apu_tim=2854):
    
    mask = (APU_data['fuel_kg_per_s'] != 0.0)

    apu_fuel_burn = APU_data['fuel_kg_per_s'] * apu_tim 

    # SOx
    APU_emission_indices['SO2'] = LTO_emission_indices['SO2'][0] if mask else 0.0
    APU_emission_indices['SO4'] = LTO_emission_indices['SO4'][0] if mask else 0.0

    # Particulate‐matter breakdown (deterministic BC fraction of 0.95)
    APU_PM10 = max(APU_data['PM10_g_per_kg'] - APU_emission_indices['SO4'], 0.0)
    bc_prop = 0.95
    APU_emission_indices['PMnvol'] = np.array(APU_PM10 * bc_prop).item()
    APU_emission_indices['PMvol'] = np.array(APU_PM10 - APU_emission_indices['PMnvol']).item()

    # NO/NO2/HONO speciation
    APU_emission_indices['NO']   = APU_data['PM10_g_per_kg'] * LTO_noProp[0]
    APU_emission_indices['NO2']  = APU_data['PM10_g_per_kg'] * LTO_no2Prop[0]
    APU_emission_indices['HONO'] = APU_data['PM10_g_per_kg'] * LTO_honoProp[0]

    APU_emission_indices['NOx'] = APU_data['NOx_g_per_kg']
    APU_emission_indices['HC'] = APU_data['HC_g_per_kg']
    APU_emission_indices['CO'] = APU_data['CO_g_per_kg']

    # CO2 via mass balance
    if mask:
        co2_ei_nom = 3160
        nvol_carb_cont = 0.95

        co2 = co2_ei_nom
        co2 -= (44/28)     * APU_emission_indices['CO']
        co2 -= (44/(82/5)) * APU_emission_indices['HC']
        co2 -= (44/(55/4)) * APU_emission_indices['PMvol']
        co2 -= (44/12)     * nvol_carb_cont * APU_emission_indices['PMnvol']
        APU_emission_indices['CO2'] = co2
    else:
        APU_emission_indices['CO2'] = 0.0

    for field in APU_emission_indices.dtype.names:
        APU_emissions_g[field] = APU_emission_indices[field] * apu_fuel_burn

    return APU_emission_indices, APU_emissions_g
