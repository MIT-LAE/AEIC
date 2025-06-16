import numpy as np
import xarray as xr

import numpy as np
import xarray as xr

def compute_vapor_pressure_and_rhi(ds):
    """
    Compute vapor pressure, saturation vapor pressure over ice (Murphy & Koop 2005),
    and RHi directly on a 4D xarray Dataset. Also appends ISSR_flag.

    Parameters:
        ds (xarray.Dataset): Must contain 't' (K), 'q' (kg/kg), and 'pressure_level' (hPa)

    Returns:
        xarray.Dataset: with new variables: vapor_pressure, saturation_esi, RHi, ISSR_flag
    """
    # Ensure longitudes are in [-180, 180]
    if 'longitude' in ds.coords:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby('longitude')

    # Convert pressure_level from hPa to Pa and broadcast
    p_Pa = ds['pressure_level'] * 100.0  # shape: (pressure_level,)
    p = p_Pa.broadcast_like(ds['q'])     # shape: (time, pressure_level, lat, lon)

    # Compute vapor pressure: e = (q * p) / (0.622 + 0.378 * q)
    q = ds['q']
    T = ds['t']
    vapor_pressure = (q * p) / (0.622 + 0.378 * q)
    vapor_pressure.name = "vapor_pressure"
    vapor_pressure.attrs = {
        "units": "Pa",
        "description": "Vapor pressure derived from specific humidity and pressure"
    }

    # Compute saturation vapor pressure over ice (Murphy & Koop 2005)
    ln_esi = 9.550426 - (5723.265 / T) + 3.53068 * np.log(T) - 0.00728332 * T
    saturation_esi = np.exp(ln_esi)
    saturation_esi.name = "saturation_esi"
    saturation_esi.attrs = {
        "units": "Pa",
        "description": "Saturation vapor pressure over ice (Murphy & Koop 2005)"
    }

    # Compute RHi
    RHi = (vapor_pressure / saturation_esi) * 100.0
    RHi.name = "RHi"
    RHi.attrs = {
        "units": "%",
        "description": "Relative Humidity with respect to ice (RHi > 100% → ISSR)"
    }

    # Compute ISSR flag
    ISSR_flag = (RHi > 100).astype(np.int8)
    ISSR_flag.name = "ISSR_flag"
    ISSR_flag.attrs = {
        "description": "Ice Supersaturated Region Flag (1 if RHi > 100%)"
    }

    # Add computed fields to dataset
    ds['vapor_pressure'] = vapor_pressure
    ds['saturation_esi'] = saturation_esi
    ds['RHi'] = RHi
    ds['ISSR_flag'] = ISSR_flag

    return ds

def print_pressure_levels_with_altitudes(ds):
    """
    Print pressure levels and corresponding altitudes (in feet) in a formatted table.

    Parameters:
        ds (xarray.Dataset): Dataset containing the 'pressure_level' coordinate in hPa.
    """
    import numpy as np

    if 'pressure_level' not in ds.coords:
        raise ValueError("The dataset does not contain a 'pressure_level' coordinate.")

    pressure_hPa = ds['pressure_level'].values
    pressure_Pa = pressure_hPa * 100.0

    # Convert pressure to altitude (meters) using barometric formula
    # Valid roughly for the troposphere (standard atmosphere)
    altitude_m = 44330.0 * (1.0 - (pressure_Pa / 101325.0)**(1 / 5.255))
    altitude_ft = altitude_m * 3.28084

    print(f"{'Level':>5} | {'Pressure (hPa)':>15} | {'Altitude (ft)':>15}")
    print("-" * 43)
    for i, (p, alt) in enumerate(zip(pressure_hPa, altitude_ft)):
        print(f"{i+1:5d} | {p:15.1f} | {alt:15.0f}")



file_path = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/RAW/20241201.nc"

ds = xr.open_dataset(file_path)

print_pressure_levels_with_altitudes(ds)

#print(ds)

ds_RHI = compute_vapor_pressure_and_rhi(ds)

#print(ds_RHI)