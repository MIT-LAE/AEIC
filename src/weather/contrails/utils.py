import xarray as xr
import pandas as pd
import numpy as np


def inspect_grib_fields(grib_file_path):
    """
    Reads a GRIB file and prints a summary of variables, dimensions, and coordinates.

    Parameters:
    - grib_file_path: Path to the .grib or .grib2 file

    Returns:
    - None
    """
    print(f"Opening GRIB file: {grib_file_path}")
    
    try:
        ds = xr.open_dataset(grib_file_path, engine="cfgrib", decode_timedelta=True)
    except Exception as e:
        print("Failed to open GRIB file:", e)
        return

    print("\n--- Dataset Overview ---")
    print(ds)

    print("\n--- Variables ---")
    for var in ds.data_vars:
        print(f"{var}: {ds[var].dims} | shape: {ds[var].shape} | units: {ds[var].attrs.get('units', 'N/A')}")

    print("\n--- Coordinates ---")
    for coord in ds.coords:
        coord_values = ds[coord].values
        if coord_values.ndim == 0:
            print(f"{coord}: value = {coord_values}")
        else:
            preview = coord_values[:5] if coord_values.size > 5 else coord_values
            print(f"{coord}: values = {preview}{' ...' if coord_values.size > 5 else ''}")

    print("\n--- Global Attributes ---")
    for attr, val in ds.attrs.items():
        print(f"{attr}: {val}")

    ds.close()

def preprocess(grib_file_path):
    """
    Reads a GRIB file, converts longitudes to [-180, 180], and flattens the data to a DataFrame,
    computes vapor pressure, saturation vapor pressure over ice, and relatve humidity w.r.t ice (RHi)
    and returns the dataframe

    Parameters:
    - grib_file_path: Path to the .grib or .grib2 file

    Returns:
    - df: A pandas DataFrame with native + derived variables and coordinates flattened
    """
    try:
        ds = xr.open_dataset(grib_file_path, engine="cfgrib", decode_timedelta=True)
    except Exception as e:
        print("Failed to open GRIB file:", e)
        return None

    # Fix longitudes to range [-180, 180] if needed
    if 'longitude' in ds.coords:
        lon = ds['longitude']
        lon_fixed = ((lon + 180) % 360) - 180
        ds = ds.assign_coords(longitude=lon_fixed)

        # Sort by longitude to maintain monotonic increasing order
        ds = ds.sortby('longitude')

    # Flatten to dataframe
    df = ds.to_dataframe().reset_index()
    
    # Ensure required varibales are present for derived variables
    required_cols = ['q', 't', 'isobaricInhPa']
    if not all(col in df.columns for col in required_cols):
        print("Missing required columns: 'q', 't', or 'isobaricInhPa'")
        return df
    
    # Convert pressure from hPa to Pa
    df['p_Pa'] = df['isobaricInhPa'] * 100
    
    # Compute vapor pressure: e = (q * p) / (0.622 + 0.378 * q)
    df['vapor_pressure'] = (df['q'] * df['p_Pa']) / (0.622 + 0.378 * df['q'])
    
    # Compute saturation vapor pressure over ice using Murphy & Koop (2005)
    T = df['t']
    ln_esi = (9.550426 - (5723.265 / T) + 3.53068 * np.log(T) - 0.00728332 * T)
    df['saturation_esi'] = np.exp(ln_esi)
    
    # Compute RHi
    df['RHi'] = (df['vapor_pressure'] / df['saturation_esi']) * 100
    return df
