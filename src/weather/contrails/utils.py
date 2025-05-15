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
    
    # Compute ISSR flag: 1 if RHi > 100, else 0
    df['ISSR_flag'] = (df['RHi'] > 100).astype(int)
    
    return df


import xarray as xr
import pandas as pd
import numpy as np

def preprocess_by_levels(grib_file_path, levels_to_extract):
    """
    Processes GRIB file one pressure level at a time to reduce memory usage.
    
    Parameters:
    - grib_file_path: path to .grib or .grib2 file
    - levels_to_extract: list of pressure levels (in hPa) to process

    Returns:
    - df_all: pandas DataFrame with ISSR_flag and derived variables across selected levels
    """
    dfs = []

    for level in levels_to_extract:
        try:
            ds = xr.open_dataset(
                grib_file_path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": level}}
            )
        except Exception as e:
            print(f"Failed to load level {level} hPa:", e)
            continue

        # Fix longitude
        if 'longitude' in ds.coords:
            lon_fixed = ((ds['longitude'] + 180) % 360) - 180
            ds = ds.assign_coords(longitude=lon_fixed)
            ds = ds.sortby('longitude')

        # Ensure required variables are present
        required_vars = ['q', 't']
        if not all(var in ds.variables for var in required_vars):
            print(f"Missing required variables at level {level}")
            continue

        # Convert to dataframe and add level
        df = ds[required_vars].to_dataframe().reset_index()
        df['isobaricInhPa'] = level
        df['p_Pa'] = df['isobaricInhPa'] * 100

        # Derived variables
        df['vapor_pressure'] = (df['q'] * df['p_Pa']) / (0.622 + 0.378 * df['q'])
        T = df['t']
        ln_esi = (9.550426 - (5723.265 / T) + 3.53068 * np.log(T) - 0.00728332 * T)
        df['saturation_esi'] = np.exp(ln_esi)
        df['RHi'] = (df['vapor_pressure'] / df['saturation_esi']) * 100
        df['ISSR_flag'] = (df['RHi'] > 100).astype(np.uint8)

        dfs.append(df)

    # Combine all levels
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        return df_all
    else:
        print("No valid levels processed.")
        return None

def preprocess_by_time_slice(grib_file_path):
    """
    Processes each time slice individually to reduce memory usage.
    
    Parameters:
    - grib_file_path: path to the GRIB file
    
    Returns:
    - combined_df: pandas DataFrame with ISSR flag and other derived fields
    """
    try:
        ds = xr.open_dataset(grib_file_path, engine="cfgrib")
    except Exception as e:
        print("Failed to open GRIB file:", e)
        return None

    # Fix longitude
    if 'longitude' in ds.coords:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby('longitude')

    required_vars = ['q', 't']
    if not all(var in ds.variables for var in required_vars):
        print("Missing required variables:", [v for v in required_vars if v not in ds.variables])
        return None

    dfs = []

    for i in range(ds.sizes['time']):
        print(f"Processing time index {i}")
        sub_ds = ds.isel(time=i)[required_vars]

        df = sub_ds.to_dataframe().reset_index()
        df['p_Pa'] = df['isobaricInhPa'] * 100

        df['vapor_pressure'] = (df['q'] * df['p_Pa']) / (0.622 + 0.378 * df['q'])
        T = df['t']
        ln_esi = (9.550426 - (5723.265 / T) + 3.53068 * np.log(T) - 0.00728332 * T)
        df['saturation_esi'] = np.exp(ln_esi)

        df['RHi'] = (df['vapor_pressure'] / df['saturation_esi']) * 100
        df['ISSR_flag'] = (df['RHi'] > 100).astype(np.uint8)

        df['valid_time'] = ds['time'].values[i]

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df



def pressure_to_altitude_ft(pressure_hPa):
    """
    Converts pressure level (in hPa) to altitude (in feet) using the standard atmosphere.

    Parameters:
    - pressure_hPa: Scalar or array-like pressure values in hPa

    Returns:
    - altitude_ft: Altitude in feet (same shape as input)
    """
    # Convert pressure to altitude in meters using the barometric formula
    altitude_m = 44330.0 * (1.0 - (np.asarray(pressure_hPa) / 1013.25) ** (1.0 / 5.255))

    # Convert to feet
    altitude_ft = altitude_m * 3.28084
    return altitude_ft