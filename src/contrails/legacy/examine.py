import xarray as xr
import pandas as pd
# Load the CSV using pandas
df = pd.read_csv("L137_coefficients.csv")

# Extract a and b as NumPy arrays
a_half = df["a"].values
b_half = df["b"].values


def summarize_era5_nc(nc_path):
    """
    Summarizes the contents of an ERA-5 NetCDF file.

    Parameters
    ----------
    nc_path : str
        Path to the .nc file

    Returns
    -------
    None
    """
    # Open dataset
    ds = xr.open_dataset(nc_path)

    # Print dataset summary
    print("\n ERA-5 Dataset Summary")
    print("-" * 40)
    print(ds)

    print("\n Dimensions:")
    for dim in ds.dims:
        print(f"  {dim}: {ds.dims[dim]}")

    print("\n Coordinates:")
    for coord in ds.coords:
        print(f"  {coord}: {ds[coord].values[:5]} ...")  # preview first 5 values

    print("\n Data Variables:")
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].shape} {ds[var].attrs.get('units', '')}")

    # Global attributes
    print("\n Global Attributes:")
    for attr in ds.attrs:
        print(f"  {attr}: {ds.attrs[attr]}")

    return ds


import numpy as np
import xarray as xr

def model_levels_to_pressure(ds, ds_lnsp, a, b):
    """
    Convert ERA-5 model levels to pressure levels (Pa).

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset with `lnsp` and model-level data (e.g., temperature).
    a, b : 1D numpy arrays
        Hybrid coefficients (length: number of model levels + 1, i.e., half levels).

    Returns
    -------
    pressure : xarray.DataArray
        4D array of pressure [time, level, lat, lon] in Pa
    """
    # Convert log surface pressure to surface pressure
    ps = np.exp(ds_lnsp["lnsp"])  # shape: [time, lat, lon]

    # Compute full-level pressures from half-level a and b coefficients
    num_levels = len(a) - 1  # 137 for ERA-5

    # Create arrays for broadcasting
    a_full = 0.5 * (a[:-1] + a[1:])
    b_full = 0.5 * (b[:-1] + b[1:])

    # Expand dimensions
    a_full = xr.DataArray(a_full, dims=["level"])
    b_full = xr.DataArray(b_full, dims=["level"])

    # Broadcast pressure
    pressure = a_full + b_full * ps  # will automatically broadcast
    pressure = pressure.transpose("time", "level", "latitude", "longitude")
    pressure.name = "pressure"

    return pressure

import xarray as xr
import numpy as np

def model_levels_to_pressure_beta(
    ds: xr.Dataset,
    ds_lnsp: xr.Dataset,
    a_half: np.ndarray,
    b_half: np.ndarray
) -> xr.DataArray:
    """
    Efficiently compute 4D pressure field from ERA5 model-level dataset using hybrid coefficients.

    Parameters
    ----------
    ds : xarray.Dataset
        Main ERA5 dataset (contains t, u, v, q, etc.)
    ds_lnsp : xarray.Dataset
        Dataset containing 'lnsp' (log surface pressure) with dims: [valid_time, latitude, longitude]
    a_half : np.ndarray
        Half-level hybrid coefficient `a` (length 138)
    b_half : np.ndarray
        Half-level hybrid coefficient `b` (length 138)

    Returns
    -------
    pressure : xarray.DataArray
        4D pressure field with dims [valid_time, level, latitude, longitude], units in Pa
    """
    # Extract and process lnsp
    lnsp = ds_lnsp["lnsp"]

    # Remove singleton model_level dimension if present
    if "model_level" in lnsp.dims:
        lnsp = lnsp.squeeze("model_level")

    # Convert lnsp to surface pressure (Pa)
    ps = np.exp(lnsp)  # shape: [valid_time, latitude, longitude]

    # Compute full-level hybrid coefficients
    a_full = 0.5 * (a_half[:-1] + a_half[1:])  # length 137
    b_full = 0.5 * (b_half[:-1] + b_half[1:])

    # Create DataArrays with "level" dimension
    a_da = xr.DataArray(a_full, dims=["level"])
    b_da = xr.DataArray(b_full, dims=["level"])

    # Ensure ps is chunked (Dask-compatible)
    if not ps.chunks:
        ps = ps.chunk({"valid_time": 1})  # or more as needed

    # Broadcast and compute pressure lazily
    pressure = a_da + b_da * ps

    # Transpose to standard dimension order if needed
    pressure = pressure.transpose(*[dim for dim in ("valid_time", "level", "latitude", "longitude") if dim in pressure.dims])
    pressure.name = "pressure"
    pressure.attrs["units"] = "Pa"

    return pressure




def interpolate_to_pressure_level(ds, pressure, target_pressure=25000):
    """
    Interpolates t, u, v, q to a constant pressure level (e.g., 250 hPa).
    
    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 model level dataset (must contain 't', 'u', 'v', 'q').
    pressure : xarray.DataArray
        4D array [time, level, lat, lon] of pressure in Pa.
    target_pressure : float
        Target pressure level in Pa (e.g., 25000 for 250 hPa).
    
    Returns
    -------
    interp_data : dict of xarray.DataArray
        Interpolated fields at target pressure level.
    """
    log_p = np.log(pressure)

    variables = ['t', 'u', 'v', 'q']
    results = {}

    for var in variables:
        data = ds[var]
        log_target = np.log(target_pressure)

        # Interpolate in log-pressure space
        interp = xr.apply_ufunc(
            np.interp,
            log_target,                   # x value to interpolate to (scalar)
            log_p,                        # x (log-pressure)
            data,                         # y (e.g., temperature)
            input_core_dims=[[], ["level"], ["level"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype]
        )

        interp.name = f"{var}_at_{int(target_pressure//100)}hPa"
        results[var] = interp

    return results



# Example usage
if __name__ == "__main__":
    
    print("Extracting model level data...", end = '')
    ds = summarize_era5_nc("ERA5/model/20241223.nc")
    print("done!")
    
    print("Extracting surface pressure level...", end = '')
    ds_lnsp = summarize_era5_nc("ERA5/model/lnsp/20241223.nc")
    print("done!")
    
    
    print("Extracting pressure data..", end = '')
    # ds: your ERA5 xarray.Dataset with 't', 'u', 'v', 'q', 'lnsp'
    #pressure = model_levels_to_pressure(ds, ds_lnsp, a_half, b_half)
    pressure = model_levels_to_pressure_beta(ds, ds_lnsp, a_half, b_half)
    print("done!")
    
    interp_vars = interpolate_to_pressure_level(ds, pressure, target_pressure=25000)

    
    
    

