import xarray as xr

import xarray as xr

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

