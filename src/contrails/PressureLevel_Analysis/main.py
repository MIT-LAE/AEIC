import numpy as np
import xarray as xr

def compute_vapor_pressure_and_rhi(ds):
    
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


file_path = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/RAW/20241201.nc"

ds = xr.open_dataset(file_path)

df_RHI = compute_vapor_pressure_and_rhi(ds)

print(df_RHI.head(10))