import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Geod
import xarray as xr

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Geod

def pressure_to_altitude_ft(pressure_hPa):
    """Convert pressure to altitude in feet."""
    altitude_m = 44330.0 * (1.0 - (pressure_hPa / 1013.25)**(1.0 / 5.255))
    return altitude_m * 3.28084  # meters to feet

def plot_issr_along_geodesic(ds, valid_time_index=0):
    """
    Plot ISSR occurrence along the geodesic arc between Boston and Dallas,
    with altitude in feet vs distance in nautical miles.

    Parameters:
        ds (xarray.Dataset): Must include 'ISSR_flag' and 'pressure_level'
        valid_time_index (int): Time index to select
    """
    # Define locations (lat, lon)
    boston = (42.3656, -71.0096)
    dallas = (32.8968, -97.0370)

    # Subset dataset at a specific time
    ds_t = ds.isel(valid_time=valid_time_index)

    # Subset around Boston–Dallas corridor
    lat_min, lat_max = min(boston[0], dallas[0]) - 5, max(boston[0], dallas[0]) + 5
    lon_min, lon_max = min(boston[1], dallas[1]) - 5, max(boston[1], dallas[1]) + 5
    ds_subset = ds_t.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    # Mask ISSR = 1
    mask = ds_subset['ISSR_flag'] == 1
    if not mask.any():
        print("No ISSR regions found in the selected time slice.")
        return

    # Extract lat/lon/pressure where ISSR == 1
    lat_1d = ds_subset.latitude.values
    lon_1d = ds_subset.longitude.values
    p_1d = ds_subset.pressure_level.values

    # Match shape to mask: (pressure, lat, lon)
    P, LAT, LON = np.meshgrid(p_1d, lat_1d, lon_1d, indexing='ij')

    issr_mask = mask.transpose('pressure_level', 'latitude', 'longitude').values
    lon_vals = LON[issr_mask]
    lat_vals = LAT[issr_mask]
    p_vals   = P[issr_mask]

    # Convert pressure → altitude in feet
    alt_vals_ft = pressure_to_altitude_ft(p_vals)

    # Distance from Boston in nautical miles
    geod = Geod(ellps='WGS84')
    _, _, dists = geod.inv(
        np.full_like(lon_vals, boston[1]),
        np.full_like(lat_vals, boston[0]),
        lon_vals,
        lat_vals
    )
    dists_nm = dists / 1852.0  # meters to nautical miles

    # Sort by distance
    sort_idx = np.argsort(dists_nm)
    dists_nm = dists_nm[sort_idx]
    alt_vals_ft = alt_vals_ft[sort_idx]

    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(dists_nm, alt_vals_ft, color='crimson', linewidth=0, marker = 's', ms = 8, mfc = "C0")
    plt.xlabel("Distance from Boston (nautical miles)", fontsize=13)
    plt.ylabel("Altitude (feet)", fontsize=13)
    plt.title("ISSR Altitude along Boston → Dallas Geodesic", fontsize=14)
    plt.grid(True, linestyle=':', linewidth=0.01)
    plt.tight_layout()
    plt.savefig("sample.png", dpi = 300)

    # Group by altitude
    unique_alts = np.unique(alt_vals_ft)
    print("=== ISSR Segment Lengths by Altitude (consecutive only) ===")
    for alt in sorted(unique_alts):
        mask_alt = alt_vals_ft == alt
        dists = dists_nm[mask_alt]
        dists_sorted = np.sort(dists)
        
        # Find consecutive segments
        segments = []
        segment = [dists_sorted[0]]
        
        
        for i in range(1, len(dists_sorted)):
            if abs(dists_sorted[i] - dists_sorted[i - 1]) <= 15:  # threshold in nm
                segment.append(dists_sorted[i])
            else:
                if len(segment) > 1:
                    segments.append(segment)
                    
        # Compute total length
        total_length_nm = sum(seg[-1] - seg[0] for seg in segments)
        fl_label = f"FL{int(round(alt / 100)):02d}"
        print(f"{fl_label} ({int(alt)} ft): {total_length_nm:.1f} nm")
                
                
            
        
        
        
        
        
    
    


# Main
# === RUN SCRIPT ===
fileName = "20241201.nc"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/12_2024/RHi_{fileName}"

ds_RHI = xr.open_dataset(file_path)

#print(ds_RHI)

plot_issr_along_geodesic(ds_RHI, valid_time_index=0)

# Assuming your DataFrame is called df_issr:
#plot_issr_along_bos_dal(df_issr)


