import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Geod
import xarray as xr


def pressure_to_altitude_km(pressure_hPa):
    """Convert pressure to altitude in kilometers using the barometric formula."""
    altitude_m = 44330.0 * (1.0 - (pressure_hPa / 1013.25)**(1.0 / 5.255))
    return altitude_m / 1000.0  # meters to kilometers

def plot_issr_along_geodesic(ds, valid_time_index=0):
    """
    Plot ISSR occurrence along the geodesic arc between Boston and Dallas,
    with altitude in km vs distance in km. Also prints total ISSR arc length
    per altitude, counting only consecutive segments.
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
    P, LAT, LON = np.meshgrid(p_1d, lat_1d, lon_1d, indexing='ij')

    issr_mask = mask.transpose('pressure_level', 'latitude', 'longitude').values
    lon_vals = LON[issr_mask]
    lat_vals = LAT[issr_mask]
    p_vals   = P[issr_mask]

    # Convert pressure → altitude in km
    alt_vals_km = pressure_to_altitude_km(p_vals)

    # Compute geodesic distance from Boston (in km)
    geod = Geod(ellps='WGS84')
    _, _, dists_m = geod.inv(
        np.full_like(lon_vals, boston[1]),
        np.full_like(lat_vals, boston[0]),
        lon_vals,
        lat_vals
    )
    dists_km = dists_m / 1000.0

    # Sort by distance
    sort_idx = np.argsort(dists_km)
    dists_km = dists_km[sort_idx]
    alt_vals_km = alt_vals_km[sort_idx]

    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(dists_km, alt_vals_km, color='crimson', linewidth=0, marker='s', ms=8, mfc="C0")
    plt.xlabel("Distance from Boston (km)", fontsize=13)
    plt.ylabel("Altitude (km)", fontsize=13)
    plt.title("ISSR Altitude along Boston → Dallas Geodesic", fontsize=14)
    plt.grid(True, linestyle=':', linewidth=0.01)
    plt.tight_layout()
    plt.savefig("sample.png", dpi=300)
    plt.close()

    # Print ISSR segment lengths by altitude
    print("=== ISSR Segment Lengths by Altitude (consecutive only, in km) ===")
    unique_alts = np.unique(alt_vals_km)
    for alt in sorted(unique_alts):
        mask_alt = alt_vals_km == alt
        dists = dists_km[mask_alt]
        dists_sorted = np.sort(dists)

        # Find consecutive segments (e.g., ≤25 km apart)
        segments = []
        if len(dists_sorted) == 0:
            continue

        segment = [dists_sorted[0]]
        for i in range(1, len(dists_sorted)):
            if abs(dists_sorted[i] - dists_sorted[i - 1]) <= 25:
                segment.append(dists_sorted[i])
            else:
                if len(segment) > 1:
                    segments.append(segment)
                segment = [dists_sorted[i]]
        if len(segment) > 1:
            segments.append(segment)

        total_length_km = sum(seg[-1] - seg[0] for seg in segments)
        print(f"{alt:.1f} km: {total_length_km:.1f} km")



# === RUN SCRIPT ===
fileName = "20241201.nc"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/12_2024/RHi_{fileName}"
ds_RHI = xr.open_dataset(file_path)

plot_issr_along_geodesic(ds_RHI, valid_time_index=0)
