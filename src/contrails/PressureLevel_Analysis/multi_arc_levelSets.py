import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Geod
import xarray as xr
from collections import defaultdict

# (Data source: Fig. 9) Rädel, G. and Shine, K.P., 2008. Radiative forcing by persistent contrails and its dependence on cruise altitudes. Journal of Geophysical Research: Atmospheres, 113(D7).
def interpolate_rf_from_altitude(alt_km, rf_df):
    """Interpolate RF (mW/m²) from altitude in km using Radel_Shine.csv data."""
    return np.interp(alt_km, rf_df["height (km)"], rf_df["RF (mW/m^2)"])

def pressure_to_altitude_km(pressure_hPa):
    """Convert pressure to altitude in kilometers using the barometric formula."""
    altitude_m = 44330.0 * (1.0 - (pressure_hPa / 1013.25)**(1.0 / 5.255))
    return altitude_m / 1000.0  # meters to kilometers



def plot_issr_from_origin_to_multiple_destinations(
    ds,
    rf_df,
    origin,
    destinations,
    valid_time_index=0,
    save_path="issr_geodesic_multiple.png"
):
    """
    Plot ISSR altitude vs geodesic distance for multiple destinations from a fixed origin.
    Compute total ISSR extent along continuous segments, and report length per altitude level.

    Parameters:
        ds (xarray.Dataset): Dataset with ISSR_flag, pressure_level, latitude, longitude
        rf_df (pd.DataFrame): (not used here but retained for extensibility)
        origin (tuple): (lat, lon) start point
        destinations (list of tuple): list of (lat, lon) endpoints
        valid_time_index (int): time index to subset dataset
        save_path (str): file path for saved figure

    Returns:
        total_issr_km (float): total ISSR segment length [km]
        issr_extent_by_level (dict): altitude_km → ISSR extent [km]
    """
    ds_t = ds.isel(valid_time=valid_time_index)
    geod = Geod(ellps='WGS84')
    total_issr_km = 0.0
    issr_extent_by_level = defaultdict(float)

    plt.figure(figsize=(10, 6), dpi=300)
    colors = plt.cm.tab10.colors

    for i, destination in enumerate(destinations):
        # Subset region around O-D pair
        lat_min, lat_max = min(origin[0], destination[0]) - 5, max(origin[0], destination[0]) + 5
        lon_min, lon_max = min(origin[1], destination[1]) - 5, max(origin[1], destination[1]) + 5
        ds_subset = ds_t.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

        # Mask ISSR=1 regions
        mask = ds_subset['ISSR_flag'] == 1
        if not mask.any():
            print(f"No ISSR regions found for path to {destination}. Skipping.")
            continue

        # Extract coordinates and pressure
        lat_1d = ds_subset.latitude.values
        lon_1d = ds_subset.longitude.values
        p_1d = ds_subset.pressure_level.values
        P, LAT, LON = np.meshgrid(p_1d, lat_1d, lon_1d, indexing='ij')

        issr_mask = mask.transpose('pressure_level', 'latitude', 'longitude').values
        lon_vals = LON[issr_mask]
        lat_vals = LAT[issr_mask]
        p_vals   = P[issr_mask]

        # Convert pressure to altitude (km)
        alt_vals_km = pressure_to_altitude_km(p_vals)

        # Distance from origin
        _, _, dists_m = geod.inv(
            np.full_like(lon_vals, origin[1]),
            np.full_like(lat_vals, origin[0]),
            lon_vals,
            lat_vals
        )
        dists_km = dists_m / 1000.0

        # Sort by distance
        sort_idx = np.argsort(dists_km)
        dists_km = dists_km[sort_idx]
        alt_vals_km = alt_vals_km[sort_idx]

        # Plot ISSR points
        plt.plot(
            dists_km,
            alt_vals_km,
            linestyle='',
            marker='s',
            ms=6,
            label=f"{origin} → {destination}",
            color=colors[i % len(colors)]
        )

        # Segment computation
        threshold_km = 50.0
        segment_lengths = []
        segment_altitudes = []

        current_segment_len = 0.0
        current_segment_alt = [alt_vals_km[0]]

        for j in range(1, len(dists_km)):
            
            gap = geod.inv(
            lon_vals[sort_idx][j - 1],
            lat_vals[sort_idx][j - 1],
            lon_vals[sort_idx][j],
            lat_vals[sort_idx][j]
            )[2] / 1000.0  # in km
            
            
            if gap < threshold_km:
                current_segment_len += gap
                current_segment_alt.append(alt_vals_km[j])
            else:
                if current_segment_len > 0:
                    segment_lengths.append(current_segment_len)
                    segment_altitudes.append(np.mean(current_segment_alt))
                current_segment_len = 0.0
                current_segment_alt = [alt_vals_km[j]]

        if current_segment_len > 0:
            segment_lengths.append(current_segment_len)
            segment_altitudes.append(np.mean(current_segment_alt))

        # Accumulate per flight level
        for seg_len, seg_alt in zip(segment_lengths, segment_altitudes):
            level = round(seg_alt, 2)
            issr_extent_by_level[level] += seg_len
            total_issr_km += seg_len

    # Final plot
    plt.xlabel(f"Distance from Origin ({origin[0]:.1f}°, {origin[1]:.1f}°) [km]", fontsize=13)
    plt.ylabel("Altitude (km)", fontsize=13)
    plt.title("ISSR Altitude along Multiple Geodesics", fontsize=14)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Print summary
    print(f"\nTotal ISSR extent across all destinations: {total_issr_km:.1f} km")
    print("ISSR extent per altitude level:")
    for level in sorted(issr_extent_by_level):
        print(f"  {level:.2f} km: {issr_extent_by_level[level]:.1f} km")

    return total_issr_km, dict(issr_extent_by_level)




# Define O-D pairs
origin = (42.3656, -71.0096)  # Boston

destinations = [
    (32.8968, -97.0370),      # Dallas
    (34.0522, -118.2437),     # Los Angeles
    (47.4502, -122.3088),     # Seattle
    (25.7933, -80.2906)       # Miami
]


fileName = "20241201.nc"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/12_2024/RHi_{fileName}"
rf_table_path = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/PressureLevel_Analysis/Radel_Shine.csv"
rf_df = pd.read_csv(rf_table_path)
ds_RHI = xr.open_dataset(file_path)

plot_issr_from_origin_to_multiple_destinations(
    ds_RHI,
    rf_df,
    origin,
    destinations,
    valid_time_index=0,
    save_path="issr_geodesic_multiple.png"
)

