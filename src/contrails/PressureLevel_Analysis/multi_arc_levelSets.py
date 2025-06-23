import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Geod
import xarray as xr


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
    Plot ISSR altitude vs geodesic distance for multiple destination locations from a fixed origin.

    Parameters:
        ds (xarray.Dataset): Dataset with ISSR_flag, pressure_level, latitude, longitude
        rf_df (pd.DataFrame): Not used in current plot, but kept for extension
        origin (tuple): (lat, lon) for the fixed start point
        destinations (list of tuple): List of (lat, lon) tuples for end points
        valid_time_index (int): Time index to subset the dataset
        save_path (str): File path to save the resulting plot
    """
    import matplotlib.cm as cm

    ds_t = ds.isel(valid_time=valid_time_index)
    colors = cm.tab10.colors

    plt.figure(figsize=(10, 6), dpi=300)

    for i, destination in enumerate(destinations):
        # Bounding box
        lat_min, lat_max = min(origin[0], destination[0]) - 5, max(origin[0], destination[0]) + 5
        lon_min, lon_max = min(origin[1], destination[1]) - 5, max(origin[1], destination[1]) + 5
        ds_subset = ds_t.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

        mask = ds_subset['ISSR_flag'] == 1
        if not mask.any():
            print(f"No ISSR regions found for path to {destination}. Skipping.")
            continue

        lat_1d = ds_subset.latitude.values
        lon_1d = ds_subset.longitude.values
        p_1d = ds_subset.pressure_level.values
        P, LAT, LON = np.meshgrid(p_1d, lat_1d, lon_1d, indexing='ij')

        issr_mask = mask.transpose('pressure_level', 'latitude', 'longitude').values
        lon_vals = LON[issr_mask]
        lat_vals = LAT[issr_mask]
        p_vals = P[issr_mask]

        alt_vals_km = pressure_to_altitude_km(p_vals)

        geod = Geod(ellps='WGS84')
        _, _, dists_m = geod.inv(
            np.full_like(lon_vals, origin[1]),
            np.full_like(lat_vals, origin[0]),
            lon_vals,
            lat_vals
        )
        dists_km = dists_m / 1000.0

        sort_idx = np.argsort(dists_km)
        dists_km = dists_km[sort_idx]
        alt_vals_km = alt_vals_km[sort_idx]

        plt.plot(
            dists_km,
            alt_vals_km,
            linestyle='',
            marker='s',
            ms=6,
            label=f"{origin} → {destination}",
            color=colors[i % len(colors)]
        )

    plt.xlabel(f"Distance ({origin[0]:.1f}°, {origin[1]:.1f}°) [km]", fontsize=13)
    plt.ylabel("Altitude (km)", fontsize=13)
    plt.title("ISSR Altitude along Multiple Geodesics", fontsize=14)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




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

