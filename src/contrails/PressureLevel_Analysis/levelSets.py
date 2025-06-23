import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from pyproj import Geod
import xarray as xr
import csv

# (Data source: Fig. 9) Rädel, G. and Shine, K.P., 2008. Radiative forcing by persistent contrails and its dependence on cruise altitudes. Journal of Geophysical Research: Atmospheres, 113(D7).
def interpolate_rf_from_altitude(alt_km, rf_df):
    """Interpolate RF (mW/m²) from altitude in km using Radel_Shine.csv data."""
    return np.interp(alt_km, rf_df["height (km)"], rf_df["RF (mW/m^2)"])


def write_issr_rf_to_csv(total_issr_by_km, total_rf_by_km, output_path="issr_rf_by_altitude.csv"):
    """
    Write ISSR extent and radiative forcing data by altitude into a CSV file.

    Parameters:
        total_issr_by_km (dict): Dictionary of {altitude_km: issr_extent_km}
        total_rf_by_km (dict): Dictionary of {altitude_km: rf_mW_per_m2}
        output_path (str): Path to save the CSV file
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Altitude_km", "ISSR_extent_km", "RF_mW_per_m2"])
        for alt in sorted(total_rf_by_km.keys()):
            writer.writerow([
                f"{alt:.2f}",
                f"{total_issr_by_km.get(alt, 0):.3f}",
                f"{total_rf_by_km.get(alt, 0):.11f}"
            ])
    print(f"Saved ISSR extent and RF data to: {output_path}")

def plot_rf_extent_altitude_contour(total_issr_by_km, total_rf_by_km, save_path="rf_extent_altitude_contour.png"):
    """
    Plot 2D contour of RF (mW/m²) as a function of horizontal ISSR extent and altitude.

    Parameters:
        total_issr_by_km (dict): {altitude_km: issr_length_km}
        total_rf_by_km (dict): {altitude_km: rf_mW_per_m2}
        save_path (str): Path to save the output PNG
    """
    # Prepare data
    altitudes = np.array(sorted(total_rf_by_km.keys()))
    extents = np.array([total_issr_by_km.get(alt, 0) for alt in altitudes])
    rfs = np.array([total_rf_by_km.get(alt, 0) for alt in altitudes])

    # Create 2D grid using simple outer product model (alt vs extent)
    X, Y = np.meshgrid(np.linspace(extents.min(), extents.max(), 100),
                       np.linspace(altitudes.min(), altitudes.max(), 100))
    Z = np.interp(Y, altitudes, rfs) * np.interp(X, extents, np.ones_like(extents))

    # Plot contour
    plt.figure(figsize=(8, 6), dpi=300)
    contour = plt.contourf(X, Y, Z, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour)
    cbar.set_label("Radiative Forcing (mW/m²)", fontsize=11)

    plt.xlabel("ISSR Horizontal Extent (km)", fontsize=12)
    plt.ylabel("Altitude (km)", fontsize=12)
    plt.title("2D Contour of Radiative Forcing", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved RF contour plot to: {save_path}")



def pressure_to_altitude_km(pressure_hPa):
    """Convert pressure to altitude in kilometers using the barometric formula."""
    altitude_m = 44330.0 * (1.0 - (pressure_hPa / 1013.25)**(1.0 / 5.255))
    return altitude_m / 1000.0  # meters to kilometers

def plot_issr_along_geodesic(ds, rf_df, origin, destination, fileName, valid_time_index=0):
    """
    Plot ISSR occurrence along the geodesic arc between Boston and Dallas,
    with altitude in km vs distance in km. Also prints total ISSR arc length
    per altitude, counting only consecutive segments.
    """
    # Define locations (lat, lon)
    #boston = (42.3656, -71.0096)
    #dallas = (32.8968, -97.0370)

    # Subset dataset at a specific time
    ds_t = ds.isel(valid_time=valid_time_index)

    # Subset around Boston–Dallas corridor
    lat_min, lat_max = min(origin[0], destination[0]) - 5, max(origin[0], destination[0]) + 5
    lon_min, lon_max = min(origin[1], destination[1]) - 5, max(origin[1], destination[1]) + 5
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

    # Compute geodesic distance from origin (in km)
    geod = Geod(ellps='WGS84')
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

    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(dists_km, alt_vals_km, color='crimson', linewidth=0, marker='s', ms=8, mfc="C0")
    plt.xlabel("Distance from Boston (km)", fontsize=13)
    plt.ylabel("Altitude (km)", fontsize=13)
    plt.title("ISSR Altitude along Boston → Dallas Geodesic", fontsize=14)
    plt.grid(True, linestyle=':', linewidth=0.01)
    plt.tight_layout()
    plt.savefig(f"Plots/Slices/{fileName}ISSR_slice.png", dpi=300)
    plt.close()
    
    
    total_rf_by_km = {}
    total_issr_by_km = {}

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
        total_issr_by_km[alt] = total_length_km
        
        rf_per_km = interpolate_rf_from_altitude(alt, rf_df) * 10**(-9)
        
        total_rf_by_km[alt] = total_length_km * rf_per_km
        
        
    # Print per-altitude RF and ISSR length
    for alt in sorted(total_rf_by_km.keys()):
        length = total_issr_by_km[alt]
        rf = total_rf_by_km[alt]
        print(f"{alt:.2f} km: {length:.1f} km → {rf:.11f} mW/m²")
        
    #plot_rf_extent_altitude_contour(total_issr_by_km, total_rf_by_km)
    
    write_issr_rf_to_csv(total_issr_by_km, total_rf_by_km, output_path=f"Raw/{fileName}.csv")
        
        
# === RUN SCRIPT ===
fileName = "20241201.nc"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/12_2024/RHi_{fileName}"
rf_table_path = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/PressureLevel_Analysis/Radel_Shine.csv"
rf_df = pd.read_csv(rf_table_path)
ds_RHI = xr.open_dataset(file_path)

fileName = "Bos-SEA"
origin = (42.3656, -71.0096)    # Boston
#destination = (32.8968, -97.0370)  # Dallas
#destination = (34.0522, -118.2437)  # Los Angeles
destination = (47.4502, -122.3088)  # Seattle



plot_issr_along_geodesic(ds_RHI, rf_df, origin, destination, fileName, valid_time_index=0)


