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
    
    
    
def plot_issr_along_geodesic(ds, rf_df, origin, destination, fileName, valid_time_index=0):
    """
    Compute and plot ISSR intersection along the geodesic arc between two points.

    Parameters:
        ds (xarray.Dataset): ERA5 ISSR dataset
        rf_df (pd.DataFrame): Radiative forcing lookup table
        origin (tuple): (lat, lon)
        destination (tuple): (lat, lon)
        fileName (str): Basename for output
        valid_time_index (int): Time index into the dataset
        npts (int): Number of points to discretize geodesic arc
    """

    geod = Geod(ellps='WGS84')
    
    # Compute number of points at ~10 km resolution
    total_distance_km = geodesic(origin, destination).km
    npts = max(int(total_distance_km // 10) - 1, 1)
    
    
    arc_coords = geod.npts(origin[1], origin[0], destination[1], destination[0], npts)
    arc_coords.insert(0, (origin[1], origin[0]))
    arc_coords.append((destination[1], destination[0]))

    arc_lats = [lat for lon, lat in arc_coords]
    arc_lons = [lon for lon, lat in arc_coords]

    ds_t = ds.isel(valid_time=valid_time_index)

    pressure_levels = ds.pressure_level.values
    arc_spacing_km = geodesic(origin, destination).km / len(arc_coords)

    total_issr_by_km = {}
    total_rf_by_km = {}

    print("=== ISSR Segment Lengths by Altitude (based on arc sampling) ===")
    for p in pressure_levels:
        issr_flags = []

        for lat, lon in zip(arc_lats, arc_lons):
            try:
                val = ds_t['ISSR_flag'].sel(
                    pressure_level=p, latitude=lat, longitude=lon, method='nearest'
                ).values.item()
                issr_flags.append(val)
            except:
                issr_flags.append(0)

        issr_flags = np.array(issr_flags)
        issr_mask = issr_flags == 1

        segments = []
        current = []

        for i, flag in enumerate(issr_mask):
            if flag:
                current.append(i)
            else:
                if len(current) > 1:
                    segments.append(current)
                current = []
        if len(current) > 1:
            segments.append(current)

        total_len = sum((len(seg) - 1) * arc_spacing_km for seg in segments)

        alt_km = pressure_to_altitude_km(p)
        total_issr_by_km[alt_km] = total_len
        rf_per_km = interpolate_rf_from_altitude(alt_km, rf_df) * 1e-9
        total_rf_by_km[alt_km] = total_len * rf_per_km

        print(f"{alt_km:.2f} km: {total_len:.1f} km → {total_rf_by_km[alt_km]:.11f} mW/m²")

    # Save data
    write_issr_rf_to_csv(total_issr_by_km, total_rf_by_km, output_path=f"Raw/{fileName}.csv")
    #plot_rf_extent_altitude_contour(total_issr_by_km, total_rf_by_km, save_path=f"Plots/Slices/{fileName}_contour.png")




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

def plot_issr_flag_altitude_vs_distance(ds_RHI, origin, destination, valid_time_index=0):
    """
    Plot ISSR flag as a function of altitude and arc distance for an O-D pair.

    Parameters:
        ds_RHI (xarray.Dataset): ERA5 ISSR dataset
        origin (tuple): (lat, lon)
        destination (tuple): (lat, lon)
        valid_time_index (int): Time index for ds_RHI
    """
    geod = Geod(ellps='WGS84')
    total_distance_km = geodesic(origin, destination).km
    npts = max(int(total_distance_km // 10) - 1, 1)
    
    arc_coords = geod.npts(origin[1], origin[0], destination[1], destination[0], npts)
    arc_coords.insert(0, (origin[1], origin[0]))
    arc_coords.append((destination[1], destination[0]))

    arc_lats = [lat for lon, lat in arc_coords]
    arc_lons = [lon for lon, lat in arc_coords]
    arc_spacing_km = total_distance_km / len(arc_coords)
    arc_distances = np.arange(len(arc_coords)) * arc_spacing_km

    ds_t = ds_RHI.isel(valid_time=valid_time_index)
    pressure_levels = ds_RHI.pressure_level.values

    issr_matrix = []
    altitudes_km = []

    for p in pressure_levels:
        row = []
        for lat, lon in zip(arc_lats, arc_lons):
            try:
                val = ds_t['ISSR_flag'].sel(
                    pressure_level=p, latitude=lat, longitude=lon, method='nearest'
                ).values.item()
                row.append(val)
            except:
                row.append(np.nan)
        issr_matrix.append(row)
        altitudes_km.append(pressure_to_altitude_km(p))

    issr_matrix = np.array(issr_matrix)
    altitudes_km = np.array(altitudes_km)

    # Plot as heatmap
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(arc_distances, altitudes_km, issr_matrix, cmap='Blues', shading='auto')
    plt.colorbar(label="ISSR Flag (1 = In ISSR)")
    plt.title(f"ISSR Flag Along Geodesic\n{origin} → {destination}")
    plt.xlabel("Distance Along Arc (km)")
    plt.ylabel("Altitude (km)")
    plt.tight_layout()
    plt.savefig("ISSR_slice_along_geodesic.png", dpi=300)


# === RUN SCRIPT ===
fileName = "20241229"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/12_2024/RHi_{fileName}.nc"
rf_table_path = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/PressureLevel_Analysis/Radel_Shine.csv"

# Get R.F per unit kilometer (incorrect for now, fix later)
rf_df = pd.read_csv(rf_table_path)

# Get RHI contours from post-processed ERA-5 data
ds_RHI = xr.open_dataset(file_path)

# Fix to just one arc for now
fileName2 = f"BOS-MIA_{fileName}_"
origin = (42.3656, -71.0096)    # Boston
destination = (32.8968, -97.0370)  # Dallas
#destination = (34.0522, -118.2437)  # Los Angeles
#destination = (47.4502, -122.3088)  # Seattle
#destination = (25.7933, -80.2906)  # Miami


plot_issr_flag_altitude_vs_distance(ds_RHI, origin=origin, destination=destination)


#plot_issr_along_geodesic(ds_RHI, rf_df, origin, destination, fileName2, valid_time_index=0)




