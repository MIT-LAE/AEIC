import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import glob

def plot_rf_contour_from_csv(csv_path, save_path="rf_contour_from_csv.png"):
    """
    Read ISSR-RF data from a CSV and plot a 2D contour of RF (mW/m²)
    as a function of altitude and ISSR horizontal extent.

    Parameters:
        csv_path (str): Path to CSV with columns Altitude_km, ISSR_extent_km, RF_mW_per_m2
        save_path (str): Path to save the contour plot PNG
    """
    df = pd.read_csv(csv_path)

    # Extract columns
    altitudes = df["Altitude_km"].values
    extents = df["ISSR_extent_km"].values
    rfs = df["RF_mW_per_m2"].values

    # Define grid for interpolation
    alt_grid = np.linspace(min(altitudes), max(altitudes), 200)
    extent_grid = np.linspace(min(extents), max(extents), 200)
    X, Y = np.meshgrid(extent_grid, alt_grid)

    # Interpolate using griddata
    Z = griddata(
        points=(extents, altitudes),
        values=rfs,
        xi=(X, Y),
        method='linear'
    )

    # Plot
    plt.figure(figsize=(8, 6), dpi=300)
    contour = plt.contourf(X, Y, Z, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour)
    cbar.set_label("Radiative Forcing (mW/m²)", fontsize=11)

    plt.xlabel("ISSR intersection (km)", fontsize=12)
    plt.ylabel("Altitude (km)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved contour plot to: {save_path}")




def plot_rf_contour_from_multiple_csvs(csv_paths, save_path="rf_contour_total.png"):
    """
    Read multiple ISSR-RF CSV files and plot a 2D contour of total RF (mW/m²)
    as a function of altitude and ISSR extent.

    Parameters:
        csv_paths (list of str): List of CSV file paths
        save_path (str): Path to save the output contour plot
    """
    all_points = []
    all_rfs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        altitudes = df["Altitude_km"].values
        extents = df["ISSR_extent_km"].values
        rfs = df["RF_mW_per_m2"].values

        # Append all data
        all_points.extend(zip(extents, altitudes))  # (x=extent, y=alt)
        all_rfs.extend(rfs)

    all_points = np.array(all_points)
    all_rfs = np.array(all_rfs)

    # Define a consistent grid
    extent_range = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))
    altitude_range = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
    extent_grid = np.linspace(*extent_range, 200)
    altitude_grid = np.linspace(*altitude_range, 200)
    X, Y = np.meshgrid(extent_grid, altitude_grid)

    # Griddata uses (x, y) → z mapping
    Z = griddata(
        points=all_points,
        values=all_rfs,
        xi=(X, Y),
        method='linear',
        fill_value=0.0
    )

    # Plot
    plt.figure(figsize=(8, 6), dpi=300)
    contour = plt.contourf(X, Y, Z, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour)
    cbar.set_label("Total Radiative Forcing (mW/m²)", fontsize=11)

    plt.xlabel("ISSR intersection (km)", fontsize=12)
    plt.ylabel("Altitude (km)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved total RF contour plot to: {save_path}")


#plot_rf_contour_from_csv("Raw/Bos-SEA.csv", save_path="Contour_Bos-SEA.png")


csv_files = glob.glob("Raw/*.csv")
plot_rf_contour_from_multiple_csvs(csv_files, save_path="Contour_Total_RF.png")