import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def inspect_and_plot_u_wind_profile(filepath):
    """
    Reads a NetCDF file with U and V wind profiles and plots
    a contour of the U-wind as a function of ground track distance and altitude.

    Parameters:
    - filepath: Path to the .nc NetCDF file.
    """
    try:
        ds = xr.open_dataset(filepath)
    except Exception as e:
        print(f"❌ Failed to read NetCDF file: {filepath}")
        print("Error:", e)
        return

    print(f"\n✅ Opened file: {filepath}\n")

    # Print metadata
    print("📋 Global Attributes:")
    for k, v in ds.attrs.items():
        print(f"  - {k}: {v}")

    print("\n📏 Coordinates:")
    for coord in ds.coords:
        values = ds[coord].values
        preview = values[:min(5, len(values))]
        print(f"  - {coord} ({ds[coord].shape}): {preview}...")

    print("\n📌 Data Variables:")
    for var in ds.data_vars:
        print(f"  - {var}: shape = {ds[var].shape}, dtype = {ds[var].dtype}")

    # Extract U-wind and convert pressure to altitude
    pressure = ds['pressure_hPa'].values
    altitude_m = 44330.0 * (1 - (pressure / 1013.25) ** (1 / 5.255))

    distance_km = ds['distance_km'].values
    u_wind = ds['u_component'].values  # shape: (npts, nlevels)

    # If pressure is increasing, flip axes to match altitude increase
    if pressure[0] < pressure[-1]:
        u_wind = u_wind[:, ::-1]
        altitude_m = altitude_m[::-1]

    # Create contour plot
    X, Y = np.meshgrid(distance_km, altitude_m)

    fig, ax = plt.subplots(figsize=(10, 6))
    cs = ax.contourf(X, Y, u_wind.T, levels=50, cmap="viridis", extend='both')

    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label("U-wind (m/s)", fontsize=14, fontname="Times New Roman")

    ax.set_title("U-wind Component Along Flight Path", fontsize=16, fontname="Times New Roman")
    ax.set_xlabel("Ground Track Distance (km)", fontsize=14, fontname="Times New Roman")
    ax.set_ylabel("Altitude (m)", fontsize=14, fontname="Times New Roman")
    ax.tick_params(labelsize=12)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(12)

    plt.tight_layout()
    plt.show()

    ds.close()


inspect_and_plot_u_wind_profile("Profiles/DCA_ORD_wind_profile.nc")
