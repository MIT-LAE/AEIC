import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


def pressure_to_altitude_ft(pressure_hPa):
    """Convert pressure (hPa) to altitude (ft) using the barometric formula."""
    altitude_m = 44330.0 * (1.0 - (np.asarray(pressure_hPa) / 1013.25) ** (1.0 / 5.255))
    return altitude_m * 3.28084


def plot_issr_conus_by_altitude(ds_RHI, time_index):
    """
    Plot ISSR = 1 points over CONUS colored by altitude (ft), with a colorbar in flight levels.

    Parameters:
    - ds_RHI: xarray.Dataset with dimensions [valid_time, pressure_level, latitude, longitude]
    - time_index: index into the valid_time dimension
    """
    # Subset at the selected time
    ds = ds_RHI.isel(valid_time=time_index)

    # Define CONUS bounds
    conus_bounds = {
        'lat_min': 24.5,
        'lat_max': 49.5,
        'lon_min': -125.0,
        'lon_max': -66.5
    }

    # Subset to CONUS region
    ds_conus = ds.sel(
        latitude=slice(conus_bounds['lat_max'], conus_bounds['lat_min']),
        longitude=slice(conus_bounds['lon_min'], conus_bounds['lon_max'])
    )

    # Mask ISSR = 1
    issr_mask = ds_conus['ISSR_flag'] == 1
    ds_issr = ds_conus.where(issr_mask, drop=True)

    # Convert pressure levels to altitude and wrap as xarray DataArray
    pressure_levels = ds_issr['pressure_level']
    altitude_ft_1d = xr.DataArray(
        data=pressure_to_altitude_ft(pressure_levels.values),
        dims=['pressure_level'],
        coords={'pressure_level': pressure_levels}
    )

    # Broadcast altitude to match ISSR_flag shape
    altitude_3d, _ = xr.broadcast(altitude_ft_1d, ds_issr['ISSR_flag'])

    # Stack for 2D plotting
    stacked = ds_issr.stack(points=("pressure_level", "latitude", "longitude"))
    stacked_altitude = altitude_3d.stack(points=("pressure_level", "latitude", "longitude"))

    # Filter non-NaN points
    valid = ~np.isnan(stacked['ISSR_flag'])
    lon_vals = stacked['longitude'].values[valid]
    lat_vals = stacked['latitude'].values[valid]
    alt_vals = stacked_altitude.values[valid]

    # Normalize altitude for transparency (alpha)
    alt_min = alt_vals.min()
    alt_max = alt_vals.max()
    alpha_vals = 1.0 - (alt_vals - alt_min) / (alt_max - alt_min)
    alpha_vals = np.clip(alpha_vals, 0.1, 0.8)

    # Set fixed color scale in feet
    altitude_min_ft = 20000
    altitude_max_ft = 40000

    # Plotting
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([
        conus_bounds['lon_min'], conus_bounds['lon_max'],
        conus_bounds['lat_min'], conus_bounds['lat_max']
    ])

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.1)

    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.05, color='gray')
    gl.xlocator = mticker.MultipleLocator(0.25)
    gl.ylocator = mticker.MultipleLocator(0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.xformatter = mticker.FuncFormatter(lambda lon, pos: f"{int(round(lon))}°" if round(lon) % 5 == 0 else "")
    gl.yformatter = mticker.FuncFormatter(lambda lat, pos: f"{int(round(lat))}°" if round(lat) % 5 == 0 else "")

    scatter = ax.scatter(
        lon_vals,
        lat_vals,
        c=alt_vals,
        cmap='plasma',
        s=10,
        marker='s',
        vmin=altitude_min_ft,
        vmax=altitude_max_ft,
        transform=ccrs.PlateCarree()
    )

    scatter.set_alpha(alpha_vals)

    # Create colorbar with Flight Level labels
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046, aspect=30)
    cbar.set_label('Flight Level', fontsize=22, fontname='Times New Roman')
    tick_vals = np.arange(20000, 40000, 3000)
    cbar.set_ticks(tick_vals)
    cbar.ax.set_xticklabels([f"FL{int(val/100):02d}" for val in tick_vals])

    for t in cbar.ax.get_xticklabels():
        t.set_fontname('Times New Roman')
        t.set_fontsize(14)

    plt.tight_layout()
    plt.savefig(f'Plots/CONUS_ISSR_MAP_time_index_{time_index}.png')


# === RUN SCRIPT ===
fileName = "20241201.nc"
file_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/PROCESSED/RHi_{fileName}"

ds_RHI = xr.open_dataset(file_path)
plot_issr_conus_by_altitude(ds_RHI, time_index=0)
