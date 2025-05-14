import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def feet_to_hPa(feet):
    """
    Approximate conversion from flight level (ft) to pressure level (hPa).
    Uses a simplified ISA (International Standard Atmosphere) model.
    """
    # Approximate pressure in hPa from altitude in meters
    meters = feet * 0.3048
    pressure = 1013.25 * (1 - 2.25577e-5 * meters)**5.25588
    return pressure

def plot_era5_wind(grib_file, flight_level_hPa=None, flight_level_ft=None, skip=10, cmap='coolwarm'):
    """
    Plot ERA-5 wind field at a specified flight level.

    Parameters
    ----------
    grib_file : str
        Path to the ERA-5 GRIB file.
    flight_level_hPa : int, optional
        Target flight level in hPa (default is None).
    flight_level_ft : int, optional
        Target flight level in feet (default is None).
    skip : int, optional
        Step size for plotting wind vectors (to avoid clutter). Default is 10.
    cmap : str, optional
        Colormap for wind speed background. Default is 'coolwarm'.
    """

    # Determine pressure level
    if flight_level_ft is not None:
        flight_level_hPa = feet_to_hPa(flight_level_ft)
        print(f"Converted {flight_level_ft} ft to approx {flight_level_hPa:.1f} hPa.")

    if flight_level_hPa is None:
        raise ValueError("You must specify either flight_level_hPa or flight_level_ft.")

    # Open the GRIB file
    ds = xr.open_dataset(grib_file, engine='cfgrib')

    # Select data at the desired pressure level
    ds_level = ds.sel(isobaricInhPa=flight_level_hPa, method="nearest")

    # Extract variables
    u = ds_level['u'].isel(time=0)
    v = ds_level['v'].isel(time=0)
    lats = ds_level['latitude']
    lons = ds_level['longitude']
    
    # Compute wind speed magnitude
    wind_speed = np.sqrt(u**2 + v**2)
    
    # Adjust longitudes from [0,360] to [-180,180]
    lons_adjusted = xr.where(lons > 180, lons - 360, lons)

    # Create the plot
    fig = plt.figure(figsize=(14,10), dpi=300)  # <--- Set DPI here!
    #ax = plt.axes(projection=ccrs.Robinson())  # <<< Use Robinson projection!
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.PlateCarree())


    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    
    
    # Add latitude and longitude gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    
    #ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    # Plot background wind speed
    wind_plot = ax.pcolormesh(
        lons_adjusted,  # <
        lats, 
        wind_speed, 
        cmap=cmap, 
        shading='auto',
        transform=ccrs.PlateCarree()
    )

    # Add colorbar
    cbar = plt.colorbar(
        wind_plot,
        orientation='horizontal',
        pad=0.05,   # Padding from plot
        fraction=0.046,  # Size of colorbar relative to plot
        aspect=30
    )
    cbar.set_label('Wind Speed (m/s)', fontsize=14)

    # Plot wind vectors
    #ax.quiver(
    #    lons_adjusted.values[::skip],
    #    lats.values[::skip], 
    #    u.values[::skip, ::skip], 
    #    v.values[::skip, ::skip],
    #    scale=700, width=0.0025, headlength=1, color='black', transform=ccrs.PlateCarree()
    #)

    # Title
    plt.title(f"ERA-5 Wind Field at {flight_level_hPa:.1f} hPa", fontsize=18)
    plt.tight_layout()
    plt.savefig("Wind_field.png")

# Example usage
if __name__ == "__main__":
    # Example: plot from 37000 feet
    plot_era5_wind('data.grib', flight_level_ft=37000)
