import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import utils as util

def plot_issr_conus_by_altitude(df):
    """
    Plots ISSR=1 points over the CONUS, colored by altitude (in feet), derived from pressure.

    Parameters:
    - df: DataFrame with 'latitude', 'longitude', 'isobaricInhPa', and 'ISSR_flag'

    Returns:
    - None (displays the plot)
    """
    # Filter for ISSR only
    df_issr = df[df['ISSR_flag'] == 1].copy()

    # Filter for CONUS
    conus_bounds = {
        'lat_min': 24.5,
        'lat_max': 49.5,
        'lon_min': -125.0,
        'lon_max': -66.5
    }

    df_conus = df_issr[
        (df_issr['latitude'] >= conus_bounds['lat_min']) &
        (df_issr['latitude'] <= conus_bounds['lat_max']) &
        (df_issr['longitude'] >= conus_bounds['lon_min']) &
        (df_issr['longitude'] <= conus_bounds['lon_max'])
    ].copy()

    # Convert pressure to altitude in feet using the function
    df_conus['altitude_ft'] = util.pressure_to_altitude_ft(df_conus['isobaricInhPa'])

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([conus_bounds['lon_min'], conus_bounds['lon_max'],
                   conus_bounds['lat_min'], conus_bounds['lat_max']])

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    scatter = ax.scatter(
        df_conus['longitude'],
        df_conus['latitude'],
        c=df_conus['altitude_ft'],
        cmap='plasma',
        s=10,
        marker='s',
        alpha=0.7,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Altitude (ft)')
    ax.set_title("ISSR Regions over CONUS Colored by Altitude (ft)")
    plt.tight_layout()
    plt.show()
