import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.animation as animation
import matplotlib.ticker as mticker

# import custom function
import utils as util


def plot_issr_conus_by_altitude(df, time_index):
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
    
    print("Pressure levels in df: ", df['isobaricInhPa'].values)
    print("Pressure levels in df ISSR: ", df_issr['isobaricInhPa'].values)
    print("Pressure levels in df CONUS: ", df_conus['isobaricInhPa'].values)
    
    # Convert pressure to altitude in feet using the function
    df_conus['altitude_ft'] = util.pressure_to_altitude_ft(df_conus['isobaricInhPa'])
    
    df['altitude_ft'] = util.pressure_to_altitude_ft(df['isobaricInhPa'])
    
    # Normalzie altitude for alpha scaling (high altitude = more transparent)
    #alt_min = df_conus['altitude_ft'].min()
    #alt_max = df_conus['altitude_ft'].max()
    
    alt_min = df['altitude_ft'].min()
    alt_max = df['altitude_ft'].max()
    
    print("Min altitude: ", alt_min)
    print("Max altitude: ", alt_max)

    alpha_values = 1.0 - (df_conus['altitude_ft'] - alt_min) / (alt_max - alt_min)
    #alpha_values = 1.0 - (df['altitude_ft'] - alt_min) / (alt_max - alt_min)
    alpha_values = np.clip(alpha_values, 0.1, 0.8)  # avoid fully invisible
    
    
    
    # Plot
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([conus_bounds['lon_min'], conus_bounds['lon_max'],
                   conus_bounds['lat_min'], conus_bounds['lat_max']])

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.1)
    #ax.add_feature(cfeature.LAND, edgecolor='black')
    
    # Add latitude and longitude gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.05, color='gray')
    gl.xlocator = mticker.MultipleLocator(0.25)
    gl.ylocator = mticker.MultipleLocator(0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    
    # Only label every 10°
    gl.xformatter = mticker.FuncFormatter(lambda lon, pos: f"{int(round(lon))}°" if round(lon) % 5 == 0 else "")
    gl.yformatter = mticker.FuncFormatter(lambda lat, pos: f"{int(round(lat))}°" if round(lat) % 5 == 0 else "")


    scatter = ax.scatter(
        df_conus['longitude'],
        df_conus['latitude'],
        c=df_conus['altitude_ft'],
        cmap='plasma',
        s=10,
        marker='s',
        alpha=None,
        transform=ccrs.PlateCarree()
    )
    
    # Set alpha per point manually
    scatter.set_alpha(None)  # clear global alpha
    scatter.set_array(df_conus['altitude_ft'].values)
    scatter.set_facecolor(scatter.to_rgba(df_conus['altitude_ft'].values))
    scatter.set_alpha(alpha_values)

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',pad = 0.05, fraction=0.046, aspect=30)
    cbar.set_label('Altitude (ft)', fontsize=22, fontname='Times New Roman')
    #ax.set_title("ISSR Regions over CONUS Colored by Altitude (ft)")
    
    # Set tick label font to Times New Roman
    for t in cbar.ax.get_xticklabels():
        t.set_fontname('Times New Roman')
        t.set_fontsize(14)
        
    plt.tight_layout()
    plt.savefig(f'Plots/CONUS_ISSR_MAP_time_index_{time_index}.png')
    #plt.show()
    
    
    
def plot_pcon_conus_by_altitude(df, time_index):
    """
    Plots Pcon=1 points over the CONUS, colored by altitude (in feet), derived from pressure.

    Parameters:
    - df: DataFrame with 'latitude', 'longitude', 'isobaricInhPa', and 'ISSR_flag'

    Returns:
    - None (displays the plot)
    """
    # Filter for contrail forming reions only only
    df_pcon = df[df['P_contrail'] > 0.80].copy()
    
    #df_pcon = df.copy()

    
    # Filter for CONUS
    conus_bounds = {
        'lat_min': 24.5,
        'lat_max': 49.5,
        'lon_min': -125.0,
        'lon_max': -66.5
    }
    
    

    df_conus = df_pcon[
        (df_pcon['latitude'] >= conus_bounds['lat_min']) &
        (df_pcon['latitude'] <= conus_bounds['lat_max']) &
        (df_pcon['longitude'] >= conus_bounds['lon_min']) &
        (df_pcon['longitude'] <= conus_bounds['lon_max'])
    ].copy()
    
    #print("Pressure levels in df: ", df['isobaricInhPa'].values)
    #print("Pressure levels in df ISSR: ", df_issr['isobaricInhPa'].values)
    #print("Pressure levels in df CONUS: ", df_conus['isobaricInhPa'].values)
    
    # Convert pressure to altitude in feet using the function
    df_conus['altitude_ft'] = util.pressure_to_altitude_ft(df_conus['isobaricInhPa'])
    
    df['altitude_ft'] = util.pressure_to_altitude_ft(df['isobaricInhPa'])
    
    # Normalzie altitude for alpha scaling (high altitude = more transparent)
    #alt_min = df_conus['altitude_ft'].min()
    #alt_max = df_conus['altitude_ft'].max()
    
    alt_min = df['altitude_ft'].min()
    alt_max = df['altitude_ft'].max()
    
    print("Min altitude: ", alt_min)
    print("Max altitude: ", alt_max)

    alpha_values = 1.0 - (df_conus['altitude_ft'] - alt_min) / (alt_max - alt_min)
    #alpha_values = 1.0 - (df['altitude_ft'] - alt_min) / (alt_max - alt_min)
    alpha_values = np.clip(alpha_values, 0.1, 0.8)  # avoid fully invisible
    
    
    
    # Plot
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([conus_bounds['lon_min'], conus_bounds['lon_max'],
                   conus_bounds['lat_min'], conus_bounds['lat_max']])

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.1)
    #ax.add_feature(cfeature.LAND, edgecolor='black')
    
    # Add latitude and longitude gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.05, color='gray')
    gl.xlocator = mticker.MultipleLocator(0.25)
    gl.ylocator = mticker.MultipleLocator(0.25)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'fontname': 'Times New Roman'}
    
    # Only label every 10°
    gl.xformatter = mticker.FuncFormatter(lambda lon, pos: f"{int(round(lon))}°" if round(lon) % 5 == 0 else "")
    gl.yformatter = mticker.FuncFormatter(lambda lat, pos: f"{int(round(lat))}°" if round(lat) % 5 == 0 else "")


    scatter = ax.scatter(
        df_conus['longitude'],
        df_conus['latitude'],
        c=df_conus['altitude_ft'],
        cmap='viridis',
        s=10,
        marker='s',
        alpha=None,
        transform=ccrs.PlateCarree()
    )
    
    # Set alpha per point manually
    scatter.set_alpha(None)  # clear global alpha
    scatter.set_array(df_conus['altitude_ft'].values)
    scatter.set_facecolor(scatter.to_rgba(df_conus['altitude_ft'].values))
    scatter.set_alpha(alpha_values)

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',pad = 0.05, fraction=0.046, aspect=30)
    cbar.set_label('Altitude (ft)', fontsize=22, fontname='Times New Roman')
    #ax.set_title("ISSR Regions over CONUS Colored by Altitude (ft)")
    
    # Set tick label font to Times New Roman
    for t in cbar.ax.get_xticklabels():
        t.set_fontname('Times New Roman')
        t.set_fontsize(14)
        
    plt.tight_layout()
    plt.savefig(f'Plots/CONUS_PCON_MAP_time_index_{time_index}.png')
    #plt.show()

def save_issr_animation(df, filename="issr_animation.mp4", fps=2):
    """
    Saves an animation of ISSR regions over CONUS for each time step, colored by altitude in feet.

    Parameters:
    - df: DataFrame with 'latitude', 'longitude', 'isobaricInhPa', 'ISSR_flag', and 'time'
    - filename: Output filename (e.g., 'issr.mp4' or 'issr.gif')
    - fps: Frames per second
    """
    # Filter ISSR = 1
    df = df[df['ISSR_flag'] == 1].copy()

    # Compute altitude in ft
    df['altitude_ft'] = util.pressure_to_altitude_ft(df['isobaricInhPa'])

    # Filter for CONUS
    df = df[
        (df['latitude'] >= 24.5) & (df['latitude'] <= 49.5) &
        (df['longitude'] >= -125.0) & (df['longitude'] <= -66.5)
    ]

    times = sorted(df['time'].unique())

    # Set up plot
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66.5, 24.5, 49.5])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    scatter = ax.scatter([], [], c=[], cmap='plasma', s=10, marker='s', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Altitude (ft)')
    title = ax.set_title("")

    def update(frame_idx):
        current_time = times[frame_idx]
        df_t = df[df['time'] == current_time]

        alt = df_t['altitude_ft']
        alpha = 1.0 - (alt - alt.min()) / (alt.max() - alt.min())
        alpha = np.clip(alpha, 0.1, 1.0)

        scatter.set_offsets(np.c_[df_t['longitude'], df_t['latitude']])
        scatter.set_array(alt)
        scatter.set_facecolor(scatter.to_rgba(alt))
        scatter.set_alpha(alpha)

        title.set_text(f"ISSR over CONUS — {np.datetime_as_string(current_time, unit='h')}")
        return scatter, title

    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=1000 // fps, blit=False)

    # Save animation
    if filename.endswith(".mp4"):
        ani.save(filename, writer='ffmpeg', fps=fps)
    elif filename.endswith(".gif"):
        ani.save(filename, writer='pillow', fps=fps)
    else:
        raise ValueError("Unsupported file format: use .mp4 or .gif")

    plt.close(fig)
