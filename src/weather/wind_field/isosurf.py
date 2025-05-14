import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define CONUS bounds (Contiguous United States)
CONUS_LAT_BOUNDS = (24.5, 49.5)    # 24.5°N to 49.5°N
CONUS_LON_BOUNDS = (-125.0, -66.5) # -125°W to -66.5°W

def pressure_to_altitude(pressure_hPa):
    """Convert pressure (hPa) to altitude (meters)."""
    return 44330.0 * (1 - (pressure_hPa / 1013.25)**(1/5.255))

def altitude_to_pressure(alt_meters):
    """Approximate conversion from altitude (meters) to pressure (hPa)."""
    return 1013.25 * (1 - 2.25577e-5 * alt_meters)**5.25588

def plot_3d_scatter_flight_levels(grib_file,
                                  lat_bounds=CONUS_LAT_BOUNDS,
                                  lon_bounds=CONUS_LON_BOUNDS,
                                  time_index=0,
                                  stretch_factor=150,
                                  cmap_name='coolwarm',
                                  marker_alpha=0.5):
    """
    Plot 3D scatter at flight levels every 10,000 ft (FL0, FL100, FL200, FL300, FL400, FL500).
    """

    print("Loading dataset...")
    ds = xr.open_dataset(grib_file, engine='cfgrib')

    # Adjust longitude if necessary
    if np.any(ds['longitude'] > 180):
        lon_min = lon_bounds[0] if lon_bounds[0] >= 0 else lon_bounds[0] + 360
        lon_max = lon_bounds[1] if lon_bounds[1] >= 0 else lon_bounds[1] + 360
    else:
        lon_min, lon_max = lon_bounds

    #print(f"Selecting region: lat {lat_bounds}, lon {lon_bounds}...")

    ds_region = ds.sel(
        latitude=slice(lat_bounds[1], lat_bounds[0]),
        longitude=slice(lon_min, lon_max)
    )

    # Extract u, v components at one time index
    u = ds_region['u'].isel(time=time_index)
    v = ds_region['v'].isel(time=time_index)

    print("Computing wind speed magnitude...")
    wind_speed = np.sqrt(u**2 + v**2)

    # Extract coordinates
    lons = ds_region['longitude'].values
    lats = ds_region['latitude'].values
    pressures_grib = ds_region['isobaricInhPa'].values

    # Define Flight Levels at increments of 10,000 ft
    flight_levels = np.array([0, 100, 200, 300, 400, 500])  # FL0, FL100, ..., FL500
    altitudes_m = flight_levels * 30.48  # 1 FL = 100 ft → 30.48 m
    pressures_target = altitude_to_pressure(altitudes_m)

    #print(f"Target flight levels (FL): {flight_levels}")
    #print(f"Corresponding pressures (hPa): {pressures_target}")

    # Interpolate wind speed at these pressure levels
    wind_interp = wind_speed.interp(isobaricInhPa=pressures_target)

    # Create 3D meshgrids
    lon3d, lat3d, alt3d = np.meshgrid(lons, lats, altitudes_m, indexing='ij')

    # Stretch altitude
    z = alt3d * stretch_factor
    z = z.flatten()

    # Flatten other coordinates
    x = lon3d.flatten()
    y = lat3d.flatten()
    c = wind_interp.values.flatten()

    # Convert longitude from [0, 360] to [-180, 180]
    x = np.where(x > 180, x - 360, x)

    print("Plotting 3D scatter...")

    #fig = plt.figure(figsize=(14, 10))
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')

    
    
    
    # Turn off 3D bounding box (spines and panes)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True


    # Set background and grid
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)

    norm = plt.Normalize(vmin=0, vmax=50)
    
    
    # Scatter plot
    sc = ax.scatter(x, y, z, c=c, norm=norm, cmap=cmap_name, marker='o', s=10, alpha=marker_alpha)

    # Axis labels
    ax.set_xlabel('Longitude (°)', fontsize=18, fontname="Times New Roman")
    ax.set_ylabel('Latitude (°)', fontsize=18, fontname="Times New Roman")
    ax.set_zlabel('Altitude (m)', fontsize=18, fontname="Times New Roman")
    
    ax.xaxis.labelpad = 20   # Push x-axis label farther (default ~10)
    ax.yaxis.labelpad = 20   # Push y-axis label farther
    ax.zaxis.labelpad = 15   # Push z-axis label farther


    z_ticks_stretched = altitudes_m * stretch_factor
    ax.set_zticks(z_ticks_stretched)
    ax.set_zticklabels([f"{int(alt)}" for alt in altitudes_m], fontname="Times New Roman", fontsize=14)

    # Set x/y limits for CONUS
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 49)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    # Set font for axis tick labels
    ax.set_xticklabels(ax.get_xticks(), fontname="Times New Roman", fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontname="Times New Roman", fontsize=14)

    # Tick customization
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(which='major', length=10, width=1.2, direction='in')

    # Colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=30, pad=0.15, orientation='horizontal')
    cbar.set_label('Wind Speed (m/s)', fontsize=22, fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_xticklabels():
        label.set_fontname('Times New Roman')


    # Adjust figure size
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    plt.savefig("wind_stack.png")
    
    plt.show()
    
def plot_u_wind_lat35_profile(grib_file, cmap='coolwarm'):
    """
    Reads GRIB file, converts longitudes to [-180, 180], extracts a latitude slice
    at 35°, and plots a contour map of the U-wind component vs pressure and longitude.
    """
    print("Loading dataset...")
    ds = xr.open_dataset(grib_file, engine='cfgrib')

     # Define color normalization
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=-8, vmax=50)  # for example, from -50 m/s to +50 m/s

    
    # Fix longitudes if needed
    if np.any(ds.longitude > 180):
        ds = ds.assign_coords(longitude=(ds.longitude % 360))
        ds['longitude'] = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.sortby('longitude')

    # Interpolate (or find nearest) to latitude = 35
    lat_val = 35.0
    u_lat_slice = ds['u'].sel(latitude=lat_val, method='nearest')

    # Select time index 0 (assuming single timestep)
    if 'time' in u_lat_slice.dims:
        u_lat_slice = u_lat_slice.isel(time=0)

    # Plot: X = longitude, Y = pressure, Z = u wind
    lons = ds.longitude.values
    pressures = ds['isobaricInhPa'].values
    altitudes = pressure_to_altitude(pressures)
    u_vals = u_lat_slice.values  # shape: (pressure, lon)

    # Create contour plot
    fig1 = plt.figure()
    ax = fig1.gca()

    contour = ax.contourf(lons, altitudes, u_vals, levels=50, norm = norm, cmap=cmap)
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('U Wind Component (m/s)', fontsize=14, fontname="Times New Roman")
    
    cbar.ax.tick_params(labelsize=14)
    
    # Force tick locations
    #cbar.set_ticks(np.linspace(-8, 50, 5))  # Example: 5 ticks at -30, -15, 0, 15, 30
    
    contour.set_clim(-8, 50)


    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax.set_xlabel("Longitude (°)", fontsize=18, fontname="Times New Roman")
    ax.set_ylabel("Altitude (m)", fontsize=18, fontname="Times New Roman")
    #ax.set_title("U-Wind Profile at Latitude 35°", fontsize=16, fontname="Times New Roman")
    
    # Customize plot spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        
    # Modify axis tick properties
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(which='major', length=10, width=1.2, direction='in')
    ax.tick_params(which='minor', length=5, width=1.2, direction='in')
    
    # Set x and y limits
    #plt.xlim([0.50, 0.85])
    plt.ylim([0, 15240])
    
    # Adjust figure size
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.tight_layout()
    
    plt.savefig("U_profile.png")

    plt.show()

def plot_v_wind_lat35_profile(grib_file, cmap='coolwarm'):
    """
    Reads GRIB file, converts longitudes to [-180, 180], extracts a latitude slice
    at 35°, and plots a contour map of the U-wind component vs pressure and longitude.
    """
    print("Loading dataset...")
    ds = xr.open_dataset(grib_file, engine='cfgrib')

     # Define color normalization
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=-3, vmax=30)  # for example, from -50 m/s to +50 m/s

    
    # Fix longitudes if needed
    if np.any(ds.longitude > 180):
        ds = ds.assign_coords(longitude=(ds.longitude % 360))
        ds['longitude'] = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.sortby('longitude')

    # Interpolate (or find nearest) to latitude = 35
    lat_val = 35.0
    v_lat_slice = ds['v'].sel(latitude=lat_val, method='nearest')

    # Select time index 0 (assuming single timestep)
    if 'time' in v_lat_slice.dims:
        v_lat_slice = v_lat_slice.isel(time=0)

    # Plot: X = longitude, Y = pressure, Z = u wind
    lons = ds.longitude.values
    pressures = ds['isobaricInhPa'].values
    altitudes = pressure_to_altitude(pressures)
    v_vals = v_lat_slice.values  # shape: (pressure, lon)

    # Create contour plot
    fig1 = plt.figure()
    ax = fig1.gca()

    contour = ax.contourf(lons, altitudes, v_vals, levels=50, norm = norm, cmap=cmap)
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('V Wind Component (m/s)', fontsize=14, fontname="Times New Roman")
    
    cbar.ax.tick_params(labelsize=14)
    
    # Force tick locations
    #cbar.set_ticks(np.linspace(-8, 50, 5))  # Example: 5 ticks at -30, -15, 0, 15, 30
    
    #contour.set_clim(-8, 50)


    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax.set_xlabel("Longitude (°)", fontsize=18, fontname="Times New Roman")
    ax.set_ylabel("Altitude (m)", fontsize=18, fontname="Times New Roman")
    #ax.set_title("U-Wind Profile at Latitude 35°", fontsize=16, fontname="Times New Roman")
    
    # Customize plot spines
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        
    # Modify axis tick properties
    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)
    
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(which='major', length=10, width=1.2, direction='in')
    ax.tick_params(which='minor', length=5, width=1.2, direction='in')
    
    # Set x and y limits
    #plt.xlim([0.50, 0.85])
    plt.ylim([0, 15240])
    
    # Adjust figure size
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.tight_layout()
    
    plt.savefig("V_profile.png")

    plt.show()


# Example usage
if __name__ == "__main__":
    #plot_3d_scatter_flight_levels(
    #    grib_file='data.grib',
    #    time_index=0,
    #    stretch_factor=200,      # Stretch Z axis for visibility
    #    cmap_name='coolwarm',     # Colormap for wind speed
    #    marker_alpha=0.08        # Set marker transparency
    #)
    
    
    #plot_u_wind_lat35_profile("data.grib")
    
    plot_v_wind_lat35_profile("data.grib")

