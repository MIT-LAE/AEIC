import json

import trajectory as traj

import utils as util

mission_path = "../missions/sample_missions_10.json"
weather_data = "ERA5/sample.grib"

# Load JSON data
with open(mission_path) as file:
    missions = json.load(file)


# Select a dummy mission for now
first_mission = missions[0]


# Get mission points based on mission def
trajectory = traj.create_dummy_traj(first_mission)

# Unpack trajectory -->
lons = trajectory["lons"]
lats = trajectory["lats"]
alts = trajectory["H"]
tas = trajectory["TAS"]

# Initialize lists -->
gs_list, heading_list, u_list, v_list = [], [], [], []

# Build wind interpolator ->
u_interp, v_interp, meta = util.build_era5_interpolators(weather_data)

# Loop over all mission points to compute heading and ground speed ->
for i in range(len(lons)):
    if i < len(lons) - 1:
        lon_next, lat_next = lons[i + 1], lats[i + 1]
    else:
        # For the last point, repeat previous heading
        lon_next, lat_next = lons[i - 1], lats[i - 1]

    gs, heading, u, v = util.compute_ground_speed(
        lon=lons[i],
        lat=lats[i],
        lon_next=lon_next,
        lat_next=lat_next,
        alt_ft=alts[i],
        tas_kts=tas[i],
        u_interp=u_interp,
        v_interp=v_interp,
    )

    # Append to respective lists for sanity check ->
    gs_list.append(gs)
    heading_list.append(heading)
    u_list.append(u)
    v_list.append(v)

# Append to trajectory ->
trajectory["GS"] = gs_list
trajectory["heading_rad"] = heading_list
trajectory["u_wind"] = u_list
trajectory["v_wind"] = v_list


# Print the first few points
print("\n")
print("#----------------SAMPLE TRAJECTORY (SNAPSHOT) -------------------------#\n")
for lon, lat, tas, gs, alt in zip(trajectory["lons"][:5], trajectory["lats"][:5],\
    trajectory["TAS"][:5], trajectory["GS"][:5], trajectory["H"][:5]):
    print(f"Lon: {lon:.2f}, Lat: {lat:.2f}, TAS: {tas:.2f} kt,\
           GS: {gs:.2f} kt, Alt: {alt:.2f} ft")
print("#-----------------------------------------------------------------------#\n")



#track, heading, drift, tas, u, v, wind_mag = util.get_tas(trajectory, "era5.grib")

#for i in range(5):
#    print(f"Pt {i}: Track={track[i]:.1f}°, Heading={heading[i]:.1f}°,\
# Drift={drift[i]:+.1f}°, "
#          f"TAS={tas[i]:.1f} kt, U={u[i]:.1f}, V={v[i]:.1f}, \
# Wind={wind_mag[i]:.1f} m/s")



#dr.plot_flight_arc(first_mission)

#util.get_flight_track(first_mission)







