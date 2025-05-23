import json
import draw as dr
import utils as util
import trajectory as traj

mission_path = "../missions/sample_missions_10.json"

# Load JSON data
with open(mission_path, 'r') as file:
    missions = json.load(file)
    

# Select a dummy mission for npw
first_mission = missions[0]


# Get mission points based on mission def
trajectory = traj.get_mission_points(first_mission)

# Print the first few points
for lon, lat, gs, alt in zip(trajectory["lons"][:5], trajectory["lats"][:5], trajectory["GS"][:5], trajectory["H"][:5]):
    print(f"Lon: {lon:.2f}, Lat: {lat:.2f}, GS: {gs} kt, Alt: {alt} ft")


#util.get_flight_track(trajectory)

weather_data = "ERA5/sample.grib"

#u, v, windmag = util.get_wind_at_points(trajectory, "ERA5/sample.grib")


track, heading, drift, tas, u, v, wind_mag = util.get_tas(trajectory, "era5.grib")

for i in range(5):
    print(f"Pt {i}: Track={track[i]:.1f}°, Heading={heading[i]:.1f}°, Drift={drift[i]:+.1f}°, "
          f"TAS={tas[i]:.1f} kt, U={u[i]:.1f}, V={v[i]:.1f}, Wind={wind_mag[i]:.1f} m/s")



#dr.plot_flight_arc(first_mission)

#util.get_flight_track(first_mission)







