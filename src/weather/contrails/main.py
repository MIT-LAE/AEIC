import utils as util
import postProcess as proc
# working data set info: specific humidity (q), temperature (T), Pressure (p) X 100

weather_grib = 'data/12-30-22.grib'

# Dataset overview 
util.inspect_grib_fields(weather_grib)

print("PRE-PROCESSING DATA...")

# Change longitude to [-180, 180], compute ISSR binary
#weather = util.preprocess_beta(weather_grib)

#weather = util.preprocess_by_levels(weather_grib,levels)

# Loop over 24 hourly time indices
for time_index in range(1,24):
    print("Processing time index: {time_index}")

    try:
        
        # Pre-process data for a single time slice
        weather = util.preprocess_single_time_slice(weather_grib, time_index)
        
        # Compute 1D CDF using RHi
        weather = util.compute_contrail_probability(weather,sigma_rhi = 3.0)
        
        


#print(weather.head())

        # Plot ISSR contour onver CONUS
        #proc.plot_issr_conus_by_altitude(weather, time_index )
        proc.plot_pcon_conus_by_altitude(weather, time_index )
    except Exception as e:
        print(f"Failed at time index {time_index}: {e}")
#proc.save_issr_animation(weather, filename="issr_animation.mp4", fps=2)