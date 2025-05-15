import utils as util
import postProcess as proc
# working data set info: specific humidity (q), temperature (T), Pressure (p) X 100

weather_grib = 'data/01-01-23.grib'


# Dataset overview 
util.inspect_grib_fields(weather_grib)

print("PRE-PROCESSING DATA...")

# Change longitude to [-180, 180], compute ISSR binary
#weather = util.preprocess_beta(weather_grib)

levels = [225, 200]  # hPa
#weather = util.preprocess_by_levels(weather_grib,levels)

weather = util.preprocess_by_time_slice(weather_grib)

print(weather.head())

#proc.plot_issr_conus_by_altitude(weather)

#proc.save_issr_animation(weather, filename="issr_animation.mp4", fps=2)