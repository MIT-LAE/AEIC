import utils as util

# working data set info: specific humidity (q), temperature (T), Pressure (p) X 100

weather_grib = 'data/01-01-25.grib'


# Dataset overview 
util.inspect_grib_fields(weather_grib)

# Change longitude to [-180, 180], compute RHi
weather = util.preprocess(weather_grib)


