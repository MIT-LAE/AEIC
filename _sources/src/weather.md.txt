# Weather

The {py:mod}`AEIC.weather` module loads ERA5-style NetCDF files and provides
wind-aware ground speeds along a mission's ground track.

The dataset is read from disk (no pre-processing required) and must contain:

 * variables: temperature `t` [K], eastward wind `u` [m/s], northward wind `v`
   [m/s];
 * coordinates: `pressure_level` [hPa], `latitude`, `longitude`;
 * `valid_time` coordinate: required when the configured `data_resolution` is
   finer than the configured `file_resolution` (i.e., a single file holds
   multiple time steps). In that case `valid_time` is sliced via
   `xarray.Dataset.sel(method='nearest', tolerance=1h)` for the requested
   timestamp. If `data_resolution` equals `file_resolution`, `valid_time` is
   absent or length-1 and no slicing is performed.

During a ground-speed query, altitude is converted to a pressure level using a
standard-atmosphere approximation, winds are interpolated at the requested
longitude/latitude, and those winds are combined with the aircraft heading
derived from the ground track (or an override supplied via `azimuth`).

Example:

```python
import pandas as pd

from AEIC.config import Config
from AEIC.config.weather import TemporalResolution
from AEIC.trajectories.ground_track import GroundTrack
from AEIC.types.spatial import Location
from AEIC.weather import Weather

Config.load()

# Construct a Weather instance pointing at a directory of daily ERA5
# NetCDF files (one file per UTC day, named YYYY-MM-DD.nc by default).
weather = Weather('data/weather', file_resolution=TemporalResolution.DAILY)

# Build a great-circle track between two locations.
track = GroundTrack.great_circle(
    Location(longitude=-71.0, latitude=42.4),   # origin
    Location(longitude=-0.5, latitude=51.5),    # destination
)

ground_speed = weather.get_ground_speed(
    time=pd.Timestamp('2024-09-01 12:00:00', tz='UTC'),
    gt_point=track.location(5000.0),  # 5 km along the track
    altitude=10000.0,                  # meters above sea level
    true_airspeed=230.0,               # m/s
)
```

## Class members

```{eval-rst}
.. autoclass:: AEIC.weather.Weather
   :members:
```
