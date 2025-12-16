Weather
==============

``Weather`` loads ERA5-style NetCDF files and provides wind-aware ground speeds
along a mission's ground track.

The dataset is read from disk (no pre-processing required) and must contain:

* variables: temperature ``t`` [K], eastward wind ``u`` [m/s], northward wind ``v`` [m/s]
* coordinates: ``pressure_level`` [hPa], ``latitude``, ``longitude``
* optional dimension: ``valid_time`` (sliced using ``mission.departure.hour`` if present)

During a ground-speed query, altitude is converted to a pressure level using a
standard-atmosphere approximation, winds are interpolated at the requested
longitude/latitude, and those winds are combined with the aircraft heading
derived from the ground track (or an override supplied via ``azimuth``).

Example::

   mission = Mission(...)
   track = GroundTrack.great_circle(
       mission.origin_position.location,
       mission.destination_position.location,
   )

   weather = Weather('data/weather/sample_weather_subset.nc', mission, track)
   ground_speed = weather.get_ground_speed(
       ground_distance=5000.0,  # meters from departure
       altitude=10000.0,        # meters
       true_airspeed=230.0,      # m/s
   )

Class members
-------------

.. autoclass:: AEIC.weather.weather.Weather
   :members:
