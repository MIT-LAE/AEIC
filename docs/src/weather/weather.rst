Weather Module
==============

``Weather`` builds regridded weather slices along a mission's ground track. It
takes an :class:`xarray.Dataset` (or a path to one) containing weather variables
temperature ``t`` [K], Eastward component of wind ``u`` [m/s], Northward component
of wind ``v`` [m/s], and pressure coordinate ``p`` [Pa]. The departure hour of the
mission selects the appropriate ``valid_time``/``time`` index, and the data is
sampled along the ``ground_track`` before being interpolated to evenly spaced
flight levels.

Class members
-------------

.. autoclass:: AEIC.weather.weather.Weather
   :members:
