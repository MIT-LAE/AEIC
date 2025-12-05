# Utilities

## Custom types

Small custom types are used to represent "locations" (2-D) and "positions"
(3-D) and some time values.

```{eval-rst}
.. autoclass:: utils.types.Location
    :members:
```

```{eval-rst}
.. autoclass:: utils.types.Position
    :members:
```

```{eval-rst}
.. autoclass:: utils.types.DayOfWeek
    :members:
```

```{eval-rst}
.. autoclass:: utils.types.TimeOfDay
    :members:
```

## Airport handling

Airport data is downloaded from the [OurAirports
project](https://ourairports.com/data/). Data for some missing airports is
added from a supplemental CSV file. The canonical way to use this module is as
follows:

```python
import utils.airports as airports

...

ap = airports.airport('LHR')
print(f'{ap.name} @ {ap.elevation} MSL')
```

```{eval-rst}
.. automodule:: utils.airports
    :members:
    :exclude-members: CountriesData, AirportsData
```

## Unit conversion

Unit conversions are done with simple multiplying factors with a common
`x_TO_y` naming pattern.

```{eval-rst}
.. NOTE::
   The flight level conversions here are based on pressure altitudes, i.e.,
   they simply scale given altitudes by 100 feet intervals.
```

```{eval-rst}
.. automodule:: utils.units
    :members:
```

## Geospatial utilities

Great circle distance and azimuth calculations are needed in several places.
These can all be done using the single `pyproj.Geod` instance defined in this
module.

```{eval-rst}
.. automodule:: utils.spatial
    :members:
```

## File handling functions

```{eval-rst}
.. automodule:: utils.files
    :members:
```

## Weather-related utility functions

```{eval-rst}
.. automodule:: utils.weather_utils
    :members:
```

## Standard atmosphere

```{eval-rst}
.. automodule:: utils.standard_atmosphere
    :members:
```

## Standard fuel

```{eval-rst}
.. automodule:: utils.standard_fuel
    :members:
```

## Miscellaneous utility functions

```{eval-rst}
.. automodule:: utils.helpers
    :members:
```
