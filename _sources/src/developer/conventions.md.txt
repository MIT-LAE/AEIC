# Conventions

## Coding standards

 * Follow [PEP 8](https://peps.python.org/pep-0008/).
 * In particular, follow [Python naming
   conventions](https://peps.python.org/pep-0008/#prescriptive-naming-conventions)
   for different entity types, *except* in cases where there is a
   well-established domain convention (e.g., maybe `air_pressure` and
   `air_temperature` style for "normal" pressures and temperatures, but `P3`
   and `T3` style for engine station pressures and temperatures).
 * For common quantity types, follow the naming conventions in the {ref}`data
   dictionary <data-dictionary>` below.
 * Install {ref}`pre-commit <pre-commit>` and ensure that your editor/IDE
   makes problems detected by Ruff visible. (Ideally you would also use a
   static type checker like [PyRight](https://github.com/microsoft/pyright).)

## Units

 * All quantities internal to AEIC code are in SI units.
 * All quantities input in non-SI conventional units are converted to SI units
   immediately.
 * All quantities output in non-SI conventional units are converted from SI
   units only at the point of output.
 * All parameters to public functions are annotated with units information in
   the format "`parameter description [units]`", e.g., "`Rate of climb/descent
   [m/s]`".

(data-dictionary)=
## Data dictionary

For common physical quantities, we use a consistent naming convention for
variables to make it immediately clear what quantities are being considered.
Variables using these conventional names are guaranteed to be in exactly the
units in the table below. "Other units" are used only for input and output
when required.

| Quantity | Units | Variable name | Other units |
| :------- | :---: | :-----------: | :---- |
| Distance | m | (various) | Nautical miles |
| Time | s | (various) | Minutes, hours |
| Mass | kg | (various) | |
| Velocity | m/s | (various) | Knots |
| Ground distance | m | `ground_distance` | Nautical miles |
| Ground speed | m/s | `ground_speed` | Nautical miles |
| Airspeed | m/s | `true_airspeed` | |
| Altitude | m | `altitude` and variants | Feet |
| Flight level | FL (1) | `flight_level` | |
| Aircraft mass | kg | `aircraft_mass` | |
| Thrust | N | `..._thrust` | kN |
| Fuel mass | kg | `fuel_mass` | |
| Fuel flow | kg/s | `fuel_flow` | |
| Rate of climb | m/s | `rate_of_climb` | ft/min? |
| Latitude | degrees (WGS-84) | `latitude` | |
| Longitude | degrees (WGS-84) | `longitude` | |
| Heading/azimuth | degrees (CW from true N) | `heading` | |
| Fuel heating value | J/kg | `...HV` | |

**More to be added here...**

Notes:
1. Pressure altitude in feet, relative to 1013 hPa, divided by 100.
