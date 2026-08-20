# BADA methods

Support for the [Base of Aircraft
Data](https://www.eurocontrol.int/model/bada) revision 3.0 (BADA-3) and custom
performance data structured in the same way is included in AEIC. Below are the
various dataclasses, methods, and helper functions that can be used to
evaluate or manipulate BADA-3 formatted data.

## Engine and fuel burn models

```{eval-rst}
.. autoclass:: AEIC.BADA.model.Bada3EngineModel
    :members:
```

```{eval-rst}
.. autoclass:: AEIC.BADA.model.Bada3JetEngineModel
    :members:
```

```{eval-rst}
.. autoclass:: AEIC.BADA.model.Bada3TurbopropEngineModel
    :members:
```

```{eval-rst}
.. autoclass:: AEIC.BADA.model.Bada3PistonEngineModel
    :members:
```

```{eval-rst}
.. autoclass:: AEIC.BADA.model.Bada3FuelBurnModel
    :members:
```

## Aircraft parameters

```{eval-rst}
.. autoclass:: AEIC.BADA.aircraft_parameters.Bada3AircraftParameters
    :members:
```

## Fuel burn base classes

```{eval-rst}
.. autoclass:: AEIC.BADA.fuel_burn_base.BaseAircraftParameters
    :members:
```

```{eval-rst}
.. autoclass:: AEIC.BADA.fuel_burn_base.BaseFuelBurnModel
    :members:
```

## Helper functions

```{eval-rst}
.. automodule:: AEIC.BADA.helper_functions
    :members:
```
