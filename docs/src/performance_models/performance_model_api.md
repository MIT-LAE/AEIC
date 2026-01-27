# Performance model API

Performance model classes are all Pydantic models derived from
{py:class}`BasePerformanceModel
<AEIC.performance.models.BasePerformanceModel>`. This is an abstract base
class that includes data common to all performance model types (aircraft name
and class, maximum altitude and payload, number of engines, optional APU
information and optional LTO and speed information) and that defines the
performance model API. The legacy table-based performance model is represented
by the {py:class}`LegacyPerformanceModel
<AEIC.performance.models.LegacyPerformanceModel>` class. This includes a
performance table represented by the {py:class}`PerformanceTable
<AEIC.performance.models.legacy.PerformanceTable>` class which performs
subsetting and interpolation within the input data.

## Loading performance models

Performance models can be loaded from TOML files. A top-level `model_type`
string field is used to distinguish between different types of performance
model and a {py:class}`PerformanceModel
<AEIC.performance.models.PerformanceModel>` wrapper class is used to enable
this: there is a {py:meth}`load
<AEIC.performance.models.PerformanceModel.load>` class method with a
polymorphic return type, that takes a path to a TOML file containing a
performance model definition and returns an instance of the correct
performance model class based on the `model_type` field. For the current
legacy-based performance models, use `model_type = "legacy"`.

## Performance evaluation

The fundamental operation of a performance model is to take an aircraft state,
(represented by a value of the {py:class}`AircraftState
<AEIC.performance.types.AircraftState>` type) and a flight rule (see below)
and to return aircraft performance data (a value of the {py:class}`Performance
<AEIC.performance.types.Performance>` type). This is achieved by calling the
{py:meth}`evaluate <AEIC.performance.models.BasePerformanceModel.evaluate>`
method on a performance model instance.

Aircraft state includes altitude, aircraft mass and optionally a target true
airspeed and/or rate of climb/descent. Some performance models may make use of
the optional values, some may not. The legacy table-based performance model
does not: all return values from the legacy model depend only on aircraft
altitude, aircraft mass and a simple climb/cruise/descent flight rule.

The performance data returned from the {py:meth}`evaluate
<AEIC.performance.models.BasePerformanceModel.evaluate>` method contains
actual achievable true airspeed, actual achievable rate of climb/descent and
fuel flow rate.

Different types of performance model may implement different kinds of
potential flight rules. For example, a performance model might support
specifying that climb and descent phases should be conducted at constant rate
of climb, constant calibrated airspeed, or constant Mach number, and the
cruise phase at constant altitude or constant lift coefficient. The simplest
flight rules, as used by the {py:class}`LegacyPerformanceModel
<AEIC.performance.models.LegacyPerformanceModel>` class, are represented by
the {py:class}`SimpleFlightRules <AEIC.performance.types.SimpleFlightRules>`
class, which simply specifies "climb", "cruise", or "descend".

## Usage example

```python
from AEIC.config import Config, config
from AEIC.performance import AircraftState, SimpleFlightRules
from AEIC.performance.models import PerformanceModel, LegacyPerformanceModel
from AEIC.utils.units import FL_TO_METERS
from AEIC.emissions.emission import Emission

# Load default AEIC configuration.
Config.load();

# Load (table-based legacy) performance model.
model = PerformanceModel.load(
    config.file_location('performance/sample_performance_model.toml')
)
assert isinstance(model, LegacyPerformanceModel)

# Create aircraft state.
state = AircraftState(altitude=350*FL_TO_METERS, aircraft_mass=70000)

# Evaluate performance model for state and flight rules.
perf = model.evaluate(state, SimpleFlightRules.CLIMB)

print(perf)
# Result:
# Performance(
#     true_airspeed=237.4846304442151,
#     rate_of_climb=33.69675733337415,
#     fuel_flow=1.5102911008506104
# )

# Pass to trajectory or emissions builders.
emitter = Emission(perf)
```

## Performance model members

After a performance model instance is created (of any type derived from
 {py:class}`BasePerformanceModel
 <AEIC.performance.models.BasePerformanceModel>`), as well as being set up for
 calls to the {py:meth}`evaluate
 <AEIC.performance.models.BasePerformanceModel.evaluate>` method, the instance
 also contains:

- Basic information about the performance model: aircraft name and class,
  number of engines, maximum altitude and payload.
- {py:attr}`lto_performance
  <AEIC.performance.models.BasePerformanceModel.lto_performance>`: modal
  thrust settings, fuel flows, and emission indices taken from the performance
  file.
- {py:attr}`apu <AEIC.performance.models.BasePerformanceModel.apu>`:
  auxiliary-power-unit properties resolved from `engines/APU_data.toml` using
  the `apu_name` specified in the performance file.
- {py:attr}`speeds <AEIC.performance.models.BasePerformanceModel.speeds>`:
  cruise speed data.

## API reference

```{eval-rst}
.. autoclass:: AEIC.performance.models.BasePerformanceModel
   :members:
   :exclude-members: apu_name, load_apu_data, model_config
```

```{eval-rst}
.. autoclass:: AEIC.performance.models.PerformanceModel
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.performance.types.AircraftState
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.performance.types.Performance
   :members:
```

```{eval-rst}
.. autoenum:: AEIC.performance.types.SimpleFlightRules
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.performance.LTOPerformance
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoenum:: AEIC.performance.types.LTOThrustMode
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.performance.types.LTOModeData
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.performance.utils.apu.APU
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.performance.types.Speeds
   :members:
   :exclude-members: model_config
```

```{eval-rst}
.. autoclass:: AEIC.performance.types.SpeedData
   :members:
   :exclude-members: model_config
```
