# Legacy performance model

The legacy performance model is a table-based model intended to replicate the
behavior of the performance model in the AEIC v2 Matlab code. It is
represented by the {py:class}`LegacyPerformanceModel
<AEIC.performance.models.LegacyPerformanceModel>` class, and is currently the
only performance model that can be used for simulations.

## Input file format

The model-specific part of the TOML file defining a legacy performance model
lives in the `flight_performance` section, which has the following fields:

 * `cols` defines the columns in the performance table - these will normally
   be `fuel_flow` (fuel flow rate [kg/s]), `fl` (altitude as flight level),
   `tas` (true airspeed [m/s]), `rocd` (rate of climb/descent [m/s]) and
   `mass` (aircraft mass [kg]);
 * `data` provides the performance table data, as a list of table rows, each
   row being a list of column values in the order given in the `cols` field.

The contents of this section of the input file is represented internally by a
value of the {py:class}`PerformanceTableInput
<AEIC.performance.models.base.PerformanceTableInput>` class. This class is
shared with the other table-based model types, see the [performance model
API](performance_model_api.md).

```{admonition} Question
Should we make this format better? For a TOML file, this setup might be the
best that we can do.
```

## Operating empty mass

The legacy model derives the aircraft empty mass from the performance tables
using the BADA 3 rule `OEM = min_mass / 1.2`. It ignores an
`operating_empty_mass_kg` field, and warns if a file sets one.

## Performance table

Within the {py:class}`LegacyPerformanceModel
<AEIC.performance.models.LegacyPerformanceModel>` class, the performance data
is stored in a private `_performance_table` attribute of type
{py:class}`PerformanceTable
<AEIC.performance.models.legacy.PerformanceTable>`. This is created from the
input data using the {py:meth}`from_input
<AEIC.performance.models.legacy.PerformanceTable.from_input>` class method:
this provides the link between the TOML input data and the interpolation
machinery used within the performance model.

The legacy performance model imposes certain limitations on the structure of
the performance table data:

 * The table is divided into three separate segments, one each for climb
   ({math}`\mathrm{ROCD} > 0`), cruise ({math}`\mathrm{ROCD} \approx 0`) and
   descent ({math}`\mathrm{ROCD} < 0`).
 * Within each table segment, the performance table data is dense in flight
   level and aircraft mass, in the sense that there is exactly one table for
   for each (flight level, aircraft mass) combination. The data thus defines a
   complete table for bilinear interpolation in flight level and mass.

This structure ensures that, for each relevant flight phase (climb, cruise or
descent), fuel flow, achievable true airspeed and achievable rate of
climb/descent are dependent only on altitude and aircraft mass.

## Performance evaluation

The {py:meth}`evaluate_impl
<AEIC.performance.models.LegacyPerformanceModel.evaluate_impl>` method
that calculates performance data for a given aircraft state simply selects the
relevant segment of the performance table data and does bilinear interpolation
in flight level and aircraft mass. The interpolation is performed by an
{py:class}`Interpolator <AEIC.performance.models.legacy.Interpolator>` helper
class, instances of which are created lazily for each flight phas as needed
(once only for any performance model instance).

## API reference

```{eval-rst}
.. autoclass:: AEIC.performance.models.LegacyPerformanceModel
   :members:
   :exclude-members: model_config, model_type, validate_pm
```

```{eval-rst}
.. autoclass:: AEIC.performance.models.legacy.PerformanceTable
   :members:
   :private-members: _interpolators
```

```{eval-rst}
.. autoenum:: AEIC.performance.models.legacy.ROCDFilter
```

```{eval-rst}
.. autoclass:: AEIC.performance.models.legacy.Interpolator
   :members:
   :special-members: __call__
```
