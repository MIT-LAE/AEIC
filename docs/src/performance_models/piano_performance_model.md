# PIANO performance model

The {py:class}`PianoPerformanceModel
<AEIC.performance.models.PianoPerformanceModel>` class represents a
performance model built from PIANO climb, cruise and descent exports. Files
for this model type use `model_type = "piano"`.

The source data is read by {py:mod}`AEIC.parsers.piano_reader`, which is
documented in [PIANO reader](../parsers/piano_reader.md).

## Key differences from the legacy model

 * Cruise is swept over Mach number as well as flight level and mass, whereas
   the legacy model is only swept over flight level and mass. Cruise
   evaluation therefore needs a Mach number, which is set by the
   {py:class}`PianoConstantMachCruise
   <AEIC.performance.types.PianoConstantMachCruise>` flight rule.
 * In the climb and descent tables, `mass` is the initial mass of a PIANO
   climb or descent profile, not the current mass of the aircraft. See
   [Climb and descent](#climb-and-descent).
 * The tables are ragged: each mass, flight level or Mach number can have its
   own range of values on the other axes. See
   [Ragged interpolation](#ragged-interpolation).
 * Each phase table carries thrust, drag and trajectory columns in addition to
   fuel flow, mass, flight level, TAS, and rate of climb/descent.
 * There is no fixed cruise speed data because Mach number is swept.
   {py:attr}`speeds <AEIC.performance.models.BasePerformanceModel.speeds>` has
   a `climb` and a `descent` entry but no `cruise` entry.
 * Two extra tables come along with the phase tables:
   `cruise_reference_mach` and `descent_idle_thrust`.

## Operating empty mass

`operating_empty_mass_kg` is a required field for a PIANO performance model.
PIANO exports contain no operating empty mass, and we cannot apply the BADA 3
rule to estimate it. This field must be passed when creating the
performance model file.

## Hold fuel flow

Similarly, `hold_flight_performance` is a required table for a PIANO
performance model. It has one entry, `fuel_flow`, which is the
fuel flow in a hold at a reference condition, in kg/s. It is passed
when creating the performance model file in lb/h with `--hold-fuel-flow-lbperh`.

The reference condition is the initial fuel flow for a hold at 3000 ft with 'normal landing'
mass (as estimated by PIANO) minimizing drag. Holding at 3000 ft mirrors the
`LegacyPerformanceModel` / BADA approach of estimating hold fuel flow from cruise
performance at FL30 (minimum altitude & minimum mass of legacy cruise table).

## Performance tables

The three phase tables are {py:class}`PerformanceTableInput
<AEIC.performance.models.base.PerformanceTableInput>` values, so each must
provide the `fuel_flow`, `fl`, `tas`, `rocd` and `mass` columns.

Row ordering in a generated file is by `(mass, fl)` for climb and descent, and
by `(mass, fl, mach)` for cruise.

### Climb table

| Column | Unit | Meaning |
|--------|------|---------|
| `fl` | flight level | Altitude. |
| `mass` | kg | Initial mass of the block. |
| `tas` | m/s | True airspeed, derived from the airspeed schedule. |
| `rocd` | m/s | Rate of climb. |
| `fuel_flow` | kg/s | Fuel flow at the row, central difference of burn over time. |
| `time` | s | Cumulative from the start of the phase. |
| `distance` | m | Cumulative from the start of the phase. |
| `burn` | kg | Cumulative from the start of the phase. |
| `fn_per_engine` | N | Net thrust per engine. |
| `drag` | N | Drag. |

If PIANO halted a climb above the last row of its block, the last row of that
block can be an extrapolated row at the halt altitude. See the
[PIANO reader](../parsers/piano_reader.md) for the method.

### Cruise table

| Column | Unit | Meaning |
|--------|------|---------|
| `fl` | flight level | Altitude. |
| `mass` | kg | Aircraft mass. |
| `mach` | - | Mach number of this point of the sweep. |
| `tas` | m/s | True airspeed. |
| `cas` | m/s | Calibrated airspeed. |
| `rocd` | m/s | Always zero: cruise is level flight. |
| `fuel_flow` | kg/s | Fuel flow. |
| `drag` | N | Drag. |
| `mcr_pct` | percent | Percent of maximum cruise thrust. |
| `lift_to_drag` | - | Lift-to-drag ratio. |
| `sfc` | kg/(N.s) | Specific fuel consumption. |
| `sar` | m/kg | Specific air range. |
| `mcl_avail_per_engine` | N | Maximum climb thrust available per engine. |
| `rocd_mcl_fix_mach` | m/s | Rate of climb at maximum climb thrust, fixed Mach. |
| `rocd_mcl_fix_cas` | m/s | Rate of climb at maximum climb thrust, fixed CAS. |

The two `rocd_mcl_*` columns keep the sign PIANO reports, such that they can be
negative where the aircraft cannot keep steady level flight even at the max climb
rating setting.

Due to rounding in the PIANO output, a `(fl, mass, Mach)` triplet might be
the same for a row coming from the PIANO Mach sweep, and a row coming from
one of the operating points but with slightly differing performance values.
In this case, the operating point value is kept over the Mach sweep.

### Descent table

| Column | Unit | Meaning |
|--------|------|---------|
| `fl` | flight level | Altitude. |
| `mass` | kg | Initial mass of the block. |
| `tas` | m/s | True airspeed, derived from the airspeed schedule. |
| `rocd` | m/s | Rate of descent, negative. |
| `fuel_flow` | kg/s | Fuel flow at the row, central difference of burn over time. |
| `time` | s | Cumulative from the start of the phase. |
| `distance` | m | Cumulative from the start of the phase. |
| `burn` | kg | Cumulative from the start of the phase. |
| `fn_per_engine` | N | Net thrust per engine. |

### Cruise reference Mach table

| Column | Unit | Meaning |
|--------|------|---------|
| `fl` | flight level | Altitude. |
| `mass` | kg | Aircraft mass. |
| `max_sar` | - | Mach at maximum specific air range. |
| `sar_99` | - | Mach at 99% of maximum specific air range. |
| `max_lim` | - | Maximum limiting Mach. |

Every (flight level, mass) group is written. If the PIANO file does not have
one of the three reference Mach numbers for a group, that column is written as
`nan`.

### Descent idle thrust table

| Column | Unit | Meaning |
|--------|------|---------|
| `mass` | kg | Aircraft mass. |
| `idle_thrust_altitude` | m | Altitude below which idle thrust is used. |

This table can be empty, meaning that no descent block gave an idle
thrust altitude.

## Input file format

The following shows the structure of a PIANO performance model file. Field
order comes from `PIANO_WRITE_SPEC` in
{py:mod}`AEIC.performance.model_builder`, which is what
{command}`make-performance-model` writes.

```
# Performance model type (one of: legacy, bada, tasopt, piano).
model_type = "piano"

# ==============================================================================
#
#  COMMON FIELDS
#
# Fields common to all performance model types.

aircraft_name = "..."
aircraft_class = "narrow" # wide, narrow, small, freight
ISA_offset = 0
maximum_altitude_ft = 41000
maximum_payload_kg = 22422
number_of_engines = 2 # Number of engines
operating_empty_mass_kg = 41413.0 # kg
APU_name = "APU 131-9" # None: APU emissions not calculated

# ------------------------------------------------------------------------------
#
# Speed data
#
# There is no [speeds.cruise] table: PIANO sweeps cruise Mach number, so a
# single cruise speed is meaningless.

[speeds.climb]
cas_low = 0.0
cas_high = 0.0
mach = 0.0
crossover_altitude_m = 0.0

[speeds.descent]
cas_low = 0.0
cas_high = 0.0
mach = 0.0
crossover_altitude_m = 0.0

# ------------------------------------------------------------------------------
#
# LTO data
#

[LTO_performance]
source = "EDB"
ICAO_UID = "01P11CM121" # Add UID for EDB data
rated_thrust = 0.0

[LTO_performance.mode_data.idle]
thrust_frac = 0.0
fuel_kgs    = 0.0
EI_NOx      = 0.0
EI_HC       = 0.0
EI_CO       = 0.0

# ... approach, climb and takeoff sub-tables. ...

# ==============================================================================
#
#  MODEL-TYPE SPECIFIC FIELDS
#

# ------------------------------------------------------------------------------
#
# Performance table data.
#

[climb_flight_performance]
cols = [
  "fl",  # Flight levels
  "mass",  # kg
  "tas",  # m/s
  "rocd",  # m/s
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "time",  # s - cumulative from start of phase
  "distance",  # m - cumulative from start of phase
  "burn",  # kg - cumulative from start of phase
  "fn_per_engine",  # N - net thrust per engine
  "drag"  # N
]

data = [
  ...
]

[cruise_flight_performance]
cols = [
  "fl",  # Flight levels
  "mass",  # kg
  "mach",  # Mach number
  "tas",  # m/s
  "cas",  # m/s
  "rocd",  # m/s
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "drag",  # N
  "mcr_pct",  # percent of maximum cruise thrust
  "lift_to_drag",  # Lift-to-drag ratio
  "sfc",  # kg/(N.s) - specific fuel consumption
  "sar",  # m/kg - specific air range
  "mcl_avail_per_engine",  # N - maximum climb thrust available per engine
  "rocd_mcl_fix_mach",  # m/s - at maximum climb thrust, fixed Mach
  "rocd_mcl_fix_cas"  # m/s - at maximum climb thrust, fixed CAS
]

data = [
  ...
]

[descent_flight_performance]
cols = [
  "fl",  # Flight levels
  "mass",  # kg
  "tas",  # m/s
  "rocd",  # m/s
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "time",  # s - cumulative from start of phase
  "distance",  # m - cumulative from start of phase
  "burn",  # kg - cumulative from start of phase
  "fn_per_engine"  # N - net thrust per engine
]

data = [
  ...
]

[cruise_reference_mach]
cols = [
  "fl",  # Flight levels
  "mass",  # kg
  "max_sar",  # Mach at maximum specific air range
  "sar_99",  # Mach at 99% of maximum specific air range
  "max_lim"  # Maximum limiting Mach
]

data = [
  ...
]

[descent_idle_thrust]
cols = [
  "mass",  # kg
  "idle_thrust_altitude"  # m - altitude below which idle thrust is used
]

data = []

# ------------------------------------------------------------------------------
#
# Hold data
#

[hold_flight_performance]
fuel_flow = 0.0 # kg/s - total aircraft fuel flow in hold
```

## Performance evaluation

The {py:meth}`evaluate_impl
<AEIC.performance.models.PianoPerformanceModel.evaluate_impl>` method
computes performance data for a given aircraft state by linear interpolation
of the tabulated data. It returns the true airspeed, the rate of climb/descent
and the fuel flow.

### Flight rules

The model accepts {py:data}`PianoFlightRules
<AEIC.performance.models.piano.PianoFlightRules>`, which is
`SimpleFlightRules | PianoConstantMachCruise`. The flight rules select the
table and the interpolation axes:

| Flight rules | Table | Axes, outer to inner |
|--------------|-------|----------------------|
| `SimpleFlightRules.CLIMB` | Climb | `(mass, fl)`, with `mass` the initial mass |
| `PianoConstantMachCruise(mach=...)` | Cruise | `(fl, mass, mach)` |
| `SimpleFlightRules.DESCEND` | Descent | `(mass, fl)`, with `mass` the initial mass |

PIANO cruise performance depends on the Mach number, so cruise requires
{py:class}`PianoConstantMachCruise
<AEIC.performance.types.PianoConstantMachCruise>`. which sets the Mach number to
cruise at. Passing `SimpleFlightRules.CRUISE` raises `ValueError`.

### Ragged interpolation

PIANO tables are ragged. There is no guarantee that the points valid at one
node of an axis are also valid at another node. For example:

 * each climb profile has its own ceiling, so the flight levels differ from
   one initial mass to the next;
 * at a high flight level, the heaviest masses can be missing from the
   cruise table, because the aircraft is too heavy to cruise there;
 * for each `(fl, mass)` pair, the cruise Mach sweep has its own range.

For interpolation of the raw data, the model uses
{py:class}`RaggedInterpolator
<AEIC.performance.models.ragged.RaggedInterpolator>` through a
{py:class}`RaggedPerformanceTable
<AEIC.performance.models.ragged.RaggedPerformanceTable>` for each flight
phase.

{py:class}`RaggedInterpolator
<AEIC.performance.models.ragged.RaggedInterpolator>` stores the table as a
tree, with one level for each axis.
Each node of an outer axis has its own sorted values on the next axis and
interpolation is nested and linear.

For a query at FL355 in the cruise table, for example, the interpolator
evaluates the `(mass, mach)` subtables at FL350 and at FL360. It then
interpolates between the two results. A query is valid only if every node
that the interpolation uses bounds the query.

A mass of `'min'` or `'max'` resolves to the lightest or heaviest mass that
is valid at the query flight level and the other coordinates. In a ragged
table, this can differ from the lightest or heaviest mass of the whole table.

### Climb and descent

PIANO does not tabulate climb and descent performance at the current mass of
the aircraft. It flies one climb or descent profile from each initial mass,
and the rows of a profile are for the current mass `initial_mass - burn`. The
climb and descent tables are therefore functions of `(initial_mass, fl)`.

To query climb/descent performance, the model converts the current mass of
the state to an initial mass, with {py:class}`PianoClimbDescentMassAxis
<AEIC.performance.models.piano.PianoClimbDescentMassAxis>`:

1. At the query flight level, compute the current mass of each profile that
   covers this flight level. The cumulative burn is linear between rows.
2. Find the two neighbouring profiles whose current masses are on each side
   of the query mass.
3. Interpolate the initial mass between these two profiles, with the linear
   weight of the query mass between their current masses.

The model then interpolates the table at `(initial_mass, fl)` and out of range
queries raise a `ValueError`. Similarly to cruise, a mass of `'min'` or `'max'`
is not converted. The table resolves it to the lightest or heaviest initial mass
that is valid at the flight level.

### Cruise

Cruise interpolates the cruise table at `(fl, mass, mach)`, with the Mach
number from the {py:class}`PianoConstantMachCruise
<AEIC.performance.types.PianoConstantMachCruise>` flight rules. Here the mass is the
current mass of the aircraft, same as in the legacy model.

Cruise also checks that the rate of climb at max climb thrust rating is larger
than what is considered the minimum potential rate of climb for level cruise.
This minimum is set in {py:class}`PianoConstantMachCruise
<AEIC.performance.types.PianoConstantMachCruise>`. If the climb potential is
below that value, the model raises `ValueError`.

## API reference

```{eval-rst}
.. autoclass:: AEIC.performance.models.PianoPerformanceModel
   :members:
   :exclude-members: model_config, model_type, require_operating_empty_mass,
      require_hold_flight_performance, build_performance_tables

.. autodata:: AEIC.performance.models.piano.PianoFlightRules

.. autoclass:: AEIC.performance.models.piano.HoldPerformance
   :members:
   :exclude-members: model_config

.. autoclass:: AEIC.performance.models.piano.PianoClimbDescentMassAxis
   :members:

.. autoclass:: AEIC.performance.models.ragged.RaggedPerformanceTable
   :members:

.. autoclass:: AEIC.performance.models.ragged.RaggedInterpolator
   :members:
   :special-members: __call__
   :exclude-members: valid_range
```
