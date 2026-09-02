# PIANO performance model

The {py:class}`PianoPerformanceModel
<AEIC.performance.models.PianoPerformanceModel>` class represents a
performance model built from PIANO climb, cruise and descent exports. Files
for this model type use `model_type = "piano"`.

```{warning}
The file format and the model builder are implemented, but performance
evaluation is not: {py:meth}`evaluate_impl
<AEIC.performance.models.PianoPerformanceModel.evaluate_impl>` raises
`NotImplementedError`.
```

The source data is read by {py:mod}`AEIC.parsers.piano_reader`, which is
documented in [PIANO reader](../parsers/piano_reader.md).

## Differences from the legacy model

 * Cruise is swept over Mach number as well as flight level and mass, whereas
   the legacy model is only swept over flight level and mass.
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
rule to estimate it. This field is must be supplied when creating the
performance model file.

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
| `mass` | kg | Aircraft mass, the block's initial mass. |
| `tas` | m/s | True airspeed, derived from the airspeed schedule. |
| `rocd` | m/s | Rate of climb. |
| `fuel_flow` | kg/s | Fuel flow, derived from the burn and time deltas. |
| `time` | s | Cumulative from the start of the phase. |
| `distance` | m | Cumulative from the start of the phase. |
| `burn` | kg | Cumulative from the start of the phase. |
| `fn_per_engine` | N | Net thrust per engine. |
| `drag` | N | Drag. |

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

### Descent table

| Column | Unit | Meaning |
|--------|------|---------|
| `fl` | flight level | Altitude. |
| `mass` | kg | Aircraft mass. |
| `tas` | m/s | True airspeed, derived from the airspeed schedule. |
| `rocd` | m/s | Rate of descent, negative. |
| `fuel_flow` | kg/s | Fuel flow, derived from the burn and time deltas. |
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

Only (flight level, mass) groups holding all three reference Mach numbers are
written, because table data is a list of numbers and TOML has no null.

% TODO FIX THIS

Due to rounding in the PIANO output, a `(fl, mass, Mach)` triplet might be
the same for a row coming from the PIANO Mach sweep, and a row coming from
one of the operating points but with slightly differing performance values.
In this case, the operating point value is kept over the Mach sweep.


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
```

## Performance evaluation

% TODO: fill this in when evaluate_impl is implemented:
% - which flight rules class the model accepts (currently SimpleFlightRules);
% - how the cruise Mach dimension is collapsed, and whether the reference Mach
%   table drives that choice;
% - whether the idle thrust table affects descent fuel flow.

Not yet implemented. {py:meth}`evaluate_impl
<AEIC.performance.models.PianoPerformanceModel.evaluate_impl>` raises
`NotImplementedError`.

## API reference

```{eval-rst}
.. autoclass:: AEIC.performance.models.PianoPerformanceModel
   :members:
   :exclude-members: model_config, model_type, require_operating_empty_mass
```
