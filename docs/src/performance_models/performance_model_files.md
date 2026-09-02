# Performance model files

Performance model files are TOML files containing data to define an aircraft
performance model. Different performance model types are supported, and are
distinguished by a top-level `model_type` field, with one of the following
values:

 * `model_type = "legacy"`: legacy table-based performance model, intended to
   replicate the behavior of the AEIC v2 Matlab code. These files are
   essentially just a conversion of BADA PTF files into TOML format, including
   some extra information from the engine database.
 * `model_type = "piano"`: table-based performance model built from PIANO
   aircraft performance exports.
 * `model_type = "bada"`: BADA3-based performance model.
 * `model_type = "tasopt"`: performance model based on TASOPT simulations.

```{note}
Legacy and PIANO model files can both be generated and loaded. So far, only
the legacy performance model can be evaluated.
```

## Creating performance model files

There is a {command}`make-performance-model` command to help with the creation
of performance model TOML files.

The command takes the output path, then a subcommand naming the model type to
build:

 * `legacy` builds a legacy model from a BADA PTF file;
 * `piano` builds a PIANO model from a set of three PIANO exports;
 * `tasopt` is registered but is a stub that raises `NotImplementedError`.

```{warning}
The script is unfinished and needs some work by Adi and/or Wyatt to pin down
some of the choices for generating the performance table and LTO data.
```

### Shared options

Calling {command}`make-performance-model` requires passing `--output-file` as the
first argument before specifying the subcommand.

Every subcommand takes these same options for LTO data and for aircraft
properties.

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--lto-source` | `edb` or `custom` | Yes | | Source of LTO performance data. |
| `--engine-file` | Path | With `edb` | | Engine database workbook. |
| `--engine-uid` | String | With `edb` | | UID of the engine to extract data for. |
| `--thrust-fractions` | 4 floats | No | `0.07 0.30 0.85 1.0` | Thrust fractions for idle, approach, climb and takeoff. |
| `--lto-file` | Path | With `custom` | | LTO TOML file. |
| `--aircraft-class` | `wide`, `narrow`, `small` or `freight` | Yes | | Aircraft class. |
| `--number-of-engines` | Integer, 1 to 8 | Yes | | Number of engines on the aircraft. |
| `--maximum-payload` | Integer | Yes | | Maximum payload [kg]. |
| `--apu-name` | String | No | | APU name. Must be in the APU database. |

### `legacy` subcommand

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--ptf-file` | Path | Yes | Input BADA PTF file. |

A complete invocation of {command}`make-performance-model` to generate a
legacy performance model file looks like this:

```shell
aeic make-performance-model \
  --output-file tmp.toml \
  legacy \
  --apu-name 'APU 131-9' \
  --number-of-engines 2 \
  --aircraft-class narrow \
  --maximum-payload 22422 \
  --ptf-file /home/bada/B738__.PTF \
  --lto-source edb \
  --engine-file engines/sample_edb.xlsx \
  --engine-uid 01P11CM121
```

Performance data is taken from a BADA PTF file, and LTO data is taken from the
engine database. The aircraft name, ISA offset, maximum altitude and speed
schedules all come from the PTF file.

### `piano` subcommand

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--cruise-file` | Path | Yes | Input PIANO cruise table export. |
| `--climb-file` | Path | Yes | Input PIANO climb details export. |
| `--descent-file` | Path | Yes | Input PIANO descent details export. |
| `--operating-empty-mass` | Float | Yes | Operating empty mass [kg]. |
| `--climb-masses` | Comma-separated floats | No | Initial mass [kg] of each climb block, in file order. |
| `--climb-cas-low-kts` | Float | No | Climb calibrated airspeed below FL100 [knots]. |
| `--climb-cas-high-kts` | Float | No | Climb calibrated airspeed above FL100 [knots]. |
| `--climb-mach` | Float | No | Climb Mach number above the crossover altitude. |
| `--climb-crossover-altitude-ft` | Float | No | Altitude above which the climb flies its Mach number [feet]. |
| `--aircraft-name` | String | No | Aircraft name, overriding the cruise file title. |
| `--maximum-altitude-ft` | Integer | No | Maximum altitude [feet], overriding the highest altitude climbed. |

The optional options fill gaps that some PIANO files may have due to missing headers:

 * `--climb-masses` is needed when there is no climb header in the PIANO
   climb file. This is because the climb halted before reaching its target
   altitude. The option is all-or-nothing: give one mass per block, in file order,
   including the blocks whose header already states a mass. A supplied mass
   that contradicts one in the file warns and overrides it.
 * The four schedule options override the climb airspeed schedule the file
   contains. If none of the climbs in the provided file have a header,
   all four options are required.

A complete invocation looks like this:

```shell
aeic make-performance-model \
  --output-file B738-piano.toml \
  piano \
  --cruise-file piano/B738-cruise.txt \
  --climb-file piano/B738-climb.txt \
  --descent-file piano/B738-descent.txt \
  --operating-empty-mass 41413 \
  --apu-name 'APU 131-9' \
  --number-of-engines 2 \
  --aircraft-class narrow \
  --maximum-payload 22422 \
  --lto-source edb \
  --engine-file engines/sample_edb.xlsx \
  --engine-uid 01P11CM121
```

Add `--climb-masses 50000,60000,70000` if any climb in the file halted before
reaching its target altitude.

### Performance model file format

TOML performance files are created from the model instances using
{py:func}`write_performance_model
<AEIC.performance.model_builder.write_performance_model>`.
Each model type defines its TOML layout via an
ordered sequence of `(field name, output key)` pairs (e.g. `LEGACY_WRITE_SPEC`
and `PIANO_WRITE_SPEC`) that defines the field order and the key each field is
written under. Writing a model type that has no write spec raises a
`ValueError`.

Then:

 * tabular data becomes a trailing `[section]` block, and an empty table
   is written as `data = []`;
 * speed and LTO data become nested sub-tables, one per flight phase and one
   per thrust mode;
 * everything else is written inline;
 * a field set to `None` is not written at all.


## Example file

The following shows an example performance model file, for a legacy
table-based performance model. The fields common to all performance models
(defined in the {py:class}`BasePerformanceModel
<AEIC.performance.models.BasePerformanceModel>` class) come first, then the
fields specific to the {py:class}`LegacyPerformanceModel
<AEIC.performance.models.LegacyPerformanceModel>` class. In this case, the
model type-specific fields are a table of performance data that the model
uses.

All of the common fields are defined as simple attributes on the
{py:class}`BasePerformanceModel
<AEIC.performance.models.BasePerformanceModel>` class and are interpreted
according to the usual rules for fields in Pydantic models. In particular, the
`speeds` and `LTO_performance` tables within the TOML data define dictionaries
mapping from flight phases to speed and performance data. The fields and
sub-types for these entries can be seen by following the types from the
{py:attr}`speeds <AEIC.performance.models.BasePerformanceModel.speeds>` and
{py:attr}`lto_performance
<AEIC.performance.models.BasePerformanceModel.lto_performance>` fields of the
{py:class}`BasePerformanceModel
<AEIC.performance.models.BasePerformanceModel>` class.

```
# Performance model type (one of: legacy, bada, tasopt, piano).
model_type = "legacy"

# ==============================================================================
#
#  COMMON FIELDS
#
# Fields common to all performance model types.

aircraft_name = "B738"
aircraft_class = "narrow" # wide, narrow, small, freight
ISA_offset = 0
maximum_altitude_ft = 41000
maximum_payload_kg = 22422
number_of_engines = 2 # Number of engines
APU_name = "APU 131-9" # None: APU emissions not calculated

# ------------------------------------------------------------------------------
#
# Speed data
#

[speeds.climb]
cas_low = 128.611
cas_high = 154.3332
mach = 0.80

[speeds.cruise]
cas_low = 128.611
cas_high = 144.04432
mach = 0.80

[speeds.descent]
cas_low = 128.611
cas_high = 149.18876
mach = 0.80

# ------------------------------------------------------------------------------
#
# LTO data
#

[LTO_performance]
source = "EDB"
ICAO_UID = "01P11CM121" # Add UID for EDB data
rated_thrust = 102.695

[LTO_performance.mode_data.approach]
thrust_frac = 0.3
fuel_kgs    = 0.278
EI_NOx      = 0.0
EI_HC       = 0.0
EI_CO       = 0.0

[LTO_performance.mode_data.climb]
thrust_frac = 0.85
fuel_kgs    = 0.754
EI_NOx      = 0.0
EI_HC       = 0.0
EI_CO       = 0.0

[LTO_performance.mode_data.takeoff]
thrust_frac = 1.0
fuel_kgs    = 0.903
EI_NOx      = 0.0
EI_HC       = 0.0
EI_CO       = 0.0

[LTO_performance.mode_data.idle]
thrust_frac = 0.07
fuel_kgs    = 0.102
EI_NOx      = 0.0
EI_HC       = 0.0
EI_CO       = 0.0

# ==============================================================================
#
#  MODEL-TYPE SPECIFIC FIELDS
#

# ------------------------------------------------------------------------------
#
# Performance table data.
#
# Three separate tables, one per flight phase. Each table has identical
# `cols` (`fuel_flow, fl, tas, rocd, mass`) and a `data` array indexed
# first by aircraft mass and then by flight level. `make-performance-model`
# emits this layout automatically.

[climb_flight_performance]
cols = [
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "fl",  # Flight levels
  "tas",  # m/s
  "rocd",  # m/s
  "mass"  # kg
]

data = [
  [ 1.3283519891023357,   0.0,  80.81275720164649,   23.82067487604034, 51434.0],
  [ 1.3287732751930788,   5.0,  81.39379412431006,  23.925338277820426, 51434.0],
  ... data elided ...
  [ 1.5102911008506101, 410.0,   235.983685523832,    50.1228713796996, 51434.0],
  [ 1.5102911008506101, 410.0,   235.983685523832,  35.296805551618895, 68534.0],
  [ 1.5102911008506101, 410.0,   235.983685523832,  26.808004912602275, 81371.0],
]

[cruise_flight_performance]
cols = [
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "fl",  # Flight levels
  "tas",  # m/s
  "rocd",  # m/s
  "mass"  # kg
]

data = [
  [ 0.5563723740683004,  60.0, 140.08552139650615, 0.0, 51434.0],
  ... data elided ...
  [ 0.7943602849587021, 410.0,   235.983685523832, 0.0, 51434.0],
  [ 0.9157915873634661, 410.0,   235.983685523832, 0.0, 68534.0],
  [ 1.1025791086163905, 410.0,   235.983685523832, 0.0, 81371.0],
]

[descent_flight_performance]
cols = [
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "fl",  # Flight levels
  "tas",  # m/s
  "rocd",  # m/s
  "mass"  # kg
]

data = [
  [0.12407137859174493,   0.0,  74.12551440329176,  -3.376099606296839, 68534.0],
  ... data elided ...
  [0.25644997757471305, 410.0,   235.983685523832, -13.102832329609141, 68534.0],
]
```
