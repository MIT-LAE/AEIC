# Performance model files

Performance model files are TOML files containing data to define an aircraft
performance model. Different performance model types are supported, and are
distinguished by a top-level `model_type` field, with one of the following
values:

 * `model_type = "legacy"`: legacy table-based performance model, intended to
   replicate the behavior of the AEIC v2 Matlab code. These files are
   essentially just a conversion of BADA PTF files into TOML format, including
   some extra information from the engine database.
 * `model_type = "bada"`: BADA3-based performance model.
 * `model_type = "tasopt"`: performance model based on TASOPT simulations.

```{note}
So far, only the legacy performance model is implemented.
```

## Creating performance model files

There is a {command}`make-performance-model` command to help with the creation
of performance model TOML files.

```{warning}
Currently, this can obviously only generate "legacy" performance model files!

Also, the script is unfinished and needs some work by Adi and/or Wyatt to pin
down some of the choices for generating the performance table and LTO data.
```

A complete invocation of {command}`make-performance-model` to generate a
legacy performance model file looks like this:

```{shell}
uv run make-performance-model \
  --output-file tmp.toml \
  legacy \
  --apu-name 'APU 131-9' \
  --number-of-engines 2 \
  --aircraft-class narrow \
  --ptf-file /home/bada/B738__.PTF \
  --engine-file engines/sample_edb.xlsx \
  --engine-uid 01P11CM121
```

Performance data is taken from a BADA PTF file, and LTO data is taken from the
engine database.

(More documentation to come when this is finished.)

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
cas_lo = 128.611
cas_hi = 154.3332
mach = 0.80

[speeds.cruise]
cas_lo = 128.611
cas_hi = 144.04432
mach = 0.80

[speeds.descent]
cas_lo = 128.611
cas_hi = 149.18876
mach = 0.80

# ------------------------------------------------------------------------------
#
# LTO data
#

[LTO_performance]
source = "EDB"
ICAO_UID = "01P11CM121" # Add UID for EDB data
Foo_kN = 102.695

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

[flight_performance]
cols = [
  "fuel_flow",  # kg/s - REQUIRED; OUTPUT COLUMN
  "fl",  # Flight levels
  "tas",  # m/s
  "rocd",  # m/s
  "mass"  # kg
]

data = [
  [ 1.3283519891023357,   0.0,  80.81275720164649,   23.82067487604034, 51434.0,  33.45038717147103],
  [0.12407137859174493,   0.0,  74.12551440329176,  -4.666077749500726, 51434.0,  5.460240925557456],
  [ 1.3287732751930788,   5.0,  81.39379412431006,  23.925338277820426, 51434.0,  33.47349770720624],
  ... data elided ...
  [ 1.5102911008506101, 410.0,   235.983685523832,  26.808004912602275, 81371.0, 47.160758804286196],
  [ 1.1025791086163905, 410.0,   235.983685523832,                 0.0, 81371.0, 29.337465977104877],
  [0.26136051703345853, 410.0,   235.983685523832, -13.952738253118492, 81371.0,  9.232965393299214],
]
```
