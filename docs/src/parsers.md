# Available parsers

The {py:mod}`AEIC.parsers` package contains readers for a number of external
file formats that AEIC consumes when building performance models or
converting reference data. These modules are primarily used by other
commands (in particular {command}`aeic make-performance-model`); most users
will not need to call them directly, but the dataclasses they return are
part of the public contract for performance-model construction.

## PTF reader

{py:mod}`AEIC.parsers.ptf_reader` parses BADA PTF files, returning a
{py:class}`AEIC.parsers.ptf_reader.PTFData` object containing per-phase
performance data for climb, cruise and descent. This is the source of
performance-table data for legacy performance models.

```{eval-rst}
.. automodule:: AEIC.parsers.ptf_reader
   :members:
```

## PIANO reader

{py:mod}`AEIC.parsers.piano_reader` parses a set of three PIANO text exports
for one aircraft: a cruise table, a climb details file and a descent details
file. The entry point is {py:meth}`PianoData.load
<AEIC.parsers.piano_reader.PianoData.load>`, which returns a single
{py:class}`PianoData <AEIC.parsers.piano_reader.PianoData>` value holding the
three performance tables plus the metadata the files carry. Values PIANO does
not always write to the files are supplied through
{py:class}`PianoOverrides <AEIC.parsers.piano_reader.PianoOverrides>`.

PIANO writes imperial units. The reader converts everything to SI, so the
tables in a {py:class}`PianoData <AEIC.parsers.piano_reader.PianoData>` value
are ready to be written to a performance model TOML file (see [PIANO
performance model](performance_models/piano_performance_model.md)).

The package is split into one reader per file, plus shared helpers:

 * `piano_data.py` defines `PianoData` and `PianoOverrides` and drives the
   three parsers;
 * `cruise_reader.py` parses the single-table cruise sweep;
 * `climb_reader.py` parses the per-mass climb detail blocks;
 * `descent_reader.py` parses the per-mass descent detail blocks;
 * `common.py` holds block splitting, airspeed schedule parsing, delta
   calculation and the cross-checks.

Only `PianoData` and `PianoOverrides` are public. The three `_parse_*`
functions are private to the package.

% TODO: say a sentence about where the three export files come from, i.e. what
% the user has to ask PIANO for. Name the PIANO menu items if we know them.

### Parsed data

A {py:class}`PianoData <AEIC.parsers.piano_reader.PianoData>` value holds:

| Field | Type | Source |
|-------|------|--------|
| `aircraft_name` | `str` | Cruise table title. |
| `maximum_altitude_ft` | `int` | Highest altitude any climb block reaches. |
| `isa_offset` | `int` | `Delta-ISA` header of the climb file. |
| `climb_speeds` | {py:class}`SpeedData <AEIC.performance.types.SpeedData>` | Climb airspeed schedule. |
| `descent_speeds` | {py:class}`SpeedData <AEIC.performance.types.SpeedData>` | Descent airspeed schedule. |
| `climb` | {py:class}`TableInput <AEIC.performance.types.TableInput>` | Climb performance table. |
| `cruise` | {py:class}`TableInput <AEIC.performance.types.TableInput>` | Cruise performance table. |
| `descent` | {py:class}`TableInput <AEIC.performance.types.TableInput>` | Descent performance table. |
| `cruise_reference_mach` | {py:class}`TableInput <AEIC.performance.types.TableInput>` | Reference Mach numbers PIANO labels in the cruise sweep. |
| `descent_idle_thrust` | {py:class}`TableInput <AEIC.performance.types.TableInput>` | Altitude below which each descent block flies at idle thrust. |

The column labels of each table are given on the [PIANO performance
model](performance_models/piano_performance_model.md) page, which also
documents the units of each column.

### Expected PIANO file schema

The reader keys on the text PIANO writes, so it depends on the export layout.
This section records the layout it expects.

#### Cruise table export

A cruise file is a single table without blocks. PIANO sweeps Mach number for
every (mass, altitude) pair. The title line is required, and gives the
aircraft name.

```
  Cruise table for some_airplane, some_engine

  Mass  Altitude  Mach   |   TAS   CAS   Drag    MCR.%    L/D   FuelFlow    SFC       SAR    MCL/eng   RoC@MCL   RoC@MCL   Buffet    NOx     HC      CO
  ----  --------  ----   |   ---   ---   ----    -----    ---   --------    ---       ---     avail.   fix.Mach  fix.CAS   onset     ---     --      --
   lb.    feet    ....   |   kts   kts   lbf.   percent   ...    lb/hr    lb/h/lbf   nm/lb     lbf.    feet/min  feet/min    Gs      lb/hr   lb/hr   lb/hr
  93000.  15000.  0.350  |  100.0  200.0  3000.   40.0   150.0    6000.    0.7000    0.08000  19000.     1000.     2000.     ...      ...     ...     ...
  93000.  15000.  0.450 maxSAR 100.0 200.0 3000.  40.0   150.0    6000.    0.7000    0.08000  19000.     1000.     2000.     ...      ...     ...     ...
```

Only the first fifteen tokens of a row are read. Buffet onset, NOx, HC and CO
are dropped. PIANO writes `...` in a column it has no value for.

The fourth column is a discriminator that separates the two kinds of row:

 * `|` marks a swept row, one point of the Mach sweep;
 * `maxSAR`, `99%SAR` and `maxLim` mark a reference Mach of the (mass,
   altitude) pair the sweep just covered;
 * anything else raises `ValueError`, because the export does not match the
   layout read here.

#### Climb details export

A climb file holds one block per initial mass. Each block is a header followed
by a `Climb details` table. The reader splits the file on the `Climb details`
marker and treats everything since the previous marker as the block's header
lines.

```
    Climb from 0.feet to:   41000.feet
    ---------------------------------------------------------------
    Time                28.00     minutes
    Fuel burn           4000.     lb.
    Distance            200.0     n.miles

    Initial mass        150000.   lb.
    Airspeed schedule   250./ 280.kcas/ mach 0.750 above 30000.feet
    Delta-ISA           +0.       deg.C.
    ---------------------------------------------------------------

    Climb details

     Alt.     Time      Dist.      Burn      FN/eng    R.o.C.     Drag
    (feet)    (sec)   (n.miles)    (lb.)     (lbf.)    (f.p.m)    (lbf.)

        0.       0.       0.0         0.     30000.    2000.     1000.
     1417.      60.       5.0        40.     30000.    2000.     1000.
```

The reader takes four values from the header:

 * `Initial mass` gives the block's mass, unless the caller supplies one;
 * `Airspeed schedule` gives the CAS below and above FL100, the Mach number
   and the crossover altitude;
 * `Delta-ISA` gives the ISA temperature offset;
 * `Fuel burn` gives the block total that the last data row is checked
   against.

Rows run from the ground up, one per altitude step, at the mass the header
states. `Time`, `Dist.` and `Burn` are cumulative from the start of the climb.
`FN/eng`, `R.o.C.` and `Drag` are the value at that altitude.

If the climb does not reach its target altitude, PIANO writes a halt note in
place of the *next* block's header, so that block carries no metadata at all:

```
    Climb to 41000.feet halted at 40000.feet,
    Rate of Climb < 50.feet/min.
    -----------------------------------------
```

A halted climb is logged at info level. It is also the reason the caller may
have to supply the block masses explicitly: a block that follows a halt has no
`Initial mass` header to read.

#### Descent details export

A descent file has the same block shape as a climb file, split on the
`Descent details` marker. Unlike climb, PIANO writes a full header for every
block, so there is no halt note and no headerless block.

```
    Descent from 41000.feet to:   0.feet
    ---------------------------------------------------------------
    Time                20.0      minutes
    Fuel burn           800.      lb.
    Distance            200.0     n.miles

    Mass                100000.    lb.
    Airspeed schedule   mach 0.750 above 30000.feet/ 280./ 250.kcas
    Idle thrust below   30000.feet
    ---------------------------------------------------------------

    Descent details

     Alt.     Time      Dist.      Burn     R.o.D.     FN/eng
    (feet)    (sec)    (n.miles)   (lb.)    (f.p.m)    (lbf.)

    41000.      00.      200.        30.     1400.     1500.
    39610.      01.      200.        30.     1400.     1500.
```

The reader takes four values from the header: `Mass`, `Airspeed schedule`,
`Idle thrust below` and `Fuel burn`.

Rows run from the top down, one per altitude step, at the mass the header
states. `Time`, `Dist.` and `Burn` are cumulative from the start of the
descent. `R.o.D.` and `FN/eng` are the value at that altitude. PIANO reports
the rate of descent as a positive number, so the reader negates it.

A descent file states no `Delta-ISA`, so the ISA offset comes from the climb
file.

### Airspeed schedules and derived true airspeed

The PIANO climb and descent tables have no true airspeed column. True airspeed
is derived from the airspeed schedule in the block header.

PIANO writes the schedule in the order the phase flies it, so the two phases
read in opposite order:

 * climb: `250./ 280.kcas/ mach 0.750 above 30000.feet`, low CAS first;
 * descent: `mach 0.750 above 30000.feet/ 280./ 250.kcas`, high CAS first.

The schedule is treated as a property of the file rather than of a block: the
first block that states one supplies it for every block. For climb this is
necessary, because a block that follows a halt has no header. For descent, a
later block that states a different schedule only warns, and the first block's
schedule is used throughout.

True airspeed at a given altitude then follows three regimes:

 * above the crossover altitude, the schedule flies its Mach number, so
   {math}`\mathrm{TAS} = M \, a(h)`;
 * below FL100, the schedule flies its low CAS, converted with
   {py:func}`cas_to_tas <AEIC.utils.standard_atmosphere.cas_to_tas>`;
 * between FL100 and the crossover altitude, the schedule flies its high CAS,
   converted the same way.

```{note}
FL100 is an assumption of the reader, not a value taken from the file. PIANO
applies the ATC speed limit that holds below FL100, which is where the low
CAS of the schedule applies.
```

Because the CAS to TAS conversion assumes a standard atmosphere, a climb file
that states a non-zero `Delta-ISA` is rejected rather than warned about.

### Fuel flow and cumulative columns

PIANO does not tabulate fuel flow for climb and descent. The reader derives it
from the cumulative burn and time columns of consecutive rows:

```{math}
\dot{m}_i = \frac{B_i - B_{i-1}}{t_i - t_{i-1}}
```

This is a **backward** difference: row {math}`i` holds the average fuel flow
over the step that ends at row {math}`i`, not the instantaneous value at that
row's altitude.

The first row of a block is the exception. It has no predecessor, so it takes
a **forward** difference instead and is assigned the same step as the second
row. The consequence to keep in mind is that the first row's fuel flow is not
measured at its own altitude; it is the average over the first step above it.
The alternative, a zero-over-zero difference, would drop the row and leave the
table thin at the start of the phase.

The same convention applies to the time, distance and burn deltas, which are
all produced by the same helper. The `time`, `distance` and `burn` columns of
the output tables keep PIANO's cumulative values unchanged; only fuel flow is
derived from the differences.

% TODO: decide whether to state the first-row convention as a known limitation
% here, or to leave it as a plain description of the behaviour.

### Dropped rows and parse failures

The reader distinguishes three cases: lines that were never data, rows that
are dropped with a warning, and files that are rejected outright.

#### Lines that are not data

PIANO opens every data row with a number, so a line whose first token does not
parse as a float is not a data row. This is how titles, column headings, unit
rows and separator lines are skipped, silently and by design.

#### Rows dropped with a warning

| Case | File | Behaviour |
|------|------|-----------|
| Line starts with a number but holds fewer than fifteen columns | cruise | Dropped. One aggregate warning gives the count and the first offending line. |
| Row spans no time, so {math}`\Delta t = 0` | climb, descent | Dropped, with one warning per row. Fuel flow is undefined over a zero-length step. |
| Block holds no data rows at all | climb, descent | Skipped, with one warning. |
| Reference Mach group missing any of `maxSAR`, `99%SAR` or `maxLim` | cruise | The whole (flight level, mass) group is dropped. |
| Block states no `Idle thrust below` altitude | descent | No idle thrust row is emitted for that mass. |
| Duplicate row at the same (flight level, mass, Mach) | cruise | The swept row is kept over the labelled one. The warning lists the columns that disagree. |

A short cruise line is dropped rather than rejected because the cruise table is
an interpolation grid: a lost row thins the grid but leaves its neighbours
usable.

An incomplete reference Mach group is dropped because table data is a list of
numbers and TOML has no null, so a group missing one label cannot be written.

A labelled reference Mach can land exactly on the swept grid, and PIANO does
not always report the same values for the two rows. Keeping the swept row
keeps the sweep self-consistent.

The descent idle thrust table can therefore end up empty. An empty table is
valid, and states that no block qualified.

#### Errors

The reader raises `ValueError` in the following cases.

| Case | File |
|------|------|
| A data line holds the right leading number but not exactly seven (climb) or six (descent) numbers | climb, descent |
| The file holds no block marker | climb, descent |
| The file has no `Cruise table for` title | cruise |
| The fourth column is neither `\|` nor a known reference Mach label | cruise |
| `Delta-ISA` is non-zero | climb |
| A block has no `Initial mass` header and no mass was supplied | climb |
| The supplied mass count does not match the block count | climb |
| No block states an airspeed schedule and the overrides are incomplete | climb |
| A block has no `Mass` header | descent |
| A block has no `Airspeed schedule` header | descent |

A malformed climb or descent row is an error rather than a drop, which is the
one asymmetry with the cruise reader. Dropping such a line would lose more
than one row's worth of information: the deltas are taken over the rows that
survive, so the row after the gap would report the fuel burn of two steps over
the time of two steps. That row's own cumulative columns stay correct and
neither cross-check can see the gap, so the result would look sound while
being wrong.

PIANO writes fixed-width columns, so the usual cause of a malformed row is a
value that outgrew its field and ran into its neighbour, merging the two into
one token.

### Cross-checks

The reader never fails on a cross-check. The checks exist to surface a misread
file, not to validate PIANO output.

**True airspeed.** The derived true airspeed is compared against the block's
own distance and time columns, as
{math}`\sqrt{(\Delta d / \Delta t)^2 + \mathrm{ROCD}^2}`. The speed schedule
and the tabulated trajectory are independent, so a large disagreement means
one of the two was misread. The check uses the median relative deviation over
the block, with a tolerance of 5%. PIANO tabulates at coarse altitude steps,
so individual rows deviate; the median resists that without hiding a
systematic error. Rows with a zero time step or zero true airspeed are
excluded, and a block with no usable row warns.

**Fuel burn.** The final cumulative burn of a block is compared against the
`Fuel burn` total in its header, with a tolerance of 1%. The check is skipped
if the header states no total. This catches a truncated table, which the true
airspeed check cannot.

**Climb mass.** When the caller supplies climb masses and a block header also
states one, the two are compared with a tolerance of 1%. The supplied mass
wins, because a caller supplies masses precisely when the headers do not carry
them.

**Descent schedule.** A descent block whose airspeed schedule differs from the
first block's warns, and the first block's schedule is used throughout.

### Overrides

{py:class}`PianoOverrides <AEIC.parsers.piano_reader.PianoOverrides>` supplies
the values PIANO does not always write:

 * `climb_masses_kg` gives the initial mass of each climb block, in file
   order. It is all-or-nothing: supply one mass per block, including the
   blocks whose header already states one.
 * `climb_cas_low_kts`, `climb_cas_high_kts`, `climb_mach` and
   `climb_crossover_altitude_ft` override the climb airspeed schedule. Each is
   applied individually over a schedule the file states. If no block states a
   schedule at all, all four are required.

% TODO: add a worked example of reading a set of exports with overrides, or
% link to the equivalent example on the performance model files page.

### API reference

```{eval-rst}
.. automodule:: AEIC.parsers.piano_reader
   :members:
```

## OPF reader

{py:mod}`AEIC.parsers.opf_reader` parses BADA OPF (operations performance
file) records into a dictionary of aerodynamic, engine-thrust, fuel-flow
and ground-movement coefficients.

```{eval-rst}
.. automodule:: AEIC.parsers.opf_reader
   :members:
```

## LTO reader

{py:mod}`AEIC.parsers.lto_reader` parses AEIC-format LTO (landing and
take-off) characteristics files, extracting the nominal thrust value and
per-mode engine emissions indices.

```{eval-rst}
.. automodule:: AEIC.parsers.lto_reader
   :members:
```
