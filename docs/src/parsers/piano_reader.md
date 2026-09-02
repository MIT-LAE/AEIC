# PIANO reader

{py:mod}`AEIC.parsers.piano_reader` parses a set of three PIANO text exports
for one aircraft: a cruise table, a climb details file and a descent details
file. Files are parsed via {py:meth}`PianoData.load
<AEIC.parsers.piano_reader.PianoData.load>`, which returns
{py:class}`PianoData <AEIC.parsers.piano_reader.PianoData>` containing climb,
cruise and descent data. Missing values/overrides are passed through
{py:class}`PianoOverrides <AEIC.parsers.piano_reader.PianoOverrides>`.

The package is split into one reader per file, plus shared helpers:

 * `piano_data.py` defines `PianoData` and `PianoOverrides` and drives the
   three parsers
 * `cruise_reader.py` parses the single-table cruise sweep
 * `climb_reader.py` parses the per-mass climb detail blocks
 * `descent_reader.py` parses the per-mass descent detail blocks
 * `common.py` holds block splitting, airspeed schedule parsing, delta
   calculation and the cross-checks.

Most users should only need `PianoData` and `PianoOverrides`.

```{eval-rst}
.. autoclass:: AEIC.parsers.piano_reader.PianoData
    :members:
```

```{eval-rst}
.. autoclass:: AEIC.parsers.piano_reader.PianoOverrides
    :members:
```

## Expected PIANO file schema

### Cruise table export

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
are dropped. PIANO uses `...` as a placeholder value. It can mean:

 * The value was not asked by the user to be computed (e.g. NOx emissions)
 * Row of all `...`: the aircraft cannot fly at the specified `(mass, altitude, mach)` point
 * Row with `...` for `MCR` and `MCL...` columns: the requested `altitude` is higher than the aircraft's operating ceiling

The fourth column is a discriminator that separates the two kinds of row:

 * `|` marks a swept row, one point of the Mach sweep
 * `maxSAR`, `99%SAR` and `maxLim` mark a reference Mach of the (mass,
   altitude) pair the sweep just covered
 * anything else raises `ValueError`, because the export does not match the
   layout read here.

### Climb details export

A climb file has one block per initial mass. Each block is a header followed
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

Every `Climb details` table is preceded by either a full header or a halt
note. If the climb does not reach its target altitude, PIANO writes the halt
note in place of that block's own header, and the block has no metadata all:

```
    Climb to 41000.feet halted at 40000.feet,
    Rate of Climb < 50.feet/min.
    -----------------------------------------
```

### Descent details export

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

## Airspeed schedules and derived true airspeed

The PIANO climb and descent tables have no true airspeed column. True airspeed
is derived from the airspeed schedule in the block header.

PIANO writes the schedule a such:

 * climb: `250./ 280.kcas/ mach 0.750 above 30000.feet`, low CAS first;
 * descent: `mach 0.750 above 30000.feet/ 280./ 250.kcas`, high CAS first.

The speed schedule is the same for all blocks in a file. Then
first block that has a schedule provides it for every other block.
For climb this is necessary, because a halted block has no header. For descent, a
later block that states a different schedule only warns, and the first block's
schedule is used throughout.

True airspeed at a given altitude then follows three regimes:

 * above the crossover altitude, the schedule flies its Mach number, so
   {math}`\mathrm{TAS} = M \, a(h)`;
 * below FL100, the schedule flies its low CAS, converted with
   {py:func}`cas_to_tas <AEIC.utils.standard_atmosphere.cas_to_tas>`;
 * between FL100 and the crossover altitude, the schedule flies its high CAS,
   converted the same way.

## Fuel flow and cumulative columns

PIANO does not tabulate fuel flow for climb and descent. The reader derives it
from the cumulative burn and time columns of consecutive rows:

```{math}
\dot{m}_i = \frac{B_i - B_{i-1}}{t_i - t_{i-1}}
```

This is a **backward** difference: row {math}`i` is the average fuel flow
over the step that ends at row {math}`i`, and not the instantaneous value at
that row's altitude.

The first row of a block is the exception. It takes
a **forward** difference instead and is assigned the same step as the second
row.

```{note}
Double check if this is the correct thing to do
```

## Dropped rows and parse failures

### Data rows dropped with a warning

| Case | File | Behaviour |
|------|------|-----------|
| Line starts with a number but holds fewer than fifteen columns | cruise | Dropped. One aggregate warning gives the count and the first offending line. |
| Row spans no time, so {math}`\Delta t = 0` | climb, descent | Dropped, with one warning per row. Fuel flow is undefined over a zero-length step. |
| Block holds no data rows at all | climb, descent | Skipped, with one warning. |
| Reference Mach group missing any of `maxSAR`, `99%SAR` or `maxLim` | cruise | The whole (flight level, mass) group is dropped. |
| Block states no `Idle thrust below` altitude | descent | No idle thrust row is emitted for that mass. |
| Duplicate row at the same (flight level, mass, Mach) | cruise | The labelled row is kept over the swept one. The warning lists the columns that disagree. |

A labelled reference Mach can land exactly on the swept grid, and PIANO does
not always report the same values for the two rows. The labelled row is kept,
because it is the operating point PIANO solved for; the swept row only shares
its key because the export rounds the Mach number.

### Errors

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

## Cross-checks

The reader emits warnings on sanity checks of the parsed values.

**True airspeed.** The derived true airspeed is compared against the block's
own distance and time columns, as
{math}`\sqrt{(\Delta d / \Delta t)^2 + \mathrm{ROCD}^2}`.
The check uses the median relative deviation over
the block, with a tolerance of 5%.

**Fuel burn.** The final cumulative burn of a block is compared against the
`Fuel burn` total in its header, with a tolerance of 1%. The check is skipped
if the header is not present.

**Climb mass.** When the caller supplies climb masses and a block header also
has a value, the two are compared with a tolerance of 1%.

**Descent schedule.** A descent block whose airspeed schedule differs from the
first block's warns, and the first block's schedule is used throughout.
