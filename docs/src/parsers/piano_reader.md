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
   calculation, the per-row fuel flow (`_row_fuel_flows`) and the
   cross-checks.

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
marker and treats everything after the previous marker as the block's header
lines.

```
    Climb from 0.feet to:   41000.feet
    ---------------------------------------------------------------
    Time                28.00     minutes
    Fuel burn           4000.     lb.
    Distance            200.0     n.miles

    Initial mass        150000.   lb.
    Airspeed schedule   250./ 280.kcas/ mach 0.750 above 30523.feet
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

Some exports add an eighth `NOx` column to the table. The reader ignores this
column.

Every `Climb details` table is preceded by either a full header or a halt
note. If the climb does not reach its target altitude, PIANO writes the halt
note in place of that block's own header, and the block has no metadata at
all:

```
    Climb to 41000.feet halted at 40000.feet,
    Rate of Climb < 50.feet/min.
    -----------------------------------------
```

The reader takes the halt altitude from this note. The last row of the block
can be below the halt altitude. By default, the reader then extrapolates the
block to the halt altitude. See
[Climb halt extrapolation](#climb-halt-extrapolation).

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

PIANO writes the schedule as follows:

 * climb: `250./ 280.kcas/ mach 0.750 above 30523.feet`, low CAS first;
 * descent: `mach 0.750 above 30000.feet/ 280./ 250.kcas`, high CAS first.

The speed schedule is the same for all blocks in a file. The
first block that has a schedule provides it for every other block.
For climb this is necessary, because a halted block has no header. For descent,
every block has a schedule and all must be identical.

The climb schedule can also be passed as a low CAS, a high CAS and
a Mach number. The supplied schedule replaces the header schedule. The
crossover altitude is derived: it is the altitude at which the high CAS equals
the Mach number in the standard atmosphere, as computed by
{py:func}`crossover_altitude <AEIC.utils.standard_atmosphere.crossover_altitude>`.

True airspeed at a given altitude then follows three regimes:

 * above the crossover altitude, the schedule flies its Mach number, so
   {math}`\mathrm{TAS} = M \, a(h)`;
 * below FL100, the schedule flies its low CAS, converted with
   {py:func}`cas_to_tas <AEIC.utils.standard_atmosphere.cas_to_tas>`;
 * between FL100 and the crossover altitude, the schedule flies its high CAS,
   converted the same way.

## Fuel flow and cumulative columns

PIANO does not tabulate fuel flow for climb and descent. The reader derives it
from the cumulative burn and time columns.

The trajectory builder needs the fuel flow at the flight level and mass of each
row. Interior rows take the central difference of cumulative fuel burn:

```{math}
\dot{m}_i = \frac{B_{i+1} - B_{i-1}}{t_{i+1} - t_{i-1}}
```

The first and the last rows of a block take the one-sided difference of their
only step. These two values are the average over that step.

The `time` and `burn` columns stay as PIANO writes them. The `burn` column is
the exact cumulative burn at the row, so `mass - burn` is the mass of the
aircraft at the row.

## Climb halt extrapolation

PIANO halts a climb when its rate of climb gets too low and does not reach
the target altitude. The last row of the `Climb details` table can then be up
1000 ft below the halt altitude. Without a correction, the climb ceiling of the
model is too low. Because the halt altitude is a valid altitude that the climb
reached, the reader adds an extrapolated row at the halt altitude.

The extrapolation runs only if all of these conditions are true:

 * the block has a halt note;
 * the halt altitude is more than 10 ft above the last row
   (`_HALT_GAP_MIN_FT`);
 * the extrapolation is on. It is on by default. To turn it off, pass
   `extrapolate_climb_to_halt=False` to {py:meth}`PianoData.load
   <AEIC.parsers.piano_reader.PianoData.load>`, or
   `--no-extrapolate-climb-to-halt` to {command}`make-performance-model`.

The fit uses only the rows above the crossover altitude, where the climb is at
constant Mach number. Each value at the halt altitude comes from a polynomial
through the last rows:

| Value | Fit |
|-------|-----|
| Rate of climb | Linear through the last two rows. |
| Net thrust per engine | Linear through the last two rows. |
| Drag | Quadratic through the last three rows. |
| Fuel flow | Linear through the average fuel flows of the last two intervals, each at the midpoint altitude of its interval. |

Time, distance and burn are cumulative, so the reader integrates them from the
last row {math}`h_n` to the halt altitude {math}`h_\mathrm{halt}`:

```{math}
\Delta t &= \int_{h_n}^{h_\mathrm{halt}} \frac{dh}{\mathrm{ROCD}(h)} \\
\Delta d &= \int_{h_n}^{h_\mathrm{halt}}
    \frac{\sqrt{\mathrm{TAS}(h)^2 - \mathrm{ROCD}(h)^2}}{\mathrm{ROCD}(h)} \, dh \\
\Delta B &= \int_{h_n}^{h_\mathrm{halt}} \frac{\dot{m}(h)}{\mathrm{ROCD}(h)} \, dh
```

{math}`\mathrm{ROCD}(h)` and {math}`\dot{m}(h)` are the fits above.
{math}`\mathrm{TAS}(h)` comes from the airspeed schedule. The integrals use
the trapezoidal rule on 101 altitude points.

The reader skips the extrapolation, with a warning, in these cases:

 * fewer than three rows are above the crossover altitude, or fewer than two
   interval fuel flows are available to fit;
 * the rate of climb, the net thrust, the drag or the fuel flow at the halt
   altitude is zero or negative, which is not physical.

The block then stops at its last row, as PIANO wrote it.

After the reader adds the row, it derives the fuel flow of every row of the
block again. The extrapolated row takes the one-sided difference of its step.
The last row that PIANO wrote now takes a central difference, which uses the
extrapolated row.

The cross-checks run on the rows that PIANO wrote, before the extrapolation.

## Dropped rows and parse failures

A line is a data row when its first token starts with a digit, behind an
optional sign. The reader tests only the start of that token, to decide if
the line is a data row in case of field overflows that merge columns.

### Data rows dropped with a warning

| Case | File | Behaviour |
|------|------|-----------|
| Line is a data row but holds fewer than fifteen columns | cruise | Dropped. One aggregate warning gives the count and the first offending line. |
| Line is a data row but at least one of the first fifteen numeric columns is not a number | cruise | Dropped. One aggregate warning gives the count and the first offending line. |
| The difference of a row spans no time: {math}`t_{i+1} = t_{i-1}`, or a zero-length step at the first or last row | climb, descent | Dropped, with one warning per row. Fuel flow is undefined. Real PIANO exports have no zero-length steps. |
| Block holds no data rows at all | climb, descent | Skipped, with one warning. |
| Block states no `Idle thrust below` altitude | descent | No idle thrust row is emitted for that mass. |
| Duplicate row at the same (flight level, mass, Mach) | cruise | (Off by default) The labelled row overrides the swept row and the warning lists the columns that disagree. |
| Missing mach number for a reference performance point | cruise |  Put a NaN for the missing Mach number in the `cruise_reference_mach` table |
| Halted block with fewer than three rows above the crossover altitude, or fewer than two interval fuel flows | climb | No row is extrapolated to the halt altitude, with one warning. The block stops at its last row. |
| Halted block whose rate of climb, net thrust, drag or fuel flow at the halt altitude extrapolates to zero or less | climb | No row is extrapolated to the halt altitude, with one warning. The block stops at its last row. |

Due to rounding in the PIANO output, a `(fl, mass, Mach)` triplet might be
the same for a row coming from the PIANO Mach sweep, and a row coming from
one of the operating points but with slightly differing performance values.
In this case, the operating point value is kept and the Mach sweep value is
dropped.

Rows that warn due to not containing the fifteen numeric columns are generally normal behaviour that occurs when PIANO indicates that the aircraft is not in a valid state. (e.g.
`Dropping X cruise line(s) that open a data row but do not hold 15 numeric columns, ...`).
Similarly, warning about a reference (labelled) performance point not having a corresponding Mach is also usually normal behaviour when PIANO indicates that the aircraft is not in a valid state.
(e.g. `Setting a NaN Mach in X incomplete cruise reference Mach group(s), at (fl, mass) of ...`).

### Errors

The reader raises `ValueError` in the following cases.

| Case | File |
|------|------|
| Line is a data row but does not contain exactly seven (climb) or six (descent) numbers | climb, descent |
| The file holds no block marker | climb, descent |
| The file has no `Cruise table for` title | cruise |
| The fourth column is neither `\|` nor a known reference Mach label | cruise |
| `Delta-ISA` is non-zero | climb |
| A block has no `Initial mass` header and no mass was supplied | climb |
| The supplied mass count does not match the block count | climb |
| No block states an airspeed schedule and no schedule was supplied | climb |
| The overrides supply only part of the climb airspeed schedule | climb |
| The crossover altitude derived from the supplied schedule is below FL100 | climb |
| A block has no `Mass` header | descent |
| A block has no `Airspeed schedule` header | descent |
| Two blocks state different airspeed schedules | descent |

## Cross-checks

The reader emits warnings on sanity checks of the parsed values.

```{admonition} Question
Are these useful, and should the tolerances be lower?
```

**True airspeed.** The derived true airspeed is compared against the block's
own distance and time columns, as
{math}`\sqrt{(\Delta d / \Delta t)^2 + \mathrm{ROCD}^2}`.
The check uses the median relative deviation over
the block, with a tolerance of 5%.

**Fuel burn.** The final cumulative burn of a block is compared against the
`Fuel burn` total in its header, with a tolerance of 0.1%. The check is skipped
if the header is not present.

**Climb mass.** When the caller supplies climb masses and a block header also
has a value, the two are compared with a tolerance of 0.1%.

**Climb crossover altitude.** The crossover altitude in the climb
`Airspeed schedule` header is compared against the altitude at which its high
CAS equals its Mach number in the standard atmosphere, with a tolerance of
0.1%.

**Climb airspeed schedule.** When the caller supplies a climb airspeed
schedule and a block header also has one, the low CAS, the high CAS and the
Mach number are compared with a tolerance of 0.1%.
