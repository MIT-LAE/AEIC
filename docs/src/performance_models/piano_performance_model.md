# Piano performance model

`model_type = "piano"` is built from PIANO's climb/cruise/descent text
exports, via the {py:class}`PianoPerformanceModel
<AEIC.performance.models.piano.PianoPerformanceModel>` class and the
`aeic make-piano-performance-model` command
({py:mod}`AEIC.commands.make_piano_performance_model`).

It uses the same table shape as the {doc}`legacy <legacy_performance_model>`
(BADA-derived) model -- climb/cruise/descent split, bilinear interpolation in
flight level and mass -- but is a separate model type rather than an
extension of `legacy`, because PIANO can offer more resolution than BADA PTF
files ever do: more than 2-3 masses per phase, and genuine per-mass
fuel_flow/ROCD variation in climb and descent (BADA PTF only ever reports a
single nominal-mass value there). See
{py:class}`PianoPerformanceTable <AEIC.performance.models.piano.PianoPerformanceTable>`.

## Required inputs

None of PIANO's climb/cruise/descent text files carry aircraft metadata like
payload, engine count, or engine identity -- unlike the performance *tables*
(altitude, speeds, fuel flow, ROCD), which the tool extracts automatically,
these have no source anywhere and must be supplied on every invocation.
**There is no default for any of these; a missing or wrong value is silent
until someone notices the resulting model looks off.**

| Field | CLI flag | How to determine it |
|---|---|---|
| Aircraft name | `--aircraft-name` | Free text, documentation only. |
| Aircraft class | `--aircraft-class` | One of `wide`/`narrow`/`small`/`freight`. No auto-detection -- you decide. |
| Maximum payload [kg] | `--maximum-payload` | **Not derivable from PIANO or BADA data** (BADA PTF files don't carry it either -- `aeic make-performance-model legacy` requires the same flag, for the same reason). If you have a PIANO plane-definition file, it may have `*freeze-oew*` (OEW) and a `max-payload/design-payload` ratio, but no absolute design payload -- computing one from pax count x mass-per-pax is an assumption, not a fact, and should be checked against a known-good source before trusting it for a new aircraft. Since there's no way to recover where this number came from later, pass `--maximum-payload-source` (e.g. `"MZFW - OEW from <file>"`) to record it as a comment in the output TOML. |
| Number of engines | `--number-of-engines` | Not parsed from any file; you already know this. |
| APU name (optional) | `--apu-name` | Must exactly match an entry in AEIC's APU database -- the command fails immediately (`APU "..." not found`) if it doesn't. Passing nothing disables APU emissions for this model. |
| Engine UID | `--engine-uid` (with `--lto-source edb --engine-file ...`) | Must be a real UID in the ICAO Emissions Databank (EDB) spreadsheet you point `--engine-file` at. **Prefer the UID that has both gaseous and nvPM data** -- a gaseous-only UID will cause nvPM lookups to fail later, at simulation time, far from this command. |

## Auto-extracted inputs

These come from the PIANO files themselves; you only need to override them
if the parser warns it's guessing, or you deliberately want a different
value than what's in the file (e.g. a real-world operational schedule
instead of PIANO's own design assumption -- if you're tracking both, be
explicit in your own records about which one an override represents, since
nothing in the TOML records that distinction for you).

| Field | Source | Override flag |
|---|---|---|
| Maximum altitude | Highest altitude reached in the climb file | `--max-alt-ft` |
| Climb/cruise/descent performance tables | `--cruise-file`, `--climb-file`, `--descent-file` | n/a -- this is the actual data being converted |
| Descent masses | Extracted automatically from each block's `Mass X.` line in the descent file | **None exists.** Unlike climb, there is no `--descent-masses-lb` flag -- descent blocks self-report their mass, climb blocks don't. |

```{warning}
Override flags are an escape hatch for when auto-parsing gets it wrong, not
a routine field to fill in "to be safe". A hand-typed override and the
source file can silently drift apart if the file is later regenerated and
the override isn't updated to match.
```

## Climb speed schedule: often not actually auto-extracted

`cas_high_kts`, `cas_low_kts`, and `climb_mach` are read from the climb
file's `Airspeed schedule` header line *if it's there* -- but in practice,
**most real PIANO climb exports don't have that line at all** (it depends on
export settings, not aircraft type; checking a sample of 107 real climb
files, only 10 had it). When it's missing, `--cas-high-kts`, `--cas-low-kts`,
and `--climb-mach` must **all** be supplied explicitly, or the command raises
`Climb speed schedule not found in file header ... Pass ... explicitly`
rather than silently guessing a default. Get the real values from wherever
you already track operational speed schedules (e.g. the `op_climb_cas_kts`/
`op_climb_mach` columns in a fleet instructions spreadsheet, if you have
one) -- don't just pick numbers that make the error go away.

`--descent-mach`/`--descent-cas-kts`/`--design-mach` remain pure overrides
(the descent file's per-block header and the cruise table itself are more
reliably present), used only if the parser warns it's guessing or you want a
different value on purpose.

## Climb masses need a manual assist

PIANO climb blocks don't self-report which mass they were run at, so you
must supply `--climb-masses-lb` explicitly, one value per climb block, in
the same (ascending) order as the file. If omitted, the tool falls back to
using the heaviest cruise masses as a rough proxy and logs a warning -- this
is a guess, not a substitute for the real values, and it raises instead of
silently proxying if there aren't even enough cruise masses to cover every
climb block. Passing the wrong *count* (not matching the number of blocks in
the file) raises immediately rather than truncating either list.

## Mach sweep cruise files

If the cruise file has a single design Mach, it's used as-is. If it's a full
Mach sweep (multiple rows per altitude, as PIANO can natively export), the
tool picks, at each altitude, the Mach matching how the aircraft is actually
flown there -- design Mach above the CAS/Mach crossover altitude, the
CAS-equivalent Mach below it -- rather than always using the design Mach
(which would exceed the aircraft's CAS limit at low altitude, and so
misrepresent TAS and fuel flow there). `scripts/piano_cas_equivalent_mach_schedule.py`
computes which (FL, Mach) pairs to actually run through PIANO to build such a
sweep, given an aircraft's design Mach and CAS schedule.

## Example invocation

```shell
aeic make-piano-performance-model \
  --output-file B738.toml \
  --cruise-file B738_cruise.txt \
  --climb-file B738_climb.txt \
  --descent-file B738_descent.txt \
  --aircraft-name B738 \
  --aircraft-class narrow \
  --number-of-engines 2 \
  --maximum-payload 22422 \
  --maximum-payload-source "MZFW - OEW from B738 plane-definition file" \
  --climb-masses-lb 113392.4,151091.4,179394.6 \
  --climb-mach 0.780 \
  --cas-high-kts 300 \
  --cas-low-kts 250 \
  --lto-source edb \
  --engine-file engines/sample_edb.xlsx \
  --engine-uid 01P11CM121
```

```{note}
`--climb-mach`/`--cas-high-kts`/`--cas-low-kts` are shown explicitly here
because, per the note above, most real climb files don't have a header to
read them from -- if yours does, they're optional.
```

## API reference

```{eval-rst}
.. autoclass:: AEIC.performance.models.piano.PianoPerformanceModel
   :members:
   :exclude-members: model_config, model_type, validate_pm
```

```{eval-rst}
.. autoclass:: AEIC.performance.models.piano.PianoPerformanceTable
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.parsers.piano_reader.PianoData
   :members:
```
