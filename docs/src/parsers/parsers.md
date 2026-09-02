# Available parsers

The {py:mod}`AEIC.parsers` package contains readers for a number of external
file formats that AEIC consumes when building performance models or
converting reference data. These modules are primarily used by other
commands (in particular {command}`aeic make-performance-model`); most users
will not need to call them directly, but the dataclasses they return are
part of the public contract for performance-model construction.

The readers are grouped by the source of the data:

 * [BADA parsers](bada_parsers.md) read the PTF and OPF files of a BADA
   aircraft data set. The PTF reader is the source of performance-table data
   for [legacy performance
   models](../performance_models/legacy_performance_model.md).
 * The [LTO reader](lto_reader.md) reads AEIC-format landing and take-off
   characteristics files.
 * The [PIANO reader](piano_reader.md) reads the cruise, climb and descent
   text exports of PIANO. It is the source of performance-table data for
   [PIANO performance models](../performance_models/piano_performance_model.md).

% TODO: add a row here for each new parser, and say which model type consumes
% it.

```{toctree}
:maxdepth: 1

bada_parsers.md
lto_reader.md
piano_reader.md
```
