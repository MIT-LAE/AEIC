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
