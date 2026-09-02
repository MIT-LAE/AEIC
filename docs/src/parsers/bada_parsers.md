# BADA parsers

Two readers cover a BADA aircraft data set: {py:mod}`AEIC.parsers.ptf_reader`
for the performance tables, and {py:mod}`AEIC.parsers.opf_reader` for the
operations performance coefficients.

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
