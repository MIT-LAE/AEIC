# Engine data

## `sample_edb.xlsx`

A small subset of the ICAO Aircraft Engine Emissions Databank (EDB),
used by `test_edb.py` to exercise `EDBEntry.get_engine`.

**Source:** ICAO Aircraft Engine Emissions Databank (EDB).  The EDB is
published by ICAO and updated periodically; this sample was extracted
from a recent revision for testing purposes.  The full EDB is available
at <https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank>.

The workbook follows the standard EDB Excel layout (sheets: `Turbofan &
Turbojet`, `Turboprop`, `APU`) with one or more engine entries.
`EDBEntry.get_engine` identifies engines by the `UID` column (ICAO
unique identifier string).

## `APU_data.toml`

Auxiliary Power Unit (APU) emission factor table used by
`get_APU_emissions`.  Values are fuel-based emission indices (g/kg fuel)
for each species, keyed by APU model / aircraft type.  Provenance:
derived from published APU characterisation data; see the source comments
in the TOML file for per-entry references.
