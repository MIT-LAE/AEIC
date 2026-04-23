# Legacy verification data

Reference outputs from the MATLAB AEIC v2.1 implementation, used by
`test_matlab_verification.py` to check that the Python AEIC trajectory
and emissions results agree with the MATLAB code to within 0.25 % MAPE.

## Missions

Six B738 city-pairs defined in `missions.toml`:

| Origin | Destination | Departure (UTC) |
|--------|-------------|-----------------|
| BOS    | LAX         | 2019-01-01 12:00 |
| SFO    | ATL         | 2019-01-01 06:00 |
| JFK    | ORD         | 2019-01-01 13:00 |
| BOS    | MIA         | 2019-01-01 05:45 |
| SJC    | LAX         | 2019-01-01 09:40 |
| PHX    | IAD         | 2019-01-01 09:40 |

Performance model: `performance-model.toml` (Boeing 737-800 / CFM56-7B).
Fuel: `fuel.toml` (conventional Jet-A).

## Directory layout

```
matlab-output/          Per-mission CSVs consumed by LegacyTrajectory
matlab-output-orig/     Raw combined MATLAB output (two CSV files)
```

### `matlab-output-orig/`

Two large CSV files produced by running MATLAB AEIC v2.1 on the six
missions above:

- `AEIC_OUTPUT_TRAJ_20260305_114449944.csv` — trajectory fields
  (columns: `AC`, `airportDepart`, `airportArrive`, `t`, `fuelFlow`,
  `acMass`, `horDist`, `lat`, `long`, `az`, `TAS`, `alt`, `roc_fpm`)
- `AEIC_OUTPUT_EMIS_20260305_114449944.csv` — per-point emission
  indices (same grouping columns plus `EI_CO2`, `EI_H2O`, `EI_HC`,
  `EI_CO`, `EI_NOx`, `EI_SOx`)

The timestamp `20260305_114449944` encodes the MATLAB run date
(2026-03-05 11:44:49).

### `matlab-output/`

Per-mission CSV files derived from the combined originals by splitting
on `(airportDepart, airportArrive)` and merging trajectory + emissions
columns into the format expected by `LegacyTrajectory`:

```
t, fuelFlow, acMass, horDist, lat, long, az, TAS, alt, roc_fpm,
EI_CO2, EI_H2O, EI_HC, EI_CO, EI_NOx, EI_SOx
```

Committed in git commit `af7e6aa` (MATLAB verification: BFFM2 bug and
starting mass, PR #112).
