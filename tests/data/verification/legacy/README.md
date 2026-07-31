# Legacy MATLAB verification data

Reference outputs from MATLAB AEIC (archive: https://zenodo.org/records/6461767) used to verify Python AEIC's
legacy B738 trajectory and emissions implementation. Both implementations run
without weather.

## Input missions

`missions.toml` contains 24 B738 flights spanning
268-2269 nmi, across both
hemispheres, high-latitude and high-elevation airports, and a date-line route.

| Range (nmi) | Routes |
|---:|---|
| 268-298 | SJC-LAX, LAX-SFO, LPB-VVI |
| 442-686 | HND-CTS, SIN-CGK, OSL-TOS, JNB-CPT |
| 919-1,452 | DEL-BLR, PEK-SZX, AKL-NAN, ANC-SEA, GUM-NRT, JFK-SJU, GRU-MAO |
| 1,699-1,981 | EZE-LIM, BOG-MEX, HKG-DPS, SFO-ATL, LHR-CAI, HNL-MAJ |
| 2,187-2,269 | ADD-JNB, CPT-NBO, LAX-HNL, BOS-LAX |

The test uses different error metrics for output parameters:

- Performance fields and emission indices: MAPE, tolerance 0.25%.
- Cumulative ground distance: MAE, tolerance 5 nautical miles.
- Position: mean WGS84 point-to-point geodesic separation divided by route
  distance, tolerance 0.25%.
- Azimuth: mean absolute circular angular error, tolerance 0.25 degrees. The
  terminal point is omitted because it has no outgoing leg.
- Point count: exact equality.

Fuel constants are CO2 `3160 g/kg`, H2O `1230 g/kg`, and LHV `43.8 MJ/kg` in
both implementations.

## Consistent inputs

`scripts/generate_matlab_verification_schedule.py` generates consistent inputs
used by MATLAB and Python AEIC:

```bash
uv run python scripts/generate_matlab_verification_schedule.py
```

- `matlab-schedule.csv`: OAG-like MATLAB schedule generated from `missions.toml`.
- `matlab-airports.csv`: 41 selected airports in MATLAB AEIC format,
  using the same coordinates and elevations as Python AEIC.
- `matlab-input.AEIC`: MATLAB AEIC input file used for this run.
- `performance-model.toml`: B738 / CFM56-7B legacy performance model.
- `fuel.toml`: Fuel and emission-index constants.

## Outputs

`matlab-output-orig/` contains the MATLAB AEIC output files:

- `AEIC_OUTPUT_TRAJ_20260731_111714857.csv`: 1,695 rows and 24 missions.
- `AEIC_OUTPUT_EMIS_20260731_111714857.csv`: 1,671 rows and 24 missions.

`matlab-output/` contains the processed 24 per-mission files produced by
`AEIC.verification.legacy.process_matlab_csvs`. These are the files used by
`tests/test_matlab_verification.py`.

Verification run with:

```bash
uv run pytest -q tests/test_matlab_verification.py
```
