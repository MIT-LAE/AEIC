# Legacy MATLAB verification data

Reference outputs from MATLAB AEIC (archive: https://zenodo.org/records/6461767) used to verify Python AEIC's legacy B738 trajectory and emissions implementation.
The comparison uses trajectories produced with a shared wind field.

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

- Performance fields and emission
  indices: MAPE, tolerance 0.15%.
- Cumulative ground distance: MAE, tolerance 5 nautical miles.
- Position: mean WGS84 point-to-point geodesic separation divided by route
  distance, tolerance 0.15%.
- Azimuth: mean absolute circular angular error, tolerance 0.2 degrees. The
  terminal point is omitted because it has no outgoing leg.
- Heading: mean absolute circular angular error, tolerance 0.2 degrees.
- Point count: exact equality.

MATLAB AEIC's spherical route stepping can overshoot the destination and add a
single 180-degree corrective leg that is absent from Python's geodesic route.
Ground speed and heading remain direct stored outputs, but the endpoint of that
unmatched corrective leg is excluded from their metrics.

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
- `matlab-input.AEIC`: MATLAB AEIC input file used for the shared-wind run.
- `performance-model.toml`: B738 / CFM56-7B legacy performance model.
- `fuel.toml`: Fuel and emission-index constants.

## Shared weather

`weather/verification-wind.nc` is a 2 x 2 x 2 global NetCDF fixture with a
spatially and vertically uniform wind:

- eastward wind (`u`): `10 m/s`
- northward wind (`v`): `5 m/s`
- latitude bounds: `-90` and `90` degrees
- longitude bounds: `-180` and `180` degrees
- pressure bounds: `200` and `1100 hPa`

The constant field makes MATLAB AEIC's nearest-neighbor sampling and Python's linear interpolation return the same vector.

## Outputs

`matlab-output-orig/` contains the MATLAB AEIC output files:

- `AEIC_OUTPUT_TRAJ_20260731_153621930.csv`: 1,695 rows and 24 missions.
- `AEIC_OUTPUT_EMIS_20260731_153621930.csv`: 1,671 rows and 24 missions.

`matlab-output/` contains the corresponding 24 processed weather-enabled
per-mission files.

Verification run with:

```bash
uv run pytest -q tests/test_matlab_verification.py
```
