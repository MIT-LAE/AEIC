# Weather test fixture

`2024-09-01.nc` is an ERA5 reanalysis slice used by
`test_compute_ground_speed` (and the broader `test_weather.py` fixture
`sample_weather`) to exercise `Weather.get_ground_speed` against real
atmospheric data.

## Provenance

- **Source:** ERA5 (ECMWF — European Centre for Medium-Range Weather
  Forecasts), hourly pressure-level analysis.
- **Valid time:** 2024-09-01 04:00 UTC (single time slice, stored
  without a `valid_time` dimension so `Weather` reads it as a daily mean).
- **Downloaded:** 2025-10-28 via cfgrib 0.9.15.0 / ecCodes 2.42.0.

## Spatial coverage

| Dimension        | Range                  | Step    |
|------------------|------------------------|---------|
| Latitude         | 33.00°N – 43.00°N      | 0.25°   |
| Longitude        | 85.00°W – 71.00°W      | 0.25°   |
| Pressure levels  | 225–1000 hPa (22 lvls) | various |

The domain covers the northeastern US / mid-Atlantic region, chosen to
include the BOS→ATL and BOS→JFK routes used by the
`test_trajectory_simulation_weather` tests.

## Variables

| Name | Long name           | Units |
|------|---------------------|-------|
| `u`  | Eastward wind       | m/s   |
| `v`  | Northward wind      | m/s   |
| `t`  | Temperature         | K     |

## Note on `test_compute_ground_speed` expected value

The assertion `gs == pytest.approx(191.02855126751604, rel=1e-4)` was
derived by running the test against this specific file.  The relaxed
tolerance (`rel=1e-4` rather than the default `1e-6`) was chosen when
the pressure-level interpolation was adjusted; the value itself has no
independent notebook provenance (SUSPICIOUS-DATA, Phase 5 High finding).
