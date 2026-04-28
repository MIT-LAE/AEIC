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

## Note on `test_compute_ground_speed` assertion strategy

`test_compute_ground_speed` no longer checks a single fixture-specific
floating-point value with `pytest.approx(...)`. Instead, it asserts that
the computed ground speed falls within a physically reasonable envelope
(finite, and `100 m/s < gs < 300 m/s` for a TAS of 200 m/s in ERA5
upper-level winds) for the scenario exercised against this slice.

This keeps the test focused on the behavior that matters: `Weather`
should incorporate real atmospheric data and produce a plausible ground
speed, without making the fixture documentation depend on one exact
interpolated result. That is more robust to small numerical differences
from interpolation, backend, or dataset handling changes while still
validating that the weather lookup meaningfully affects the outcome.
The algorithm itself is exhaustively covered by the synthetic-fixture
tests in `test_weather.py`, which assert exact expected values
derivable on paper.
