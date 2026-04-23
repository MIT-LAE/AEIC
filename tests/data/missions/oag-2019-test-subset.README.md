# oag-2019-test-subset.sqlite

SQLite database built from a subset of the 2019 OAG schedule dataset,
used by `test_query_result` in `test_mission_db.py` to exercise the
`Query`, `Filter`, and `FrequentFlightQuery` classes.

## Provenance

Ingested from a 2019 OAG bulk-schedule extract (full year, global) via
`convert_oag_data`, keeping only schedules with departure dates in 2019
(2019-01-01 through 2019-12-31 UTC) and filtering to a subset of routes.
Row counts were verified against the SQLite shell immediately after
ingestion.

## Table row counts

| Table       | Rows  |
|-------------|------:|
| `flights`   | 1 196 |
| `schedules` | 1 197 |
| `airports`  |   809 |
| `countries` |   137 |

(One flight has two schedule entries — a day of week repeat — giving
1 197 schedules from 1 196 distinct flight records.)

## Query verification

The counts below were verified with SQLite shell queries and are
asserted in `test_query_result`:

| Query description                              | SQL sketch                                           | Expected count |
|------------------------------------------------|------------------------------------------------------|----------------|
| All schedules                                  | `SELECT COUNT(*) FROM schedules` (via `Query()`)     | 1 197          |
| `min_distance=3000` (nm, great-circle)         | `WHERE distance >= 3000`                             |    99          |
| `country='IT'` (either endpoint)               | `WHERE origin_country='IT' OR dest_country='IT'`     |    36          |
| `max_distance=3000, country=['US','CA']`        | combined filter                                      |   307          |
| `FrequentFlightQuery(airport='DTW')` — sum     | `SUM(number_of_flights)` for all DTW OD pairs        |    13          |
