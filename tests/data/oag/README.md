# OAG test extract

`2019-extract.csv` is a hand-curated 29-row subset of a 2019 OAG
schedule file, used by `test_oag_conversion` to exercise
`convert_oag_data`.

## Provenance

Extracted manually from the full 2019 OAG schedule dataset by selecting
a small number of rows that exercise different code paths in
`convert_oag_data` — both rows that pass `CSVEntry.is_row_valid()` and
are loaded into the database, and rows that are skipped by that
validity check (e.g. non-direct flights with `stops != 0`, non-operating
carrier entries, surface-vehicle services, and non-aircraft equipment
codes in `EXCLUDE_EQUIPMENT`). The file format matches the OAG
bulk-schedule CSV format: one header row followed by data rows, with
columns including `carrier`, `fltno`, `depapt`, `arrapt`, `deptim`,
`arrtim`, `days`, `acftchange`, `genacft`, `inpacft`, `service`,
`seats`, `distance`, etc.

## Expected conversion result

`test_oag_conversion` asserts:

```python
assert cur.fetchone()[0] == 8   # COUNT(*) FROM flights
```

In this extract, 8 of the 29 input rows are converted into flight
records. The remaining rows are silently dropped because they fail
`CSVEntry.is_row_valid()` — they are not direct flights (`stops != 0`),
are flagged as non-operating (`operating == 'N'`), use a surface-vehicle
service code (`service` in `('V', 'U')`), or use an equipment code in
`EXCLUDE_EQUIPMENT`. `is_row_valid()` does not emit warnings for these
rejections.

For this extract, `test_oag_conversion` explicitly asserts that no
`oag_warnings.txt` file is created in the test's `tmp_path` (warnings
are only written for rows that pass `is_row_valid()` but then fail
later distance/airport-resolution checks, none of which occur here).
