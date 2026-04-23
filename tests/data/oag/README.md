# OAG test extract

`2019-extract.csv` is a hand-curated 29-row subset of a 2019 OAG
schedule file, used by `test_oag_conversion` to exercise
`convert_oag_data`.

## Provenance

Extracted manually from the full 2019 OAG schedule dataset by selecting
a small number of rows that exercise different code paths in
`convert_oag_data` (valid flights, unknown airports, etc.).  The file
format matches the OAG bulk-schedule CSV format: one header row followed
by data rows, with columns including `carrier`, `fltno`, `depapt`,
`arrapt`, `deptim`, `arrtim`, `days`, `acftchange`, `genacft`,
`inpacft`, `service`, `seats`, `distance`, etc.

## Expected conversion result

`test_oag_conversion` asserts:

```python
assert cur.fetchone()[0] == 8   # COUNT(*) FROM flights
```

8 of the 29 rows produce valid flight records after filtering out:

- Rows where origin or destination is not in the airport database.
- Rows with a non-zero stop count (`stops != '00'`).
- Rows with implausible or zero distance.
- Duplicate rows (identified by `flt_dupe` column).

Warnings for rejected rows are written to `oag_warnings.txt` in the
test's `tmp_path`.
