"""Filter airports.csv to rows whose iata_code appears in test-airport-list.txt."""

import csv
from pathlib import Path

HERE = Path(__file__).parent

codes_file = HERE / "test-airport-list.txt"
input_csv = HERE / "airports.csv"
output_csv = HERE / "test-airports.csv"

iata_codes = {
    line.strip() for line in codes_file.read_text().splitlines() if line.strip()
}

# Read all raw lines so we can write them verbatim while parsing with csv.reader
raw_lines = input_csv.open(newline="").read().splitlines(keepends=True)
header_line = raw_lines[0]
data_lines = raw_lines[1:]

reader = csv.reader(data_lines)
header = next(csv.reader([header_line]))
iata_col = header.index("iata_code")

matched = 0
with output_csv.open("w", newline="") as f_out:
    f_out.write(header_line)
    for raw_line, row in zip(data_lines, reader):
        if row[iata_col] in iata_codes:
            f_out.write(raw_line)
            matched += 1

print(f"Wrote {matched} matching rows to {output_csv}")
