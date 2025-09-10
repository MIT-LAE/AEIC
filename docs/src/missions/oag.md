# OAG flight data

Currently the primary source of scheduled flight data for AEIC is OAG data.
This is converted from the CSV files provided by OAG into AEIC mission
databases using code in the `missions.oag` package. This contains classes
representing records from OAG CSV files and a specialized database class
(`OAGDatabase`) to generate the database entries needed to represent OAG data.

In normal use, these classes will mostly only be used via that
`convert-oag-data` utility. Once a mission database has been created from OAG
data, it can be accessed using the classes in the top-level `missions`
package.

## Database creation

To convert a CSV file containing OAG data to an AEIC SQLite mission database,
use the `convert-oag-data` script like this:

```
uv run convert-oag-data -i 2024.csv -y 2024 -d oag-2024.sqlite -w warnings-2024.txt
```

The options to the `convert-oag-data` script are:

 - `-i` / `--in-file` (required): input OAG CSV file;
 - `-d` / `--db-file` (required): output SQLite database file;
 - `-y` / `--year` (required): input data year;
 - `-w` / `--warnings-file` (optional): additional output file for data
   quality control warnings (defaults to standard output).

### Input record filtering

Rows from the input OAG CSV file are ignored if any of the following
conditions apply:

 - The row is for a flight that is not of IATA service type J, S, or Q. These
   three represent "normal" passenger flights.
 - The row is for a flight with stops. The OAG file format includes rows for
   all sub-flights of multi-leg flights, but we only want to include the
   direct flights in the mission database.
 - The row is for a "non-operating" carrier. These rows record details of
   duplicate flights that are operating by one carrier but labelled as flights
   of another carrier. Excluding non-operating carrier flights means that we
   do not include duplicate flights in the mission database.
 - The row is for a non-aviation equipment type. The OAG CSV files include
   many legs that are not flights, and these can be excluded by ignoring rows
   for a fixed list of generic equipment types.

In addition, no mission database records are created, and a warning is
recorded, for any rows from the input CSV file that:

1. Include unknown airport codes. This can occur for historical airports that
   have closed, or for some special IATA airport codes that are not real
   airports.
2. Contain a suspicious great circle distance between origin and destination
   airports. In this context, "suspicious" means either too small (less than 1
   kilometer), or not matching the calculated great circle distance between
   the origin and destination airports (an absolute difference of at least 50
   km and relative difference of more than ±10%).

The suspicious distance check helps to identify cases where airports are
miscoded, either in the sense of their locations not being recorded correctly,
or the OAG data including the wrong IATA code for an airport.

### File sizes

The 2024 OAG CSV file contains 3,609,388 relevant flight records, i.e., flight
records that pass the filtering described above, is 618 Mb in size and results
in an SQLite database that is about 1.5 Gb in size (which is a perfectly OK
size for an SQLite database to be, by the way).

The database contains 3,609,322 **flights** and 17,362,184 **flight
instances**. The difference in the number of flights in the original CSV file
and the database arises from the existence of some flights to non-existent
airports in the input file (specifically, using the fictitious airport codes
QPX and QPY).

The difference in size between the CSV and the SQLite files is explained by
the flight schedule data being expanded into a separate table to enable
quicker queries in the SQLite database, as well as the additional space taken
up by indexes to optimize queries.
