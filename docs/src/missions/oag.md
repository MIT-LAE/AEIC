# OAG data access

The `missions.oag` package gives access to OAG flight schedule data. The OAG
data is converted from CSV files in the pre-processed format provided by Carla
into SQLite database files. The database files use a schema optimized for the
query patterns used in AEIC. This optimization is important because the OAG
files contain a large number of records, and we want to be able to filter them
in several different ways.

To understand the way the database is organized, it helps to distinguish
between **flights** and **flight instances**. A **flight instance** is a
single flight between an origin and a destination airport departing at a given
time on a given date. A **flight** represents a sequence of **flight
instances** by defining a schedule of days of the week and departure time
between a pair of effective dates. A **flight** corresponds to a single row in
the input OAG CSV file, and each **flight** may have multiple **flight
instances**. The **flight instances** are reified in the SQLite database to
make querying the schedule of flights more efficient.

## Usage example

Here's a basic usage example to give an idea of how the API works:

```python
# Recommended way to import the OAG package.
import missions.oag as oag

# Open the database.
db = oag.Database('oag-2019.sqlite')

# Build a scheduled flight query.
query = oag.Query(
    # Filter on flight characteristics.
    filter=oag.Filter(
        # Flight distance between 9500 and 10000 kilometers.
        min_distance=9500, max_distance=10000,
        # Seat capacity >= 500.
        min_seat_capacity=500,
        # Flight origin or destination in US or Canada.
        country=['US', 'CA']
    )
)

# Iterate over the flight results: there are only 46 from this query...
for flight in db(query):
    # ... so it's practical just to print some data.
    print(flight.departure.isoformat(), flight.carrier + flight.flight_number)
```

## Reference documentation

The main classes of interest in the API are:

- `oag.Database`: the main database class;
- `oag.Query`: a query that returns a sequence of **flight instances**;
- `oag.QueryResult`: a single result from an `oag.Query` query;
- `oag.FrequentFlightQuery`: a query that returns most frequent
  origin/destination pairs appearing in **flight instances**;
- `oag.FrequentFlightQueryResult`: a single result from an
  `oag.FrequentFlightQuery` query;
- `oag.CountQuery`: a query that counts **flight instances** matching given
  conditions;
- `oag.Filter`: a filter on **flight** characteristics usable with all query
  types.

### Database class

The `missions.oag.Database` class is a wrapper around a connection to an
SQLite database file (using the Python standard library's
[`sqlite3`](https://docs.python.org/3/library/sqlite3.html) package). The
`Database` class hides the details of both the database structure and the
underlying SQL interface to SQLite, instead exposing a simple
application-specific query API.

The normal workflow for querying the OAG database is to create a `Database`
instance, passing the path to the SQLite database file to the constructor:

```python
db = oag.Database('oag-2019.sqlite')
```

The `Database` object is callable, and when you call it with query objects
(see below), it returns a Python
[generator](https://realpython.com/introduction-to-python-generators/) that
you can iterate over to get individual results:

```python
for flight in db(oag.Query()):
    print(flight.carrier + flight.flight_id)
```

(Don't try to run this code! It will iterate through *every* **flight
instance** in the database in departure time order. An empty `Query` selects
all **flight instances**.)

```{eval-rst}
.. autoclass:: missions.oag.Database
   :special-members: __init__, __call__
```

### Queries

Database queries come in three flavors. **Flight instance** queries,
represented by the `oag.Query` class, return **flight instances** in departure
time order, filtered by various criteria. Frequent flight queries, represented
by the `oag.FrequentFlightQuery` class, return pairs of origin and destination
airports that have the most flights between them, again filtered by various
criteria. Count queries, represented by the `oag.CountQuery` class, count the
number of **flight instances** that match given conditions.

The filtering criteria for the different query types share some features in
common, so the query classes are derived from a `QueryBase` base class. Each
of the query classes has a `RESULT_TYPE` member that gives the type of the
results returned when you run one of these queries.

#### Base query class

The base query class includes filter parameters for the **flight instance**
start and end dates to consider, as well as an `oag.Filter` value that filters
on **flight** characteristics (like origin and destination, distance, etc.).

```{eval-rst}
.. autoclass:: missions.oag.query.QueryBase
   :inherited-members: filter, start_date, end_date
   :exclude-members: to_sql
```

#### Scheduled flight queries

The `oag.Query` class returns individual **flight instances** in departure
time order, corresponding to a given set of filter conditions. This query type
supports **flight** characteristics filtering (using `oag.Filter`), start and
end date filtering (from `oag.QueryBase`) and random and "every nth day"
sub-sampling (using the `sample` and `every_nth` parameters).

These queries return results as a generator of `oag.query.QueryResult` values,
each of which basically contains all of the known information about the
**flight instances**.

The following examples illustrate some uses of `oag.Query`.

Return all **flight instances** for all **flights** with a distance between
1000 and 5000 kilometers:

```python
q = oag.Query(filter=oag.Filter(min_distance=1000, max_distance=5000))
```

Return a random 5% sample of **flight instances** for all flights between
France and China:

```python
q = oag.Query(filter=oag.Filter(country=['FR', 'CN']), sample=0.05)
```

Return all 787 **flight instances** from France to China departing every 8th
day starting on March 1 2019:

```python
q = oag.Query(
    filter=oag.Filter(
        origin_country='FR',
        destination_country='CN',
        aircraft_type='787'
    ),
    start_date=date(2019, 3, 1),
    every_nth=8
)
```

```{eval-rst}
.. autoclass:: missions.oag.Query
   :members: every_nth, sample, RESULT_TYPE
   :exclude-members: to_sql
```

```{eval-rst}
.. autoclass:: missions.oag.query.QueryResult
   :members:
   :exclude-members: from_row
```

#### Frequent flights queries

The `oag.FrequentFlightQuery` class returns airport pairs (discounting the
direction, i.e., BOS → LHR is the same as LHR → BOS) and counts of flights
between them matching a given filter condition. The filter conditions
supported are the same as for **flight instance** queries, i.e. represented by
an `oag.Filter` instance. Results are returned as a generator of
`oag.query.FrequentFlightQueryResult` values, which contain the airport codes
and a count of the number of **flight instances**.

For example, if we want to find the ten most common routes flown by 787s, we
can do:

```python
q = oag.FrequentFlightQuery(filter=oag.Filter(aircraft_type='787'), limit=10)
for f in db(q):
    print(f.airport1, f.airport2, f.number_of_flights)
```

with output

```
JFK LHR 2156
HNL SFO 2109
IAH LHR 1840
BOS LHR 1753
EWR SFO 1664
FRA IAD 1439
LAD LIS 1408
FRA ORD 1342
BKK MNL 1338
BKK HKT 1283
```

```{eval-rst}
.. autoclass:: missions.oag.FrequentFlightQuery
   :members: limit, RESULT_TYPE
   :exclude-members: to_sql
```

```{eval-rst}
.. autoclass:: missions.oag.query.FrequentFlightQueryResult
   :members:
   :exclude-members: from_row
```

#### Count queries

Sometimes we just want a count of the number of **flight instances** matching
a filter. For example, before running some long computation on each **flight
instance**, it's useful to know if there are millions of them... Running an
`oag.CountQuery` query returns a single integer count value, i.e., there is no
generator involved.

For example, if we want to count the total number of 777 **flight instances**
in the database, we can do:

```python
>>> import missions.oag as oag
>>> db = oag.Database('oag-2019.sqlite')
>>> db(oag.CountQuery(filter=oag.Filter(aircraft_type='777')))
108846
```

```{eval-rst}
.. autoclass:: missions.oag.CountQuery
   :members: RESULT_TYPE
   :exclude-members: to_sql
```

### Filters

```{eval-rst}
.. autoclass:: missions.oag.Filter
   :members:
   :exclude-members: to_sql
```

## Database creation

To convert a CSV file containing OAG data to an SQLite database, use the
`convert-oag-data` script like this:

```
uv run convert-oag-data \
  /home/carlau/EmissionsData/OAG_Data/proc_AEIC_ready_UTC_csv/proc_AEIC_ready_UTC_2019.csv \
  oag-2019.sqlite
```

That input file contains 6,169,161 flight records, is 618 Mb in size and
results in an SQLite database that is about 3.1 Gb in size (which is a
perfectly OK size for an SQLite database to be, by the way).

The database contains 6,169,126 **flights** and 38,709,635 **flight
instances**. The difference in the number of flights in the original CSV file
and the database arises from the existence of some flights to non-existent
airports in the input file (specifically, using the fictitious airport codes
QPX and QPY).

The difference in size between the CSV and the SQLite files is explained by
the flight schedule data being expanded into a separate table to enable
quicker queries in the SQLite database, as well as the additional space taken
up by indexes to optimize queries.

## Database schema

The database schema for the OAG database is described [on the GitHub
wiki](https://github.com/MIT-LAE/AEIC/wiki/OAG-database) for AEIC.
