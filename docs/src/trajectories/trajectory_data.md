# Trajectory data

The `Trajectory` class is a data container that stores both per-point data for
a trajectory and per-trajectory metadata values.

The data stored in `Trajectory` values is defined by a list of "field sets"
using the `FieldSet` and `FieldMetadata` classes. All trajectories include the
"base" field set (defined in the `BASE_FIELDS` value in the
`AEIC.trajectories.trajectory` module). Trajectories may contain other field
sets in addition to the base field set. Additional field sets are added to a
trajectory using the `add_fields` method.

The `Trajectory` class has a flexible system for saving and accessing
trajectory data: attribute access on a `Trajectory` value is managed with
reference to the field sets included in the trajectory. This means that any
field in the base field set can be accessed directly for a trajectory, as, for
example, `traj.rate_of_climb` or `traj.n_cruise`. The first of these is a
per-point Numpy array value, while the second is an integer metadata value.
Other data types are supported to work with emissions data, and there is an
"emissions" field set (defined in `EMISSIONS_FIELDS` values in the
`AEIC.emissions.emission` module).

This approach is designed to work cleanly with the `TrajectoryStore` class for
saving trajectory data to NetCDF files.

```{eval-rst}
.. NOTE::
   Currently trajectories have to be created with a fixed number of points.
   This matches the way the legacy trajectory builder works, but I will extend
   the class to allow incremental construction of trajectories once that's
   needed.
```

```{eval-rst}
.. WARNING::
   There is more documentation to come here.
```

```{eval-rst}
.. autodata:: AEIC.trajectories.trajectory.BASE_FIELDS
```

```{eval-rst}
.. autodata:: AEIC.emissions.emission.EMISSIONS_FIELDS
```

```{eval-rst}
.. autoclass:: AEIC.trajectories.trajectory.Trajectory
   :members:
   :special-members:
```

## Field sets

Each field in a field set is represented by a `FieldMetadata` value, which
records the type of each field and whether it is a per-point data field
(`metadata = False`) or a per-trajectory metadata field (`metadata = True`).
Additional fields provide NetCDF metadata for the field description and units,
and allow fields to be marked as required, or to be provided with a default
value.

A field set is a collection of `FieldMetadata` records, keyed by the field
name. Field sets are stored by name in a registry, and use an MD5-based hash
to ensure that named field sets from different sources correspond to the same
sets of fields.

```{eval-rst}
.. NOTE::
   We probably need a mechanism for versioning field sets. At the moment,
   adding a new field to the base field set, for example, will invalidate
   files that were created before the new field was added. That's not
   sustainable, so we need to implement some sort of simple version control
   for these things.
```

```{eval-rst}
.. autoclass:: AEIC.storage.field_sets.FieldMetadata
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.storage.field_sets.FieldSet
   :members:
```

```{eval-rst}
.. autoenum:: AEIC.types.storage.Dimension
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.types.storage.Dimensions
   :members:
```

## Flight phases

```{eval-rst}
.. automodule:: AEIC.trajectories.phase
   :members:
```

## Ground tracks

```{eval-rst}
.. WARNING::
   Ground tracks currently only *really* work for great circle routes between
   an origin and a destination. Trajectory builders that need more complex
   ground tracks with intermediate waypoints will need to take account of the
   exceptions that `GroundTrack.step` raises when an attempt is made to step
   past a waypoint. There should probably be an option on creation of a
   `GroundTrack` to determine whether these exceptions are generated: for a
   "dense" ground track, maybe we don't care about stepping past waypoints;
   for a "flight plan" ground track defined by a sequence of navigation aid
   positions, maybe we would like our trajectory to hit each waypoint exactly,
   so we do need to care about the "stepped over a waypoint" exceptions.
```

```{eval-rst}
.. automodule:: AEIC.trajectories.ground_track
   :members:
```
