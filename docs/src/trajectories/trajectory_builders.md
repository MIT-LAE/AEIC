# Trajectory builders

This is a system of organization for the different classes that do trajectory
simulation.

The basic idea here is to separate the logic of flying trajectories from the
data container representing a trajectory (the `Trajectory` class and the
associated idea of field sets), and to allow for easy swapping of different
trajectory building algorithms (the current AEIC v2 "legacy" one, and later
TASOPT-based, simulations based on ADS-B data, or whatever). The builder API
has a base abstract `Builder` class, plus a concrete class derived from that
for each type of trajectory builder. (The only one of these that's fully
implemented so far is `LegacyBuilder`, which does AEIC v2 trajectories.) Each
builder class has an options class that wraps up any options that the builder
supports. The main method used to simulate a trajectory is called `fly`: you
pass a performance model and a mission and get back a trajectory.

The part of all this that probably requires some explanation is the `Context`
classes. Each of the trajectory builders will have state that it needs to keep
track of during the calculations that go on in the `fly` method and the
subsidiary methods that `fly` calls. Instead of either a). threading lots of
parameters through all the function calls; or b). having a bunch of instance
variables on the builder classes that have poorly defined lifecycles and
initialization states (lots of things being set to `None` in constructors, for
example), there is the idea of a "context", which is a separate object which
maintains all of the required state in one place. Because there is a single
context object whose lifecycle is bounded by any particular call to the `fly`
method, managing the lifecycle is easy. To avoid saying
`self.context.whatever` all over the place, there is some more Python magic
using `__getattr__` to route attribute accesses on the builder to the context
instead. It ends up being quite comfortable to program against. (One
disadvantage is that it isn't thread-safe: if you share a trajectory builder
across threads, the context will get mixed up. But trajectory builders are
lightweight things, so it's not a problem to create them in each thread where
you need one.)

You can see how all of this works together in the
`tests/test_trajectory_simulation.py` test file. There's a test function there
that simulates a list of sample missions from an input file, saves them to a
NetCDF file and reloads them.

A trajectory builder class should have a `fly_...` method for each flight
phase that it implements, e.g. `fly_climb`, `fly_cruise`, `fly_descent` at a
minimum. These methods are passed the `Trajectory` being constructed, plus a
set of additional builder-specific keyword arguments passed through from
`fly`, used for any builder-specific specials.

Documentation for the base `Builder` class is currently sparse, and the best
reference to how to use these things is to look at the `LegacyBuilder`
implementation. Actually *using* the trajectory builders is simple, but
implementing a builder is more complicated!

```{eval-rst}
.. WARNING::
   Some of this code is definitely unfinished. There needs to be a way to
   incrementally extend trajectories instead of having to specify the length
   of the trajectory up-front, and there needs to be a smoother API for the
   builders to save trajectory data for different flight phases. At the
   moment, there are no helper methods to deal with the "extra" LTO flight
   phases at all.
```

```{eval-rst}
.. autoclass:: AEIC.trajectories.builders.base.Builder
   :members:
```

## Legacy trajectory builder

The legacy trajectory relies on
[BADA-3](https://www.eurocontrol.int/model/bada)-like performane data in the
AEIC performance data format. Specifically, it requires data that has
prescribed climb and descent profiles, as well as cruise data at a single
altitude ($7000\,\text{ft}$ below operating ceiling).

```{eval-rst}
.. automodule:: AEIC.trajectories.builders.legacy
   :members:
```
