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

The base `Builder` class, the shared `Options` dataclass, and the
`Context` dataclass are documented below. `LegacyBuilder` is the most
complete reference implementation and the recommended starting point for
anyone writing a new builder. Actually *using* the trajectory builders is
simple, but implementing one is more involved.

```{eval-rst}
.. WARNING::
   The builder API is still settling. There are no helper methods on the
   `Trajectory` or `Builder` classes for the "extra" LTO flight phases
   (taxi, take-off roll, etc.) — those still need to be handled in the
   concrete builder.
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

## Work-in-progress builders

The following builders are re-exported from
{py:mod}`AEIC.trajectories.builders` but are **not yet fully implemented**.
They are listed here so that consumers who notice them in the public API are
aware of their status; expect signatures, behaviour, and option sets to
change as the implementations land.

### TASOPT builder

{py:class}`TASOPTBuilder <AEIC.trajectories.builders.tasopt.TASOPTBuilder>`
will drive trajectory simulation from TASOPT-based performance data.

```{eval-rst}
.. WARNING::
   This builder is a stub and cannot yet fly trajectories end-to-end.
```

### ADS-B builder

{py:class}`ADSBBuilder <AEIC.trajectories.builders.ads_b.ADSBBuilder>`
will reconstruct trajectories from ADS-B track data.

```{eval-rst}
.. WARNING::
   This builder is a stub and cannot yet fly trajectories end-to-end.
```

### Dymos builder

{py:class}`DymosBuilder <AEIC.trajectories.builders.dymos.DymosBuilder>`
will drive trajectory simulation using the Dymos optimal-control library.

```{eval-rst}
.. WARNING::
   This builder is a stub and cannot yet fly trajectories end-to-end.
```
