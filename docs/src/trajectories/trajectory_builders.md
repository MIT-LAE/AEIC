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

## Adjustable legacy trajectory builder

`AdjustableLegacyBuilder` is a variant of `LegacyBuilder` with the same
phase-integration logic and the same default behavior, but with additional
per-flight hooks for changing the assumptions that are normally hard-coded in
the legacy builder. It is intended for sensitivity studies, policy scenarios,
or workflows where the basic AEIC v2 trajectory model is still desired but
where cruise altitude, reserve policy, descent planning, or similar inputs need
to vary by mission.

With no adjustment arguments, `AdjustableLegacyBuilder` should produce the same
trajectory as `LegacyBuilder` for the same mission, performance model, and
builder options.

```python
import AEIC.trajectories.builders as tb

builder = tb.AdjustableLegacyBuilder(
    options=tb.Options(iterate_mass=False),
    legacy_options=tb.LegacyOptions(),
)

traj = builder.fly(performance_model, mission)
```

### Adjustment arguments

Adjustments are passed as keyword arguments to `fly`. Each adjustment may be:

- `None`, which uses the standard legacy default;
- a `float`, which is used directly; or
- a callable, which receives the current adjustable context, the mission, the
  performance model, and any adjustment-specific keyword arguments.

The supported adjustment keywords are:

| Keyword | Units | Legacy default |
| --- | --- | --- |
| `climb_start_altitude` | m | departure airport altitude + $3000\,\text{ft}$ |
| `cruise_altitude` | m | aircraft maximum altitude - $7000\,\text{ft}$ |
| `descent_end_altitude` | m | arrival airport altitude + $3000\,\text{ft}$ |
| `descent_distance` | m | proportional to `des_start_altitude - des_end_altitude` |
| `reserve_fuel` | kg | 5% of nominal trip fuel |
| `divert_distance` | m | 100 NM for flights up to 3 hours, otherwise 200 NM |
| `hold_time` | s | 45 minutes for flights up to 3 hours, otherwise 30 minutes |

For example, fixed values can be used to pin a scenario:

```python
traj = builder.fly(
    performance_model,
    mission,
    climb_start_altitude=1500.0,
    cruise_altitude=9000.0,
    descent_end_altitude=1200.0,
    descent_distance=200_000.0,
    reserve_fuel=1_500.0,
    divert_distance=150_000.0,
    hold_time=30 * 60.0,
)
```

Callable adjustments are useful when the value depends on mission or aircraft
properties:

```python
def cruise_altitude(context, mission, performance):
    # Fly 1000 m below the aircraft ceiling, but never below the climb start.
    return max(context.clm_start_altitude, performance.maximum_altitude - 1000.0)


def reserve_fuel(context, mission, performance, *, fuel_mass):
    # Use a larger reserve fraction for this scenario.
    return 0.10 * fuel_mass


traj = builder.fly(
    performance_model,
    mission,
    cruise_altitude=cruise_altitude,
    reserve_fuel=reserve_fuel,
)
```

The fuel-policy callables receive additional keyword-only inputs:

- `reserve_fuel(..., fuel_mass=...)`, where `fuel_mass` is the nominal trip
  fuel estimate used by the starting-mass calculation;
- `divert_distance(..., approx_time=...)`, where `approx_time` is the nominal
  flight time estimate;
- `hold_time(..., approx_time=...)`, using the same nominal flight time
  estimate.

### Validation and clamping

The adjustable builder preserves the legacy guardrails where possible:

- a climb start altitude at or above the aircraft ceiling is reset to the
  departure airport altitude;
- a cruise altitude below the climb start altitude is raised to the climb start
  altitude;
- a cruise altitude above the aircraft ceiling is lowered to the ceiling;
- a descent end altitude at or above the ceiling is lowered to the ceiling;
- a descent end altitude above the descent start altitude raises a
  `ValueError`;
- a negative descent distance raises a `ValueError`.

```{eval-rst}
.. automodule:: AEIC.trajectories.builders.adjustable_legacy
   :members:
```

## Work-in-progress builders

The following builders are re-exported from
{py:mod}`AEIC.trajectories.builders` but are **not yet fully implemented**.
They are listed here so that consumers who notice them in the public API are
aware of their status; expect signatures, behaviour, and option sets to
change as the implementations land.

Each WIP builder ships with a matching `*Options` dataclass that is also
re-exported from {py:mod}`AEIC.trajectories.builders`. These option
classes are placeholders; their fields and defaults will change as the
corresponding builder is implemented.

### TASOPT builder

{py:class}`TASOPTBuilder <AEIC.trajectories.builders.tasopt.TASOPTBuilder>`
will drive trajectory simulation from TASOPT-based performance data.
Configured via
{py:class}`TASOPTOptions <AEIC.trajectories.builders.tasopt.TASOPTOptions>`.

```{eval-rst}
.. WARNING::
   This builder and its options class are stubs and cannot yet fly
   trajectories end-to-end.
```

### ADS-B builder

{py:class}`ADSBBuilder <AEIC.trajectories.builders.ads_b.ADSBBuilder>`
will reconstruct trajectories from ADS-B track data. Configured via
{py:class}`ADSBOptions <AEIC.trajectories.builders.ads_b.ADSBOptions>`.

```{eval-rst}
.. WARNING::
   This builder and its options class are stubs and cannot yet fly
   trajectories end-to-end.
```

### Dymos builder

{py:class}`DymosBuilder <AEIC.trajectories.builders.dymos.DymosBuilder>`
will drive trajectory simulation using the Dymos optimal-control library.
Configured via
{py:class}`DymosOptions <AEIC.trajectories.builders.dymos.DymosOptions>`.

```{eval-rst}
.. WARNING::
   This builder and its options class are stubs and cannot yet fly
   trajectories end-to-end.
```
