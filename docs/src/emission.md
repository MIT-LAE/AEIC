# Emissions module

The {py:func}`compute_emissions <AEIC.emissions.emission.compute_emissions>`
function in the {py:mod}`AEIC.emissions` module uses a
{py:class}`PerformanceModel <AEIC.performance.models.PerformanceModel>`, a
{py:class}`Fuel <AEIC.types.Fuel>` definition and a flown
{py:class}`Trajectory <AEIC.trajectories.trajectory.Trajectory>` to compute
emissions for the entire mission.

Emissions are computed for the flown trajectory, LTO operations, APU use, GSE
use, as well as life-cycle CO₂ emissions, for all requested species. Emissions
methods for different species are controlled by options in the `[emissions]`
section of the configuration data (represented by the
{py:class}`EmissionsConfig <AEIC.config.emissions.EmissionsConfig>` class).

The output from the emissions calculations include per-species emission
indices (grams per kilogram of fuel) and emission values (grams), all wrapped
in a single {py:class}`EmissionsOutput <AEIC.emissions.EmissionsOutput>` value
for downstream analysis.

## Usage example

```python
import tomllib

import AEIC.trajectories.builders as tb
from AEIC.config import Config, config
from AEIC.performance.models import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from AEIC.missions import Mission
from AEIC.emissions import compute_emissions
from AEIC.types import Fuel, Species, ThrustMode

Config.load();

perf = PerformanceModel.load(config.file_location(
    'performance/sample_performance_model.toml'
))

missions_file = config.file_location('missions/sample_missions_10.toml')
with open(missions_file, 'rb') as f:
    mission_dict = tomllib.load(f)
mission = Mission.from_toml(mission_dict)[0]

with open(config.emissions.fuel_file, 'rb') as fp:
    fuel = Fuel.model_validate(tomllib.load(fp))

builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
traj = builder.fly(perf, mission)

output = compute_emissions(perf, fuel, traj)

print("Total CO2 (g)", output.total[Species.CO2])
print("Taxi NOx (g)", output.lto[Species.NOx][ThrustMode.IDLE])
print("Per-segment PM number", output.trajectory[Species.PMnvol])
```

## Computation workflow

The `compute_emissions` function calculates emissions for a given trajectory,
based on a specific performance model and fuel. It performs the following
steps:

1. Calculate fuel burn per segment along the trajectory from the fuel mass
   values provided in the trajectory.
2. Calls the `AEIC.emissions.trajectory.get_trajectory_emissions` function to
   calculate per-segment emission indices and emission values along the
   trajectory. (If the emissiosn configuration flag `climb_descent_mode` is
   set to `lto`, trajectory emissions are only returned for the cruise phase
   of the flight.)
   - For some species (CO₂, H₂O, SO₂, SO₄), constant fuel-dependent emission
     index values are used.
   - For other species (NOₓ, volatile and non-volatile particular matter),
     emissions are calculated using the user-specified calculation method.
   - NOₓ emissions are divided into NO₂, NO and HONO emissions using fixed
     speciation ratios.
3. Calls the `AEIC.emissions.lto.get_LTO_emissions` function to calculate
   ICAO-style landing and take off emissions using the per-mode inputs
   embedded in the performance file. (If the emissiosn configuration flag
   `climb_descent_mode` is set to `trajectory`, LTO emissions are not returned
   for the climb and approach phases of the flight, since these are included
   in the trajectory emissions.)
4. If APU emissions are requested and the performance model provides an APU
   definition, calls the `AEIC.emissions.apu.get_APU_emissions` function to
   calculate APU emissions.
5. If GSE emissions are requested, calls the
   `AEIC.emissions.gse.get_GSE_emissions` function to calculate GSE emissions.
6. Combines all emissions for each chemical species to produce full trajectory
   totals.
7. If requested, calculates a lifecycle CO₂ emissions adjustment and applies
   it to the totals.
8. Collects together all emissions data into a single
   {py:class}`EmissionsOutput <AEIC.types.EmissionsOutput>` value and returns
   it.

```{eval-rst}
.. autofunction:: AEIC.emissions.compute_emissions
```

## Types

### Chemical species

The {py:enum}`Species <AEIC.types.Species>` enumerated type lists the chemical
species known to AEIC.

```{eval-rst}
.. autoenum:: AEIC.types.Species
```

### Emissions output

The {py:class}`EmissionsOutput <AEIC.emission.emission.EmissionsOutput>` class
holds emission index and emission quantities for trajectory, LTO, APU, GSE and
total emissions, as well as some ancillary quantities like fuel burn per
segment. The emission indices and emission quantities are stored as values of
the generic type {py:class}`EmissionsDict <AEIC.types.EmissionsDict>`, with a
value type of `float` (for APU, GSE and total emissions),
{py:class}`ModeValues <AEIC.types.ModeValues>` for LTO, and `np.ndarray` for
trajectory emissions. This structure captures the different types of
per-species emissions from the different sources.

```{eval-rst}
.. autoclass:: AEIC.emissions.emission.EmissionsOutput
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.emissions.emission.EmissionsDict
   :members:
```

```{eval-rst}
.. autoclass:: AEIC.types.ModeValues
   :members:
```

## Helper functions

```{eval-rst}
.. autofunction:: AEIC.emissions.trajectory.get_trajectory_emissions
```

```{eval-rst}
.. autofunction:: AEIC.emissions.lto.get_LTO_emissions
```

```{eval-rst}
.. autofunction:: AEIC.emissions.apu.get_APU_emissions
```

```{eval-rst}
.. autofunction:: AEIC.emissions.gse.get_GSE_emissions
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.co2
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.h2o
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.sox
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.hcco
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.nox
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.pmnvol
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.ei.pmvol
   :members:
```

```{eval-rst}
.. automodule:: AEIC.emissions.lifecycle_CO2
   :members:
```
