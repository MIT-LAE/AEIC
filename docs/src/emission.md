# Emissions module

The {py:func}`compute_emissions <AEIC.emissions.emission.compute_emissions>`
function in the {py:mod}`AEIC.emissions` module uses a
{py:class}`PerformanceModel <AEIC.performance.models.PerformanceModel>`, a
{py:class}`Fuel <AEIC.types.Fuel>` definition and a flown
{py:class}`Trajectory <AEIC.trajectories.trajectory.Trajectory>` to compute
emissions for the entire mission. It layers multiple methods for emission
calculations from user choices in the configuration file.

## Overview

- Computes trajectory, LTO, APU, GSE, and life-cycle {math}`\mathrm{CO_2}`.
- Emits per-species emission indices (grams per kilogram of feul) and emission
  values (grams) wrapped in an {py:class}`EmissionsOutput
  <AEIC.emissions.EmissionsOutput>` class for downstream analysis.

## Configuration inputs

The `[emissions]` section of the configuration TOML file is validated through
{py:class}`EmissionsConfig <AEIC.config.emissions.EmissionsConfig>`. Keys and
meanings are summarised below.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Allowed values
     - Description
   * - ``fuel``
     - any fuel name matching ``fuels/<name>.toml``
     - Selects the fuel file used for EIs and life-cycle data.
   * - ``climb_descent_mode``
     - ``trajectory`` / ``lto``
     - When ``trajectory``, the emissions are calculated over all segments of
       the trajectory; otherwise only the cruise part of the trajectory is
       used and climb, descent and approach emissions are calculated from LTO
       data.
   * - ``co2_enabled`` / ``h2o_enabled`` / ``sox_enabled``
     - ``true`` / ``false``
     - Toggles calculation of fuel-dependent, constant EI species.
   * - ``nox_method``
     - ``BFFM2`` / ``P3T3`` / ``none``
     - Selects the method for {math}`\mathrm{NO_x}` calculation (None disables calculation).
   * - ``hc_method`` / ``EI_CO_method``
     - ``BFFM2`` / ``none``
     - Selects the method for HC/CO calculation (None disables calculation).
   * - ``pmvol_method``
     - ``fuel_flow`` / ``FOA3`` / ``none``
     - Chooses the PMvol method.
   * - ``pmnvol_method``
     - ``meem`` / ``scope11`` / ``FOA3`` / ``none``
     - Chooses the PMnvol method.
   * - ``apu_enabled`` / ``gse_enabled`` / ``lifecycle_enabled``
     - ``true`` / ``false``
     - Enables non-trajectory emission sources and life-cycle {math}`\mathrm{CO_2}` adjustments.
```

## Usage example

```python
import tomllib

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
mission = perf.missions[0]

missions_file = config.file_location('missions/sample_missions_10.toml')
with open(missions_file, 'rb') as f:
    mission_dict = tomllib.load(f)
mission = Mission.from_toml(mission_dict)[0]

with open(config.emissions.fuel_file, 'rb') as fp:
    fuel = Fuel.model_validate(tomllib.load(fp))

traj = Trajectory(perf, mission, optimize_traj=True, iterate_mass=False)
traj.fly_flight()

output = compute_emissions(perf, fuel, traj)

print("Total CO2 (g)", output.total[Species.CO2])
print("Taxi NOx (g)", output.lto[Species.NOx][ThrustMode.IDLE])
print("Per-segment PM number", output.trajectory[Species.PMnvol])
```

## Inner containers

The module defines dataclasses that document both inputs and
outputs of the computation:

- {py:class}`EmissionsConfig`: user-facing configuration parsed from the TOML
  file. It validates enums ({py:class}`LTOInputMode`, {py:class}`EINOxMethod`,
  {py:class}`PMvolMethod`, {py:class}`PMnvolMethod`), resolves defaults, and
  ensures databank paths are present when required.
- {py:class}`EmissionSettings`: flattened, runtime-only view of the above. It
  keeps booleans for metric flags, file paths, and LTO/auxiliary toggles so
  subsequent runs avoid re-validating the original mapping.
- {py:class}`AtmosphericState`: carries temperature, pressure, and Mach arrays
  that emission-index models reuse when HC/CO/{math}`\text{NO}_x`/PM need
  ambient conditions.
- {py:class}`EmissionSlice`: describes any source (trajectory, LTO, APU, GSE).
  It stores ``indices`` (emission indices in g/kg) and the realized
  ``emissions_g``.
- {py:class}`TrajectoryEmissionSlice`: extends ``EmissionSlice`` with
  ``fuel_burn_per_segment`` (kg) and ``total_fuel_burn`` (kg) so users can
  derive intensity metrics.
- {py:class}`EmissionsOutput`: top-level container returned by
  :meth:`Emission.emit`. It exposes ``trajectory``, ``lto``, ``apu``, ``gse``,
  ``total`` (summed structured array), and optional ``lifecycle_co2_g``.

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
3.

1. `EmissionsConfig` is materialized from `PerformanceModel.config.emissions`
   and converted to `EmissionSettings`.
2. Fuel properties are read from `fuels/<Fuel>.toml`. These provide
   {math}`\mathrm{CO_2}`/{math}`\mathrm{H_2O}`/{math}`\mathrm{SO_x}` emission
   indices, and life-cycle factors.
3. `emit(traj)` resets internal arrays sized to the trajectory steps
4. {py:meth}`Emission.get_trajectory_emissions` computes EI values for each mission point:
   - Constant EI species ({math}`\mathrm{CO_2}`, {math}`\mathrm{H_2O}`, {math}`\mathrm{SO}_x`).
   - Methods for HC/CO/{math}`\mathrm{NO_x}`/PMvol/PMnvol applied according to user specification.
5. {py:meth}`Emission.get_LTO_emissions` builds the ICAO style landing and
   take off emissions using the per-mode inputs embedded in the performance
   file.
6. {py:func}`AEIC.emissions.apu.get_APU_emissions` and
   {py:func}`AEIC.emissions.gse.get_GSE_emissions` contributions are added if
   enabled.
7. {py:meth}`Emission.sum_total_emissions` aggregates each pollutant into
   `self.summed_emission_g` and, when requested, life-cycle
   {math}`\mathrm{CO_2}` is appended via
   {py:meth}`Emission.get_lifecycle_emissions`.

## Types

### `Species`

The {py:enum}`Species <AEIC.types.Species>` enumerated type lists the chemical
species known to AEIC.

```{eval-rst}
.. autoenum:: AEIC.types.Species
   :members:
```

### `EmissionsOutput`

The {py:class}`EmissionsOutput <AEIC.emission.emission.EmissionsOutput>` class
holds emission index and emission quantities for trajectory, LTO, APU, GSE and
total emissions, as well as some ancillary quantities like fuel burn per
segment. The emission indices and emission quantities are stored as values of
the generic type {py:class}`EmissionsDict
<AEIC.emissions.types.EmissionsDict>`, with a value type of `float` (for APU,
GSE and total emissions), {py:class}`ModeValues <AEIC.types.ModeValues>` for
LTO, and `np.ndarray` for trajectory emissions. This structure captures the
different types of per-species emissions from the different sources.

```{eval-rst}
.. autoclass:: AEIC.emissions.emission.EmissionsOutput
   :members:
```

### `EmissionsDict`


```{eval-rst}
.. autoclass:: AEIC.emissions.emission.EmissionsDict
   :members:
```

### `ModeValues`

```{eval-rst}
.. autoclass:: AEIC.types.ModeValues
   :members:
```



## API reference

```{eval-rst}
.. autofunction:: AEIC.emissions.compute_emissions
```


## Helper functions

```{eval-rst}
.. autofunction:: AEIC.emissions.apu.get_APU_emissions
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
