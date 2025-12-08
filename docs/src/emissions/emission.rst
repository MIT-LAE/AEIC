.. _emissions-module:

Emissions Module
================

``AEIC.emissions.emission.Emission`` is the module that uses
:class:`AEIC.performance_model.PerformanceModel` and a flown
:class:`AEIC.trajectories.trajectory.Trajectory` to compute emissions for the
entire mission. It layers multiple methods for emission calculations
from user choices in the configuration file.

Overview
----------

- Computes trajectory, LTO, APU, GSE, and life-cycle :math:`\mathrm{CO_2}`.
- Uses :class:`EmissionsConfig` / :class:`EmissionSettings` objects so
  configuration defaults and switches are enforced before any computation begins.
- Emits structured arrays (grams by species) plus convenience
  containers (``EmissionSlice`` and ``EmissionsOutput``) for downstream analysis.
- Has helper methods such as :meth:`Emission.emit_trajectory` or
  :meth:`Emission.emit_lto` when only a subset is needed.

Configuration Inputs
--------------------

The ``[Emissions]`` section of the configuration TOML file is validated through
:class:`EmissionsConfig`. Keys and meanings are summarised below.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Allowed values
     - Description
   * - ``Fuel``
     - any fuel name matching ``fuels/<name>.toml``
     - Selects the fuel file used for EIs and life-cycle data.
   * - ``climb_descent_usage``
     - ``true`` / ``false``
     - When ``true``, the emissions are calculated over all segments of the trajectory;
       otherwise climb/descent reverts to only LTO.
   * - ``CO2_calculation`` / ``H2O_calculation`` / ``_calculation``
     - ``true`` / ``false``
     - Toggles calculation of fuel-dependent, constant EI species.
   * - ``EI_NOx_method``
     - ``BFFM2`` / ``P3T3`` / ``none``
     - Selects the method for :math:`\mathrm{NO_x}` calculation (None disables calculation).
   * - ``EI_HC_method`` / ``EI_CO_method``
     - ``BFFM2`` / ``none``
     - Selects the method for HC/CO calculation (None disables calculation).
   * - ``EI_PMvol_method``
     - ``fuel_flow`` / ``FOA3`` / ``none``
     - Chooses the PMvol method.
   * - ``EI_PMnvol_method``
     - ``meem`` / ``scope11`` / ``FOA3`` / ``none``
     - Chooses the PMnvol method.
   * - ``LTO_input_mode``
     - ``performance_model`` / ``EDB``
     - Pulls LTO EI/fuel-flow data from the performance tables or EDB file.
   * - ``EDB_input_file``
     - path
     - Required when ``LTO_input_mode = "EDB"`` to locate the ICAO databank file.
   * - ``APU_calculation`` / ``GSE_calculation`` / ``LC_calculation``
     - ``true`` / ``false``
     - Enables non-trajectory emission sources and life-cycle :math:`\mathrm{CO_2}` adjustments.

Usage Example
-------------

.. code-block:: python

   from AEIC.performance_model import PerformanceModel
   from AEIC.trajectories.trajectory import Trajectory
   from AEIC.emissions.emission import Emission

   perf = PerformanceModel("IO/default_config.toml")
   mission = perf.missions[0]

   traj = Trajectory(perf, mission, optimize_traj=True, iterate_mass=False)
   traj.fly_flight()

   emitter = Emission(perf)
   output = emitter.emit(traj)

   print("Total CO2 (g)", output.total['CO2'])
   print("Taxi NOx (g)", output.lto.emissions_g['NOx'][0])

   # Need only the trajectory segment
   segments = emitter.emit_trajectory(traj)
   print("Per-segment PM number", segments.emissions_g['PMnvol'])

Inner Containers
------------------

The module defines dataclasses that document both inputs and
outputs of the computation:

- :class:`EmissionsConfig`: user-facing configuration parsed from the TOML file.
  It validates enums (:class:`LTOInputMode`, :class:`EINOxMethod`,
  :class:`PMvolMethod`, :class:`PMnvolMethod`), resolves defaults, and ensures
  databank paths are present when required.
- :class:`EmissionSettings`: flattened, runtime-only view of the above. It keeps
  booleans for metric flags, file paths, and LTO/auxiliary toggles so subsequent
  runs avoid re-validating the original mapping.
- :class:`AtmosphericState`: carries temperature, pressure, and Mach arrays that
  emission-index models reuse when HC/CO/:math:`\text{NO}_x`/PM need ambient conditions.
- :class:`EmissionSlice`: describes any source (trajectory, LTO, APU, GSE). It
  stores ``indices`` (emission indices in g/kg) and the realized ``emissions_g``.
- :class:`TrajectoryEmissionSlice`: extends ``EmissionSlice`` with
  ``fuel_burn_per_segment`` (kg) and ``total_fuel_burn`` (kg) so users can derive
  intensity metrics.
- :class:`EmissionsOutput`: top-level container returned by :meth:`Emission.emit`.
  It exposes ``trajectory``, ``lto``, ``apu``, ``gse``, ``total`` (summed
  structured array), and optional ``lifecycle_co2_g``.

Computation Workflow
--------------------

The ``Emission`` object is instanced once per performance model:

1. ``EmissionsConfig`` is materialized from
   ``PerformanceModel.config.emissions`` and converted to ``EmissionSettings``.
2. Fuel properties are read from ``fuels/<Fuel>.toml``. These provide :math:`\mathrm{CO_2}`/:math:`\mathrm{H_2O}`/:math:`\mathrm{SO_x}`
   emission indices, and life-cycle factors.
3. ``emit(traj)`` resets internal arrays sized to the trajectory steps
4. :meth:`Emission.get_trajectory_emissions` computes EI values for each mission point:

   - Constant EI species (:math:`\mathrm{CO_2}`, :math:`\mathrm{H_2O}`, :math:`\mathrm{SO}_x``).
   - Methods for HC/CO/:math:`\mathrm{NO_x}`/PMvol/PMnvol applied according to user specification.
5. :meth:`Emission.get_LTO_emissions` builds the ICAO style landing and take off emissions using either
   databank values (``LTO_input_mode = "edb"``) or the per-mode inputs embedded in
   the performance file.
6. :func:`AEIC.emissions.APU_emissions.get_APU_emissions` and
   :meth:`Emission.get_GSE_emissions` contributions are added if enabled.
7. :meth:`Emission.sum_total_emissions` aggregates each pollutant into
   ``self.summed_emission_g`` and, when requested, life-cycle :math:`\mathrm{CO_2}` is appended via
   :meth:`Emission.get_lifecycle_emissions`.

Structured Arrays
-----------------

All emission indices and gram totals share the dtype emitted by the private
``__emission_dtype`` helper. Each field is ``float64``:

``CO2``, ``H2O``, ``HC``, ``CO``, ``NOx``, ``NO``, ``NO2``, ``HONO``,
``PMnvol``, ``PMnvolGMD``, ``PMvol``, ``OCic``, ``SO2``, ``SO4``.

If ``EI_PMnvol_method`` is ``SCOPE11`` or ``MEEM``, the additional ``PMnvolN``
field is emitted. Metric-specific flags (see ``Emission.metric_flags``) determine
which fields are populated; disabled species stay as ``0``, making it easy to filter downstream.

API Reference
-------------

.. autoclass:: AEIC.emissions.emission.EmissionsConfig
   :members:

.. autoclass:: AEIC.emissions.emission.EmissionSettings
   :members:

.. autoclass:: AEIC.emissions.emission.Emission
   :members: __init__, emit, emit_trajectory, emit_lto, emit_apu, emit_gse
   :show-inheritance:

.. autoclass:: AEIC.emissions.emission.EmissionSlice
   :members:

.. autoclass:: AEIC.emissions.emission.TrajectoryEmissionSlice
   :members:

.. autoclass:: AEIC.emissions.emission.EmissionsOutput
   :members:

Helper Functions
------------------

.. automodule:: AEIC.emissions.APU_emissions
   :members:
   :undoc-members:

.. automodule:: AEIC.emissions.EI_CO2
   :members:

.. automodule:: AEIC.emissions.EI_H2O
   :members:

.. automodule:: AEIC.emissions.EI_SOx
   :members:

.. automodule:: AEIC.emissions.EI_HCCO
   :members:

.. automodule:: AEIC.emissions.EI_NOx
   :members:

.. automodule:: AEIC.emissions.EI_PMnvol
   :members:

.. automodule:: AEIC.emissions.EI_PMvol
   :members:

.. automodule:: AEIC.emissions.lifecycle_CO2
   :members:
