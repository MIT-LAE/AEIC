Performance Model
=================

``AEIC.performance_model.PerformanceModel`` takes aircraft performance,
missions, and emissions configuration data as input and produces the
data structure needed by trajectory solvers and the emissions pipeline.
It builds a fuel-flow performance table as a function of aircraft mass,
altitude, rate of climb/descent, and true airspeed.

Overview
----------

- Loads project-wide TOML configuration and as :class:`PerformanceConfig`.
- Supports two input modes via :class:`PerformanceInputMode`: reading BADA style OPF
  files or ingesting the custom performance-model TOML tables.
- Automatically loads mission definitions, LTO data (either from the performance
  file or EDB databank), APU characteristics, and BADA3-based engine parameters.
- Provides convenience accessors such as :attr:`PerformanceModel.missions`,
  :attr:`PerformanceModel.performance_table`, and :attr:`PerformanceModel.model_info`
  that later modules can consume without re-parsing TOML files.

Usage Example
-------------

.. code-block:: python

   from AEIC.performance_model import PerformanceModel

   perf = PerformanceModel("IO/default_config.toml")
   print("Loaded missions:", len(perf.missions))

   table = perf.performance_table
   fl_grid, tas_grid, roc_grid, mass_grid = perf.performance_table_cols
   print("Fuel-flow grid shape:", table.shape)

   # Pass to trajectory or emissions builders
   from emissions.emission import Emission
   emitter = Emission(perf)

Configuration Schema
--------------------

``PerformanceConfig`` converts the nested TOML mapping into a frozen dataclass
with well-defined fields. Key sections include:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Section.Key
     - Required
     - Description
   * - ``[Missions].missions_folder``
     - ✓
     - Directory containing the mission TOML files (relative to the repo root).
   * - ``[Missions].missions_in_file``
     - ✓
     - File within ``missions_folder`` that lists available missions under ``[flight]``.
   * - ``[General Information].performance_model_input``
     - ✓
     - Determines :class:`PerformanceInputMode`; accepts ``"opf"`` or ``"performancemodel"``.
   * - ``[General Information].performance_model_input_file``
     - ✓
     - Path to the OPF file or the performance-model TOML containing
       ``[flight_performance]`` and ``[LTO_performance]`` sections.
   * - ``[Emissions]``
     - ✓
     - Stored as :attr:`PerformanceConfig.emissions` and forwarded to
       :class:`emissions.emission.EmissionsConfig` for validation of LTO/fuel
       choices (see :ref:`Emissions Module <emissions-module>`).

Data Products
-------------

After :meth:`PerformanceModel.initialize_performance` runs, the instance
contains:

- :attr:`PerformanceModel.missions`: list of mission dictionaries read from the
  ``missions_in_file``.
- :attr:`PerformanceModel.ac_params`: populated :class:`BADA.aircraft_parameters.Bada3AircraftParameters`
  reflecting either OPF inputs or the performance table metadata.
- :attr:`PerformanceModel.engine_model`: a :class:`BADA.model.Bada3JetEngineModel`
  initialised with ``ac_params`` for thrust and fuel-flow calculations.
- :attr:`PerformanceModel.performance_table`: the multidimensional NumPy array
  mapping (flight level, TAS, ROCD, mass, …) onto fuel flow (kg/s).
- :attr:`PerformanceModel.performance_table_cols` and
  :attr:`PerformanceModel.performance_table_colnames`: the coordinate arrays and
  names that describe each dimension of ``performance_table``.
- :attr:`PerformanceModel.LTO_data`: modal thrust settings, fuel flows, and
  emission indices pulled from the performance file (when ``LTO_input_mode =
  "performance_model"``) or parsed via :func:`parsers.LTO_reader.parseLTO`.
- :attr:`PerformanceModel.EDB_data`: ICAO engine databank entry loaded when
  ``LTO_input_mode = "EDB"``.
- :attr:`PerformanceModel.APU_data`: auxiliary-power-unit properties resolved
  from ``engines/APU_data.toml`` using the ``APU_name`` specified in the
  performance file.
- :attr:`PerformanceModel.model_info`: the remaining metadata (e.g., cruise
  speeds, aerodynamic coefficients) trimmed away from ``flight_performance`` after
  the table is created.

API Reference
-------------

.. autoclass:: AEIC.performance_model.PerformanceInputMode
   :members:

.. autoclass:: AEIC.performance_model.PerformanceConfig
   :members:

.. autoclass:: AEIC.performance_model.PerformanceModel
   :members: __init__, initialize_performance, read_performance_data, create_performance_table
