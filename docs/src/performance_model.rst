Performance Model
=================

``AEIC.performance_model.PerformanceModel`` takes aircraft performance,
missions, and emissions configuration data as input and produces the
data structure needed by trajectory solvers and the emissions pipeline.
It builds a fuel-flow performance table as a function of aircraft mass,
altitude, rate of climb/descent, and true airspeed.

Overview
----------

- Supports two input modes via :class:`AEIC.config.PerformanceInputMode`:
  reading BADA style OPF files or ingesting the custom performance-model TOML
  tables.
- Automatically loads mission definitions, LTO data (either from the performance
  file or EDB databank), APU characteristics, and BADA3-based engine parameters.
- Provides convenience accessors such as
  :attr:`PerformanceModel.performance_table`, and
  :attr:`PerformanceModel.model_info` that later modules can consume without
  re-parsing TOML files.

Usage Example
-------------

.. code-block:: python

   from AEIC.config import Config
   from AEIC.performance_model import PerformanceModel

   # Load default AEIC configuration.
   Config.load()

   perf = PerformanceModel("IO/sample_performance_model.toml")
   table = perf.performance_table
   fl_grid, tas_grid, roc_grid, mass_grid = perf.performance_table_cols
   print("Fuel-flow grid shape:", table.shape)

   # Pass to trajectory or emissions builders
   from AEIC.emissions.emission import Emission
   emitter = Emission(perf)

Data Products
-------------

After a :class:`PerformanceModel` instance is created, the instance contains:

- :attr:`PerformanceModel.ac_params`: populated :class:`AEIC.BADA.aircraft_parameters.Bada3AircraftParameters`
  reflecting either OPF inputs or the performance table metadata.
- :attr:`PerformanceModel.engine_model`: a :class:`AEIC.BADA.model.Bada3JetEngineModel`
  initialised with ``ac_params`` for thrust and fuel-flow calculations.
- :attr:`PerformanceModel.performance_table`: the multidimensional NumPy array
  mapping (flight level, TAS, ROCD, mass, …) onto fuel flow (kg/s).
- :attr:`PerformanceModel.performance_table_cols` and
  :attr:`PerformanceModel.performance_table_colnames`: the coordinate arrays and
  names that describe each dimension of ``performance_table``.
- :attr:`PerformanceModel.LTO_data`: modal thrust settings, fuel flows, and
  emission indices pulled from the performance file (when ``LTO_input_mode =
  "performance_model"``) or parsed via :func:`AEIC.parsers.LTO_reader.parseLTO`.
- :attr:`PerformanceModel.EDB_data`: ICAO engine databank entry loaded by
  :meth:`PerformanceModel.get_engine_by_uid` when ``LTO_input_mode = "edb"``.
- :attr:`PerformanceModel.APU_data`: auxiliary-power-unit properties resolved
  from ``engines/APU_data.toml`` using the ``APU_name`` specified in the
  performance file.
- :attr:`PerformanceModel.model_info`: the remaining metadata (e.g., cruise
  speeds, aerodynamic coefficients) trimmed away from ``flight_performance`` after
  the table is created.

API Reference
-------------

.. autoenum:: AEIC.config.PerformanceInputMode
   :members:

.. autoclass:: AEIC.performance_model.PerformanceModel
   :members: __init__, read_performance_data, create_performance_table
