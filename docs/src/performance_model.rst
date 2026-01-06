Performance Model
=================

The classes in the :mod:`AEIC.performance` module take aircraft performance,
missions, and emissions configuration data as input and produce data
structures needed by trajectory solvers and the emissions pipeline. In
particular, the :class:`TablePerformanceModel` class builds a fuel-flow
performance table as a function of aircraft mass, altitude, rate of
climb/descent, and true airspeed.

.. warning::

   This code is currently in development. The performance model API will
   change significantly relatively soon.

   This documentation describes performance models as currently implemented,
   which include only a table-based model with all data read from a single
   TOML file, based on the legacy Matlab code. This model claculates fuel flow
   from aircraft state in a simple way, and can also return feasible ROC and
   TAS values given aircraft state.

   Future iterations of the performance model code will incorporate
   mechanistic models based on BADA as well as more complex table-based models
   that will take a more explicitly state-based approach based on "trajectory
   rules".


Usage Example
-------------

.. code-block:: python

   from AEIC.config import Config
   from AEIC.performance.models import PerformanceModel

   # Load default AEIC configuration.
   Config.load()

   perf = PerformanceModel.load("performance/sample_performance_model.toml")
   table = perf.flight_performance.performance_table
   fl_grid, tas_grid, roc_grid, mass_grid = perf.performance_table_cols
   print("Fuel-flow grid shape:", table.shape)

   # Pass to trajectory or emissions builders
   from AEIC.emissions.emission import Emission
   emitter = Emission(perf)


Class Hierarchy
---------------

Performance model classes are all Pydantic models derived from
:class:`AEIC.performance.base.BasePerformanceModel`. This is an abstract base
class that includes data common to all performance model types (aircraft name
and class, maximum altitude and payload, number of engines, APU information
and optional LTO and speed information) and that defines the performance model
API. The legacy table-based performance model is represented by the
:class:`AEIC.performance.table.TablePerformanceModel` class. This includes a
performance table represented by the
:class:`AEIC.performance.table.PerformanceTable` class which performs
subsetting and interpolation within the input data.

.. note::

   At the moment, the trajectory builder code that uses the performance table
   reaches directly into the performance model to access the performance
   table, but I'm going to change that soon to modify the performance table
   class to expose a sensible subsetting and interpolation API.


Loading Performance Models
--------------------------

Performance models can be loaded from TOML files. A ``model_type`` field is
used to distinguish between different types of performance model and a
:class:`AEIC.performance.PerformanceModel` wrapper class is used to enable
this: there is a :meth:`PerformanceModel.load` class method with a polymorphic
return type, that takes a path to a TOML file containing a performance model
definition and returns an instance of the correct performance model class
based on the ``model_type`` field. For the current legacy-based performance
models, use ``model_type = "table"``.


Performance Model Members
-------------------------

After a :class:`PerformanceModel` instance is created (of any derived type),
the instance contains:

- Basic information about the performance model: aircraft name and class,
  number of engines, maximum altitude and payload.
- :attr:`lto_performance`: modal thrust settings, fuel flows, and emission
  indices pulled from the performance file.
- :attr:`apu`: auxiliary-power-unit properties resolved from
  ``engines/APU_data.toml`` using the ``apu_name`` specified in the
  performance file.
- :attr:`speeds`: cruise speed data.


Table-based Performance Model Members
-------------------------------------

As well as the above common members, a :class:`TablePerformanceModel` also
contains:

- :attr:`performance_table`: the multidimensional NumPy array
  mapping (flight level, TAS, ROCD, mass, …) onto fuel flow (kg/s).
- :attr:`performance_table_cols` and
  :attr:`performance_table_colnames`: the coordinate arrays and
  names that describe each dimension of ``performance_table``.

.. note::

   These properties of :class:`TablePerformanceModel` forward to a separate
   :class:`PerformanceTable` class. For the moment, this class is just a thin
   wrapper around a Numpy array, but it will be modified to offer a coherent
   subsetting and interpolation API.


API Reference
-------------

.. autoclass:: AEIC.performance.BasePerformanceModel
   :members:
   :exclude-members: apu_name, load_apu_data, model_config

.. autoclass:: AEIC.performance.TablePerformanceModel
   :members:
   :exclude-members: model_config, model_type, validate_pm

.. autoclass:: AEIC.performance.PerformanceModel
   :members:
   :exclude-members: model_config
