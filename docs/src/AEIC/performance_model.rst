Performance Model
=======================

The ``PerformanceModel`` class encapsulates aircraft performance data. It builds a structured matrix of fuel flows as a function of altitude, TAS, ROCD, mass.

Class Definition
----------------

.. autoclass:: AEIC.performance_model.PerformanceModel
   :members:

Attributes
----------

.. attribute:: config

   Dictionary of all parsed key-value pairs from the configuration TOML file.

.. attribute:: missions

   List of missions (flights) parsed from the input TOML file.

.. attribute:: ac_params

   Instance of ``Bada3AircraftParameters`` representing aircraft configuration parameters.

.. attribute:: engine_model

   Instance of ``Bada3JetEngineModel`` initialized with aircraft parameters.

.. attribute:: LTO_data

   Dictionary of emissions or fuel flow data during the landing-takeoff cycle.

.. attribute:: model_info

   Parsed metadata from TOML describing the aircraft performance model (e.g., cruise speeds).

.. attribute:: performance_table

   Multidimensional NumPy array of fuel flow rates indexed by input variables.

.. attribute:: performance_table_cols

   List of sorted values for each input dimension of the performance table.

.. attribute:: performance_table_colnames

   Names of each input dimension (excluding fuel flow) used to construct the table.
