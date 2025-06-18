Performance Model
=======================

.. automodule:: AEIC.performance_model.PerformanceModel
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``PerformanceModel`` class encapsulates aircraft performance data and related methods for trajectory, and emissions modeling. It builds a structured matrix of fuel flows as a function of altitude, TAS, ROCD, mass.

Class Definition
----------------

.. class:: PerformanceModel(config_file="IO/default_config.toml")

   A performance model for an aircraft. Initializes aircraft configuration, performance data, and engine model.

   :param config_file: Path to a TOML configuration file (default is ``IO/default_config.toml``).

Methods
-------

.. method:: __init__(config_file="IO/default_config.toml")

   Initializes the performance model by reading the configuration, loading mission data, and setting up performance and engine models.

.. method:: initialize_performance()

   Initializes aircraft performance characteristics from either OPF or TOML sources. Also loads LTO cycle data and sets up the engine model using BADA3 parameters.

.. method:: read_performance_data()

   Parses the TOML input file containing flight and LTO performance data. Separates model metadata and prepares the data for table generation.

.. method:: create_performance_table(data_dict)

   Dynamically constructs a multidimensional performance table from parsed TOML data. The table expresses fuel flow as a function of several input variables.

   :param data_dict: Dictionary containing keys ``cols`` (column labels) and ``data`` (performance entries).
   :type data_dict: dict

   :raises ValueError: If the ``FUEL_FLOW`` column is not found in the input data.

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
