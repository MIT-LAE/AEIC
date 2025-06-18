Emissions
=========

The ``Emission`` class encapsulates the full calculation of aircraft emissions for a mission:

- **Trajectory emissions** (CO₂, H₂O, SO₂, NOₓ, HC, CO, particulate species) for every time step
- **LTO cycle emissions** (taxi, approach, climb, takeoff)
- **APU** (auxiliary power unit) emissions
- **Ground Service Equipment** (GSE) emissions
- **Life-cycle CO₂** additive for fuel production

All results are stored internally in structured NumPy arrays and can be summed.

----

Usage Example
-------------

.. code-block:: python

   from AEIC.performance_model import PerformanceModel
   from AEIC.trajectories.trajectory import Trajectory
   from emissions import Emission

   # Initialize performance model & trajectory (user code)
   perf = PerformanceModel.from_edb('path/to/config_file.toml')
   # Load mission and set trajectory
   traj = Trajectory(perf, mission, optimize_traj, iterate_mass)
   traj.fly_flight()

   # Compute emissions
   em = Emission(
       ac_performance=perf,
       trajectory=traj,
       EDB_data=True,
       fuel_file='fuels/conventional_jetA.toml'
   )

   # Access summed emissions (g)
   total = em.summed_emission_g
   print("Total CO₂ (g):", total['CO2'])
   print("Total NOx (g):", total['NOx'])

----

Constructor
-----------

.. code-block:: python

   Emission(
       ac_performance: PerformanceModel,
       trajectory: Trajectory,
       EDB_data: bool,
       fuel_file: str
   )

**Parameters**

+-------------------+----------------------+----------------------------------------------------------------------------------------+
| Name              | Type                 | Description                                                                            |
+===================+======================+========================================================================================+
| ``ac_performance``| ``PerformanceModel`` | Aircraft performance object providing climb/cruise/descent and LTO data matrices.      |
+-------------------+----------------------+----------------------------------------------------------------------------------------+
| ``trajectory``    | ``Trajectory``       | Flight trajectory containing altitude, speed, fuel‐mass, and fuel‐flow time series.    |
+-------------------+----------------------+----------------------------------------------------------------------------------------+
| ``EDB_data``      | ``bool``             | If ``True``, uses tabulated EDB emissions; otherwise uses user‐specified LTO settings. |
+-------------------+----------------------+----------------------------------------------------------------------------------------+
| ``fuel_file``     | ``str``              | Path to TOML file of fuel properties (e.g. CO₂ factors, sulfur content, lifecycle CO₂).|
+-------------------+----------------------+----------------------------------------------------------------------------------------+

Upon instantiation, the following steps occur:

1. Fuel TOML is loaded.
2. Array shapes are initialized based on trajectory lengths and config flags.
3. Fuel burn per segment is derived from the trajectory’s fuel-mass time-series.
4. **Trajectory**, **LTO**, **APU**, and **GSE** emissions are computed.
5. All sources are summed and life-cycle CO₂ is added.

----

Attributes
----------

+----------------------------------+-----------+--------------------------------------------------------------------+
| Name                             | Type      | Description                                                        |
+==================================+===========+====================================================================+
| ``fuel``                         | ``dict``  | Fuel properties loaded from TOML (e.g. ``EI_CO2``, ``LC_CO2``).    |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``Ntot``, ``NClm``, ``NCrz``, ``NDes`` | ``int`` | Total, climb, cruise, descent time-step counts.                    |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``traj_emissions_all``          | ``bool``  | Whether climb/descent use performance model or LTO‐data.           |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``pmnvol_mode``                 | ``str``   | PM number estimation method for LTO (e.g. ``"SCOPE11"``).          |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``fuel_burn_per_segment``        | ``ndarray``| Fuel burned (kg) each time step.                                   |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``emission_indices``             | ``ndarray``| Emission indices (g per kg fuel) for every species & time step.    |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``pointwise_emissions_g``        | ``ndarray``| Emissions (g) per time step for trajectory phase.                  |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``LTO_emission_indices``         | ``ndarray``| Emission indices for each LTO mode.                                |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``LTO_emissions_g``              | ``ndarray``| Emissions (g) for each LTO mode.                                   |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``APU_emission_indices``         | ``ndarray``| APU emission indices (g per kg fuel).                              |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``APU_emissions_g``              | ``ndarray``| APU emissions (g).                                                 |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``GSE_emissions_g``              | ``ndarray``| GSE emissions (g) per engine‐start cycle.                          |
+----------------------------------+-----------+--------------------------------------------------------------------+
| ``summed_emission_g``            | ``ndarray``| Total emissions (g) aggregated over all sources per species.       |
+----------------------------------+-----------+--------------------------------------------------------------------+

----

Public Methods
--------------

``get_trajectory_emissions(trajectory, ac_performance, EDB_data=True)``
   Calculate emission indices and per-segment emissions for the entire flight profile.

   :param trajectory: ``Trajectory`` – flight trajectory
   :param ac_performance: ``PerformanceModel`` – performance data
   :param EDB_data: ``bool`` – whether to use tabulated EDB emissions (default: ``True``)

``get_LTO_emissions(ac_performance, EDB_LTO=True, pmnvol_switch_lc="SCOPE11")``
   Compute Landing-and-Take-Off cycle emission indices & quantities.

   :param ac_performance: ``PerformanceModel``
   :param EDB_LTO: ``bool`` – use EDB LTO settings (default: ``True``)
   :param pmnvol_switch_lc: ``str`` – black carbon number emissions model (default: ``"SCOPE11"``)

``get_GSE_emissions(wnsf: str)``
   Calculate Ground Service Equipment emissions based on aircraft type.

   :param wnsf: ``str`` – one of ``"w"``, ``"n"``, ``"s"``, or ``"f"``

``sum_total_emissions()``
   Aggregate pointwise trajectory, LTO, APU, and GSE emissions into ``summed_emission_g``.

``get_lifecycle_emissions(fuel: dict, traj: Trajectory)``
   Add life-cycle CO₂ (g) for fuel production to ``summed_emission_g['CO2']``.

----

Emission dtype Fields
---------------------

The private helper ``__emission_dtype(shape)`` defines a structured NumPy dtype with the following fields (all ``float64``):

- **CO2**: Carbon dioxide
- **H2O**: Water vapor
- **HC**: Hydrocarbons
- **CO**: Carbon monoxide
- **NOx**: Total nitrogen oxides
- **NO**: Nitric oxide
- **NO2**: Nitrogen dioxide
- **HONO**: Nitrous acid
- **PMnvol**: Black carbon
- **PMnvol_lo**: Lower bound black carbon
- **PMnvol_hi**: Upper bound black carbon
- **PMnvolN**: Black carbon number
- **PMnvolN_lo**: Lower bound number
- **PMnvolN_hi**: Upper bound number
- **PMnvolGMD**: Geometric mean diameter of black carbon (nm)
- **PMvol**: Organic particulate matter mass
- **OCic**: Organic carbon (incomplete combustion)
- **SO2**: Sulfur dioxide
- **SO4**: Sulfate

.. note::

   If ``pmnvol_mode`` is disabled, the ``*_lo``, ``*_hi``, and ``PMnvolN`` fields are omitted.

----

Notes
-----

- **Structured arrays** are used heavily—one field per pollutant, shaped by segment or mode count.
- Private helper ``__emission_dtype(shape)`` defines the NumPy dtype fields.
- Fuel burn is computed as the decrease in ``traj.traj_data['fuelMass']``.
