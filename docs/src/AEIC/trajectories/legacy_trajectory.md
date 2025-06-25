# Legacy ``MATLAB`` Trajectory

The legacy trajectory relies on [BADA-3](<https://www.eurocontrol.int/model/bada>)-like
performane data in the AEIC performance data format. Specifically, it requires data that
has prescribed climb and descent profiles, as well as cruise data at a single altitude
($7000\,\text{ft}$ below operating ceiling).


## Legacy Trajectory Class

```{eval-rst}
.. autoclass:: AEIC.trajectories.legacy_trajectory.LegacyTrajectory
   :members:
```