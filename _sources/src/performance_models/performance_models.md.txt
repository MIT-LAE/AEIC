# Performance models

The classes in the {py:mod}`AEIC.performance` module take aircraft
performance, missions, and emissions configuration data as input and produce
data structures needed by trajectory solvers and the emissions pipeline. In
particular, the {py:class}`LegacyPerformanceModel <AEIC.performance.models.LegacyPerformanceModel>`
class builds a fuel-flow, rate of climb/descent, and true airspeed performance
table as a function of aircraft mass and altitude.

```{note}
Some of the details of how this works will probably change as we implement
new kinds of performance models. So far, we only have the table-based
legacy performance model intended to replicate the behavior of the old
Matlab code, but the organization of the performance table classes is
intended to be extensible to more complex use cases.
```

```{toctree}
:maxdepth: 1

performance_model_api.md
legacy_performance_model.md
performance_model_files.md
```
