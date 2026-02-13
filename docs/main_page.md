# Getting Started

Aviation Emissions Inventory Code (AEIC) is used to estimate aviation
emissions using an aircraft performance model and a set of missions. It
produces inventories that can be sliced by aircraft type, time period, or
operating scenario to support analysis and reporting. Core modules cover
mission definitions, trajectories, performance models, emissions estimation,
gridding, and weather.

## Installation

`AEIC` is not currently available on PyPI, but you can install releases from
GitHub. The latest available version is v0.2.0. If you are using `pip`, do

```shell
pip install git+https://github.com/MIT-LAE/AEIC.git@v0.2.0
```

If you are using `uv`, do

```shell
uv add git+https://github.com/MIT-LAE/AEIC.git@v0.2.0
```

## Units

AEIC uses SI units interally. Conventional non-SI units are used for input and
output as appropriate.
