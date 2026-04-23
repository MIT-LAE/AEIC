[![codecov](https://codecov.io/gh/MIT-LAE/AEIC/graph/badge.svg?token=oOnGuSTTVm)](https://codecov.io/gh/MIT-LAE/AEIC)

# AEIC

Aviation Emissions Inventory Code (AEIC) is used to estimate aviation emissions using an aircraft performance model and a set of missions. It produces inventories that can be sliced by aircraft type, time period, or operating scenario to support analysis and reporting. Core modules cover mission definitions, trajectories, performance models, emissions estimation, gridding, and weather.

## Installation

`AEIC` is not currently available on PyPI, but you can install releases from
GitHub. The latest available version is v0.3.0. If you are using `pip`, do

```shell
pip install git+https://github.com/MIT-LAE/AEIC.git@v0.3.0
```

If you are using `uv`, do

```shell
uv add git+https://github.com/MIT-LAE/AEIC.git@v0.3.0
```

## Local Development

If you intend to develop `AEIC` itself, see the
[developer setup guide](docs/src/developer/tools.md) for the recommended
`uv`-based workflow, test runner configuration, and pre-commit hooks.

## Units and Non-Dimensionals

`AEIC` works in SI units. The only exception is the pressure altitude, which has both SI and imperial flight level representations. All non-dimensional quantities are treated internally as SI.
