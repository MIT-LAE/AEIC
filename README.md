[![codecov](https://codecov.io/gh/MIT-LAE/AEIC/graph/badge.svg?token=oOnGuSTTVm)](https://codecov.io/gh/MIT-LAE/AEIC)

# AEIC

Aviation Emissions Inventory Code (AEIC) is used to estimate aviation emissions using an aircraft performance model and a set of missions. It produces inventories that can be sliced by aircraft type, time period, or operating scenario to support analysis and reporting. Core modules cover mission definitions, trajectories, performance models, emissions estimation, gridding, and weather.

## Installation

`AEIC` is not currently available on PyPI. As such, the current best method for simple usage is identical to the local development setup detailed below.

## Local Development

If you intend to develop the source code of `AEIC`, you should create a fork, clone the fork locally, and install `AEIC` in development mode. For example:

1. Fork the main git repository

2. Clone the forked repository locally

   ```console
   git clone git@github.com:{YourName}/AEIC.git
   ```

3. In your Python environment, install in development mode

   ```console
   cd AEIC
   pip install --editable .
   ```

You should now be able to import `AEIC` as you would with any standard library.

### uv

If you prefer, you can also use `uv` to maintain dependencies and a virtual environment while developing `AEIC`.

First [install](https://docs.astral.sh/uv/getting-started/installation/) `uv`. Run `uv sync` in the top-level directory, a virtual environment will be created, all of the dependencies will be installed and the Python version will be pinned at something that corresponds to what is in the `requires-python` field in the `pyproject.toml` (this is a mandatory field if you're using `uv`). In addition to installing dependencies, `uv` also automatically does an editable install of the local package (just like you had done `pip install -e .`), which is almost always what you want for local development.

### pre-commit

The `pre-commit` tool helps to manage Git hooks, in particular, it lets you set up scripts for checking whether changes you're tracking in Git are OK, before you commit them. This lets you catch things like large file commits before they get into the repository history. It also means fewer "Ruff fixes" commits, since you can include running `Ruff` as a pre-commit hook, so that you never end up committing code that `Ruff` doesn't like.

#### Usage instructions

1. Run `pip install --user pre-commit` or `uv run pip install --user pre-commit` (this installs the `pre-commit` executable in `~/.local/bin`, so make sure you have that on your path).
2. Run `pre-commit install` at the top of your working copy of the repository. This sets up the necessary Git hooks to run the `pre-commit` tool.
3. The `pre-commit` hooks will run and will prevent you from committing until you make your changes "nice".

## Units and Non-Dimensionals

`AEIC` works in SI units. The only exception is the pressure altitude, which has both SI and imperial flight level representations. All non-dimensional quantities are treated internally as SI.
