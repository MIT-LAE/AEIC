# Setup and tools

## Project setup and dependency management

 * All tool configuration and dependency information for AEIC code is held in
   the top-level `pyproject.toml` file.
 * AEIC uses Python 3.13.
 * Dependencies are managed using the [`uv` tool](https://docs.astral.sh/uv).


## Local Development

If you intend to develop the source code of AEIC, you should create a fork,
clone the fork locally, and install AEIC in development mode using the
[`uv` tool](https://docs.astral.sh/uv) (see [below](#uv)). For example:

1. Fork the main Git repository.

2. Clone the forked repository locally:

   ```shell
   git clone git@github.com:{YourName}/AEIC.git
   ```

3. In your Python environment, install in development mode:

   ```shell
   cd AEIC
   uv sync
   ```

You should now be able to import AEIC as you would with any standard library
within a Python session managed by `uv`.

## uv

The [`uv` tool](https://docs.astral.sh/uv) tool manages dependencies within a
virtual environment. Installation instructions are
[here](https://docs.astral.sh/uv/getting-started/installation/).

The most basic functionality needed is to run `uv sync` in the top-level
directory. This creates a virtual environment, installs all of the
dependencies, pins the Python version (based on the mandatory
`requires-python` field in the `pyproject.toml`), and automatically does an
editable install of the local package (just like you had done `pip install -e
.`), which is almost always what you want for local development.

(pre-commit)=
## pre-commit

The `pre-commit` tool helps to manage Git hooks, in particular, it lets you
set up scripts for checking whether changes you're tracking in Git are OK,
before you commit them. This lets you catch things like large file commits
before they get into the repository history. It also means fewer "Ruff fixes"
commits, since you can include running `ruff` as a pre-commit hook, so that
you never end up committing code that `ruff` doesn't like.

1. Run `pip install --user pre-commit` or `uv run pip install --user
   pre-commit` (this installs the `pre-commit` executable in `~/.local/bin`,
   so make sure you have that on your path).
2. Run `pre-commit install` at the top of your working copy of the repository.
   This sets up the necessary Git hooks to run the `pre-commit` tool.
3. The `pre-commit` hooks will run and will prevent you from committing until
   you make your changes "nice".

## Testing

Automated tests are in the `tests` directory and use the
[Pytest](https://docs.pytest.org/en/stable/) test runner. GitHub Actions are
set up to run all tests when pull requests are created.
