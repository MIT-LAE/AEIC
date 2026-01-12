# System configuration

AEIC uses a lot of configuration data loaded from different files. The system
configuration system provides a clean way to manage these different
configuration items, using models based on
[Pydantic](https://docs.pydantic.dev/latest/) to validate input data from TOML
files or programmatic sources.

## File paths

AEIC has a set of built-in data files, but many use cases will require access
to additional external files (for aircraft performance data, weather data,
mission databases and so on). AEIC can be given access to directories
containing additional configuration or data files by adding those directories
to the `AEIC_PATH` environment variable. This is a colon-separated sequence of
absolute directory paths (just like the usual Linux `PATH` variable).

Anywhere that a file path is required in AEIC, AEIC will search through the
directories in the `AEIC_PATH` variable and in the default AEIC data
directory. This search process works slightly differently to the Linux `PATH`
variable: leading directories in the file path that you supply to AEIC are
included as part of the search. For example, if you set `AEIC_PATH` to
`/big/aeic/data` and attempt to load a performance model with a path
`performance/b738-new.toml`, then AEIC will load the file
`/big/aeic/data/performance/b738-new.toml` if it exists.

## Configuration initialization and access

The AEIC configuration system is accessed via the {py:mod}`AEIC.config`
package. In normal usage, you import both the {py:class}`Config
<AEIC.config.Config>` class and the {py:data}`config <AEIC.config.config>`
singleton instance from this module:

```python
from AEIC.config import Config, config
```

A selected AEIC configuration must then be loaded using the
{py:meth}`Config.load <AEIC.config.Config.load>` method. Calling
`Config.load()` without arguments loads the default AEIC configuration (from
the file `src/AEIC/data/default_config.toml` in the AEIC source tree).
Individual configuration settings can be overridden by providing either a TOML
file (that needs to contain only the options that are modified), or by passing
keyword arguments to the {py:meth}`Config.load <AEIC.config.Config.load>`
method. For example, here we set the `lifecycle_enabled` emissions option to
false (it defaults to true). All other values are taken from the default
configuration that's included in the AEIC package.

```python
from AEIC.config import Config, LTOInputMode

Config.load(emissions=dict(lifecycle_enabled=False))
```

The global AEIC system configuration can be accessed via the {py:data}`config
<AEIC.config.config>` proxy object in the {py.mod}`AEIC.config` package.
Attempting to access values in this object before initializing the system
configuration will result in an error:

```python
from AEIC.config import Config, config

print(config.path)
# Raises "ValueError: AEIC configuration is not set"

Config.load()

print(config.path)
# Prints the default AEIC data directory, assuming we have not set AEIC_PATH.
```

Normally, in AEIC code that is doing calculations that relies on configuration
values, you will do `from AEIC.config import config` at the top of your Python
file and then use values like `config.emissions.lifecycle_enabled`, and these
will come from whatever the currently loaded configuration is.

Sometimes, for debugging, you may want to get a real {py:class}`Config
<AEIC.config.Config>` object for the current configuration, instead of going
via the `config` proxy. You can do this using the {py:meth}`Config.get
<AEIC.config.Config.get>` method.

## Configuration uniqueness and immutability

Only one {py:class}`Config <AEIC.config.Config>` object may exist at any time.
This prevents accidentally running part of a simulation with one set of
configuration data and another part with a different configuration. There
should be no cases in code within AEIC where this poses an obstacle. (It does
make testing a little more difficult, but this is dealt with in our test setup
code.)

The configuration is frozen: it may not be modified it once it is initialized.
This is also intended to avoid problems with simulations running with
inconsistent configuration settings. (Again, this does mean that some special
tricks are needed for modifying configuration data during tests, but it's
worth the small amount of additional effort to have immutable configuration
data for non-test use cases.)

In normal use and development of AEIC code, you should not need to do this,
but it's possible to reset the configuration so that you can load a different
one using the {py:meth}`Config.reset <AEIC.config.Config.reset>` method. This
is important for testing and interactive use of the configuration system, but
you should not be doing it in "normal" AEIC code!

## Main configuration class

```{eval-rst}
.. autoclass:: AEIC.config.Config
    :members:
```

## Individual module configuration classes

```{eval-rst}
.. autoclass:: AEIC.emissions.config.EmissionsConfig
    :members:
    :no-index:
```

```{eval-rst}
.. autoclass:: AEIC.weather.config.WeatherConfig
    :members:
```
