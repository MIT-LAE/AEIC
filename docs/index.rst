Welcome to AEIC's documentation!
================================

The **Aviation Emissions Inventory Tool** (``AEIC``) is a Python library for
modelling and aggregating global aviation emissions over time. Based on a
previous version made using ``MATLAB`` by Simone et al. (found `here
<https://zenodo.org/records/6461767>`_), this updated library seeks to migrate
the MATLAB functionality to Python and add support for previously unsupported
performance models, updated trajectory dynamics, parallelization, and more.

.. note::
   This project is under active development

.. include:: main_page.md
   :parser: myst_parser.docutils_

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   main_page

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: AEIC modules

   src/configuration.md
   src/performance_models/performance_models.md
   src/trajectories/trajectories.md
   src/emission.md
   src/bada.md
   src/gridding.md
   src/missions.md
   src/mission_database.md
   src/oag.md
   src/parsers.md
   src/utilities.md
   src/weather.md

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer documentation

   src/developer/tools.md
   src/developer/conventions.md
