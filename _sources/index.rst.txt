.. AEIC documentation master file, created by
   sphinx-quickstart on Wed Feb  5 10:33:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AEIC's documentation!
================================

The **Aviation Emissions Inventory Tool** (``AEIC``) is a Python library for modelling and aggregating global aviation emissions over time. 
Based on a previous version made using ``MATLAB`` by Simone et al. (found `here <https://zenodo.org/records/6461767>`_), this updated
library seeks to migrate the MATLAB functionality to Python and add support for previously unsupported performance models, 
updated trajectory dynamics, parallelization, and more.

.. note::
   This project is under active development

.. include:: main_page.rst
   :start-after: .. begin-getting-started
   :end-before: .. end-getting-started

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   main_page

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: AEIC Modules

   src/AEIC/performance_model
   src/AEIC/trajectories/trajectories.md
   src/BADA/bada
   src/gridding/gridding
   data/IO/io_formats
   src/missions/mission
   src/parsers/parsers
   src/utils/utils
   src/weather/weather
