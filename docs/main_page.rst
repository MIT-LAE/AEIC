.. begin-getting-started

Getting Started
===============
Currently, the workflow in ``AEIC`` is limited to a modeling process based on the ``MATLAB`` implementation, but more functionality will be
added over time. We outline the basics of setup here.

Installation
------------
``AEIC`` is not currently available on PyPI. As such, the current best method for simple usage is identical to the local development
setup detailed below.

Local Development
-----------------
If you intend to develop the source code of ``AEIC``, you should create a fork, clone the fork locally, and install ``AEIC`` in development
mode. For example:

1. Fork the main git repository

2. Clone the forked repository locally

   .. code-block:: console

      git clone git@github.com:{YourName}/AEIC.git

3. In your Python environment, install in development mode

   .. code-block:: console

      cd AEIC
      pip install --editable .

You should now be able to import ``AEIC`` as you would with any standard library.

Units and Non-Dimensionals
--------------------------
``AEIC`` works in SI units. The only exception is the pressure altitude, which has both SI and imperial flight level representations. All
non-dimensional quantities are treated internally as SI.

.. end-getting-started
