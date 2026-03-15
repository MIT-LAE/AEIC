import base64
import subprocess
import sys
import textwrap

import cloudpickle


def run_in_subprocess(func, *args, **kwargs):
    """Helper function to run a given function with arguments in a separate
    subprocess.

    This is needed for some tests using NetCDF files, since the netCDF4 library
    has problems with large number of file open and close events in the same
    process. By running the test code in a separate subprocess, we can avoid
    these issues and ensure that file handles are properly closed after the
    test."""

    payload = base64.b64encode(cloudpickle.dumps((func, args, kwargs))).decode()

    code = textwrap.dedent(
        """
        import base64
        import cloudpickle

        func, args, kwargs = cloudpickle.loads(base64.b64decode("{payload}"))
        func(*args, **kwargs)
        """
    ).format(payload=payload)

    subprocess.run([sys.executable, "-c", code], check=True)
