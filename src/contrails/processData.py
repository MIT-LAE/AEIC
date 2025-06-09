import numpy as np
import xarray as xr

from pycontrails.datalib.ecmwf.model_levels import ml_to_pl

# Path to model data
print("Loading model data...", end = ' ')
ds_ml = xr.open_dataset("/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/model/20241223.nc")
print("done")


# Path to surface log-pressure data

ds_lnsp = xr.open_dataarray("/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/model/lnsp/20241223.nc")
print("Loading log data...", end = ' ')
# Define pressure level data from FL200 -> FL450 in increments of 10 hectopascals
pl = np.arange(150, 410, 10)
print("done")

# Convert model data to pressure-level data
print("Converting model data to pressure level data...", end = '')
ds_ml2ps = ml_to_pl(ds_ml, target_pl=pl, lnsp=ds_lnsp)
print("done")


# Save as netCDF
print("Saving to disk", end = '')
ds_ml2ps.to_netcdf("/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/processed_model/ERA5_20241223_pl.nc")
print("done")