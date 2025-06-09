#import cdsapi
import numpy as np
import xarray as xr

ds_ml = xr.open_dataset("/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/model/20241223.nc")
ds_lnsp = xr.open_dataarray("/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/model/lnsp/20241223.nc")

from pycontrails.datalib.ecmwf.model_levels import ml_to_pl

low_pl = 80
high_pl = 470
spacing = 10

pl = np.arange(low_pl, high_pl, spacing)

levels = len(pl)



ds_ml2ps = ml_to_pl(ds_ml, target_pl=pl, lnsp=ds_lnsp)


output_path = f"/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/processed_model/ERA5_{levels}_pressure_levels_2024_12_23.nc"

ds_ml2ps.to_netcdf(output_path)

print(f"✅ Saved pressure-level dataset to {output_path}")