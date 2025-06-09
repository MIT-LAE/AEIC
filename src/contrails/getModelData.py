from typing import Any, Optional
from datetime import datetime, timedelta
import cdsapi
import os
from concurrent.futures import ThreadPoolExecutor

ERA5_PRESSURE_LEVELS = [
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "125",
    "150",
    "175",
    "200",
    "225",
    "250",
    "300",
    "350",
    "400",
    "450",
    "500",
    "550",
    "600",
    "650",
    "700",
    "750",
    "775",
    "800",
    "825",
    "850",
    "875",
    "900",
    "925",
    "950",
    "975",
    "1000",
]

# Defined here https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
# [63, 95] is approximately 100 hPa to 500 hPa but take extra for padding during interpolation
ERA5_MODEL_LEVELS_SUB = list(range(60, 99))

ERA5_VARIABLES_COCIP = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]

GLOBAL_EXTENT = [-180, 180, -90, 90]

PRODUCT_TYPES = ["reanalysis"]

ECMWF_IDS = {
    "temperature": 130,
    "u_component_of_wind": 131,
    "v_component_of_wind": 132,
    "specific_humidity": 133,
    "lnsp": 152,
}

ECMWF_TYPE = [
    "reanalysis-era5-pressure-levels",
    "reanalysis-era5-single-levels",
    "reanalysis-era5-complete",
]


def make_CDSAPI_request(
    time: datetime,
    variables: list[str],
    pressure_levels: list[int | str] = ERA5_PRESSURE_LEVELS,
    extent: list[float] = GLOBAL_EXTENT,
    ecmwf_type: str = "reanalysis-era5-complete",
) -> dict[str, Any]:
    # Remove hour and minute from datetime
    time = time.replace(hour=0)
    time = time.replace(minute=0)

    # Time list
    times = [(time + timedelta(hours=i)).strftime("%H:%M") for i in range(24)]

    # Re-order extent to comply with CDSAPI convention
    extent = [extent[3], extent[0], extent[2], extent[1]]

    # Convert pressure levels to strings
    pressure_levels = [str(p) for p in pressure_levels]

    CDSAPI_settings = {
        "variable": variables,
        "product_type": "reanalysis",
        "year": f"{time.year}",
        "month": f"{str(time.month).rjust(2, '0')}",
        "day": f"{str(time.day).rjust(2, '0')}",
        "time": times,
        "area": extent,
        "format": "netcdf",
    }

    if ecmwf_type == "reanalysis-era5-pressure-levels":
        CDSAPI_settings["pressure_level"] = pressure_levels

    return CDSAPI_settings

def make_MARSAPI_request(
    time: datetime,
    variables: list[str],
    levels: list[int | str] = ERA5_MODEL_LEVELS_SUB,
    extent: list[float] = GLOBAL_EXTENT,
    grid: str = "0.25/0.25",
    lnsp: bool = False,
) -> dict[str, str]:
    # Full information at https://apps.ecmwf.int/codes/grib/param-db/
    # To write efficient MARS requests
    # https://confluence.ecmwf.int/display/UDOC/HRES%3A+Atmospheric+%28oper%29%2C+Model+level+%28ml%29%2C+Forecast+%28fc%29%3A+Guidelines+to+write+efficient+MARS+requests

    extent_reordered = [str(extent[3]), str(extent[0]), str(extent[2]), str(extent[1])]
    extent_str = "/".join(extent_reordered)

    levels_list = [str(level) for level in levels]
    levels_str = "/".join(levels_list)
    # add the first level to get lnsp
    # levels_str = "1/" + "/".join(levels_list)

    if "relative_humidity" in variables:
        print(
            "relative_humidity not available in model level data, skipping this variable"
        )
    
     # Logarithm of surface pressure is required for
    # conversion from model levels to pressure levels
    # Always add it to the request
    # if "lnsp" not in variables:
    #     variables.append("lnsp")

    vars_str = "/".join(
        [str(ECMWF_IDS[var]) for var in variables if var != "relative_humidity"]
    )
    
    MARSAPI_request = {
        "levtype": "ml",  # model level
        "stream": "oper",
        "time": "00/to/23/by/1",  # get 24h
        "type": "an",
        "format": "netcdf",
    }
        
    MARSAPI_request["date"] = time.strftime("%Y-%m-%d")
    MARSAPI_request["area"] = extent_str
    MARSAPI_request["grid"] = grid
    
    if lnsp:
        MARSAPI_request["levelist"] = "1"
        MARSAPI_request["param"] = "152"
    else:
        MARSAPI_request["levelist"] = levels_str
        MARSAPI_request["param"] = vars_str
        
    return MARSAPI_request

def download_ERA5_data(
    path: str,
    time: datetime,
    variables: list[str],
    levels: list[str | int],
    extent: list[float],
    ecwmf_type: str,
    grid: Optional[str] = None,
    ):
    """
    Downloads ERA5 pressure level data using the CDSAPI

    Parameters
    ----------
    path : str
        Location to store downloaded data
    time : datetime
        Day for which to get ERA5 data
    variables : List[string] (optional)
        Variables to download
    levels : list[int | str]
        Levels to download (hPa or model levels)
    extent : list[float]
        Extent [min_lon, max_lon, min_lat, max_lat]
    """

    assert ecwmf_type in ECMWF_TYPE
    
    c = cdsapi.Client()

    if ecwmf_type != "reanalysis-era5-complete":
        if grid is not None:
            print("grid arg is ignored unless downloading ERA5 on model levels")

        request = make_CDSAPI_request(
            time=time,
            variables=variables,
            pressure_levels=levels,
            extent=extent,
        )
    else:
        if grid is None:
            grid = "0.25/0.25"
            print(f"Grid not specified, defaulting to {grid = }")

        request = make_MARSAPI_request(
            time=time,
            variables=variables,
            levels=levels,
            extent=extent,
            grid=grid,
        )
        

        request_lnsp = make_MARSAPI_request(
            time, variables=["lnsp"], levels=[1], extent=extent, grid=grid, lnsp=True
        )

        path_lnsp = path.split("/")[-1]
        path_lnsp = "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/model/lnsp/" + path_lnsp
        c.retrieve(
            ecwmf_type,
            request_lnsp,
            path_lnsp,
        )

    c.retrieve(ecwmf_type, request, path)
    print(f"\nSaved to {path}\n")

def downloader(
    path_dir: str, times: list[datetime], nthreads: int = 1, **download_kwargs
) -> None:
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"{path_dir} is not a directory")

    executor = ThreadPoolExecutor(max_workers=nthreads)

    ##### TODO group request by months for MARS access
    # have each thread do a separate month (no concurrent reads on a single tape)
    # maybe group every day within a month to a single request
    # and separate the dataset later

    for time in times:
        path = path_dir + time.strftime("%Y%m%d.nc")
        if os.path.exists(path):
            print(f"{path} exists, skipping")
            continue
        
        #executor.submit(download_ERA5_data, path, time, **download_kwargs)
        
        download_ERA5_data(path, time, **download_kwargs)
        

def get_dir(product: str) -> str:
    match product:
        case "reanalysis-era5-pressure-levels":
            return "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/multi_level/"
        case "reanalysis-era5-single-levels":
            return "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/single_level/"
        case "reanalysis-era5-complete":
            return "/home/prateekr/Workbench/AEIC_DEV/AEIC/src/contrails/ERA5/model/"
        case _:
            raise ValueError(f"{product} is not in {ECMWF_TYPE}")
        

if __name__ == "__main__":
    start_time = datetime(2024, 12, 23)
    times = [start_time + timedelta(days=i) for i in range(2)]
    
    ncpus = 5
    
    product = ECMWF_TYPE[2]
    path_dir_save = get_dir(product)
    
    downloader(
        path_dir_save,
        times,
        ncpus,
        variables=ERA5_VARIABLES_COCIP,
        levels=ERA5_PRESSURE_LEVELS,
        extent=GLOBAL_EXTENT,
        #extent=[-150, -50, 10, 60],
        ecwmf_type=product,
    )    
    
    
    
    