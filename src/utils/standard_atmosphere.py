import numpy as np
from typing import Union
from .custom_types import FloatOrNDArray
from numpy.typing import NDArray, ArrayLike
from .consts import *#h_p_tropo, beta_tropo, p0, T0, rho0, a0, R_air, g0, kappa


def temperature_at_altitude_isa_bada4(altitude: FloatOrNDArray) -> NDArray:
    """Return the temperature at the provided altitude(s).
    Units are SI (m, Kelvin)

    Parameters
    ----------
    altitude : Union[float,NDArray]
        Altitude in meters.

    Returns
    -------
    NDArray
        Temperature in Kelvin.

    Raises
    ------
    ValueError
        If altitude is greater than 25000m.
    """
    altitude = np.asarray(altitude)
    temperature = np.where(
        altitude <= h_p_tropo, T0 + beta_tropo * altitude, T0 + beta_tropo * h_p_tropo
    )
    if np.any(altitude > 25000):
        raise ValueError("Altitude out of range [0-25000m]")
    return temperature


def pressure_at_altitude_isa_bada4(altitude: FloatOrNDArray) -> FloatOrNDArray:
    """Return the pressure at the provided altitude(s).
    Units are SI (m, PA)

    Parameters
    ----------
    altitude : Union[float,NDArray]
        Altitude in meters.

    Returns
    -------
    NDArray
        Pressure in Pascals.

    Raises
    ------
    ValueError
        If altitude is greater than 25000m.
    """
    altitude = np.asarray(altitude)
    temperature = temperature_at_altitude_isa_bada4(altitude)
    p_tropo = p0 * ((T0 + beta_tropo * h_p_tropo) / T0) ** (-g0 / (beta_tropo * R_air))
    pressure = np.where(
        altitude <= h_p_tropo,
        p0 * (temperature / T0) ** (-g0 / (beta_tropo * R_air)),
        p_tropo
        * np.exp(
            -g0 / (R_air * (T0 + beta_tropo * h_p_tropo)) * (altitude - h_p_tropo)
        ),
    )
    return pressure


def altitude_from_pressure_isa_bada4(pressure: Union[float, NDArray]) -> NDArray:
    """Return the altitude at the provided pressure(s).
    Units are SI (PA, m)

    Parameters
    ----------
    pressure : Union[float,NDArray]
        Pressure in Pascals.

    Returns
    -------
    NDArray
        Altitude in meters.

    Raises
    ------
    ValueError
        If pressure is less than 0.
    """
    pressure = np.asarray(pressure)
    temperature_tropo = temperature_at_altitude_isa_bada4(h_p_tropo)
    pressure_tropo = p0 * (temperature_tropo / T0) ** (-g0 / (beta_tropo * R_air))
    altitude = np.where(
        pressure >= pressure_tropo,
        T0 / beta_tropo * ((pressure / p0) ** (-beta_tropo * R_air / g0) - 1),
        h_p_tropo
        - R_air
        * (T0 + beta_tropo * h_p_tropo)
        / g0
        * np.log(pressure / pressure_tropo),
    )
    return altitude


def calculate_speed_of_sound(temperature: Union[float, NDArray]) -> NDArray:
    """Calculate the speed of sound depending on the provided temperature(s).
    Units are SI (K, m/s)

    Parameters
    ----------
    temperature : Union[float,NDArray]
        Temperature in Kelvin.

    Returns
    -------
    NDArray
        Speed of sound in m/s.

    Raises
    ------
    ValueError
        If temperature is greater than 216.69K.
    """
    temperature = np.asarray(temperature)
    return np.sqrt(1.4 * 287.05 * temperature)


def speed_of_sound_at_altitude(altitude: Union[float, NDArray]) -> NDArray:
    """Calculate the speed of sound depending on the provided altitude(s).
    Units are SI (m, m/s)

    Parameters
    ----------
    altitude : Union[float,NDArray]
        Altitude in meters.

    Returns
    -------
    NDArray
        Speed of sound in m/s.

    Raises
    ------
    ValueError
        If altitude is greater than 25000m.
    """
    altitude = np.asarray(altitude)
    temperature = temperature_at_altitude_isa_bada4(altitude)
    return calculate_speed_of_sound(temperature)


def calculate_air_density(
    pressure: Union[float, NDArray], temperature: Union[float, NDArray]
) -> NDArray:
    """Calculate the air density depending on the provided pressure and temperature.
    Units are SI (Pa, K)

    Parameters
    ----------
    pressure : Union[float,NDArray]
        Pressure in Pascals.
    temperature : Union[float,NDArray]
        Temperature in Kelvin.

    Returns
    -------
    NDArray
        Air density in kg/m^3.
    """
    pressure = np.asarray(pressure)
    temperature = np.asarray(temperature)
    return pressure / (R_air * temperature)
