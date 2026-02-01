# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from AEIC.constants import T0, R_air, g0, p0

if TYPE_CHECKING:
    from AEIC.types import FloatOrNDArray


# Tropospheric temperature lapse rate [K/m]
beta_tropo = -0.0065  # (−6.5 K/km)

# Pressure-based tropopause altitude [m]
h_p_tropo = 11_000.0  # ISA tropopause altitude


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


def altitude_from_pressure_isa_bada4(pressure: float | NDArray) -> NDArray:
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


def calculate_speed_of_sound(temperature: float | NDArray) -> NDArray:
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


def speed_of_sound_at_altitude(altitude: float | NDArray) -> NDArray:
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
    pressure: float | NDArray, temperature: float | NDArray
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
