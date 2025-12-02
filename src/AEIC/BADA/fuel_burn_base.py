"""Base classes for fuel burn modelling."""

# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid


@dataclass
class BaseAircraftParameters(ABC):
    """Abstract class for aircraft parameters."""

    @abstractmethod
    def __init__(self) -> None:
        pass


class BaseFuelBurnModel(ABC):
    """Abstract class for fuel burn models."""

    @abstractmethod
    def __init__(self, aircraft_parameters) -> None:
        self.aircraft_parameters = aircraft_parameters
        pass

    @abstractmethod
    def calculate_specific_ground_range():
        pass

    @abstractmethod
    def iterate_flight_simulation_constant_initial_mass():
        pass

    def update_mass_vector(self, mass, specific_ground_range, segment_distance):
        specific_ground_range_corrected = np.where(
            specific_ground_range < 1, np.inf, specific_ground_range
        )
        mass[1:] = mass[0] - cumulative_trapezoid(
            1 / specific_ground_range_corrected, dx=segment_distance
        )
        return mass

    def update_mass_vector_backward(
        self, mass, specific_ground_range, segment_distance
    ):
        specific_ground_range_corrected = np.where(
            specific_ground_range < 1, np.inf, specific_ground_range
        )
        # backwards means last element stays and the rest get adjusted by
        # addition instead of subtraction
        cumulative_integral = cumulative_trapezoid(
            1 / specific_ground_range_corrected[::-1], dx=segment_distance
        )[::-1]
        mass[:-1] = mass[-1] + cumulative_integral
        return mass
