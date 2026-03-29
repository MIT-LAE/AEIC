# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from AEIC.performance.types import ThrustModeValues
from AEIC.storage import Dimension, Dimensions, FieldMetadata, FieldSet
from AEIC.trajectories.trajectory import Trajectory
from AEIC.types import Species, SpeciesValues


@dataclass
class SimpleExtras:
    """Extra fields (simple types only) add to trajectories for testing."""

    FIELD_SETS: ClassVar[list[FieldSet]] = [
        FieldSet(
            'simple_extras',
            f1=FieldMetadata(description='Test 1', units='unit1'),
            f2=FieldMetadata(description='Test 2', units='unit2'),
            mf=FieldMetadata(
                dimensions=Dimensions(Dimension.TRAJECTORY),
                field_type=np.int32,
                description='Test metadata',
                units='unit3',
            ),
        )
    ]

    # Note type of mf is int, not np.int32. Conversion is done by the
    # trajectory store when saving data. The field_type in FieldMetadata has to
    # be a Numpy type!

    f1: np.ndarray
    f2: np.ndarray
    mf: int

    @classmethod
    def random(cls, npoints: int) -> SimpleExtras:
        return cls(
            f1=np.random.rand(npoints),
            f2=np.random.rand(npoints),
            mf=np.random.randint(0, 100),
        )

    @classmethod
    def fixed(cls, npoints: int, f1: float, f2: float, mf: int) -> SimpleExtras:
        return cls(f1=np.full(npoints, f1), f2=np.full(npoints, f2), mf=mf)


@dataclass
class ComplexExtras:
    """Extra fields (complex types including species and thrust mode
    dependence) add to trajectories for testing."""

    FIELD_SETS: ClassVar[list[FieldSet]] = [
        FieldSet(
            'complex_extras',
            tot=FieldMetadata(
                dimensions=Dimensions.from_abbrev('TS'),
                description='Total per-species value',
                units='g',
            ),
            seg=FieldMetadata(
                dimensions=Dimensions.from_abbrev('TSP'),
                description='Per-segment per-species value',
                units='g',
            ),
            tm=FieldMetadata(
                dimensions=Dimensions.from_abbrev('TM'),
                description='Per-thrust mode value',
                units='g',
            ),
            tm2=FieldMetadata(
                dimensions=Dimensions.from_abbrev('TSM'),
                description='Per-species per-thrust mode value',
                units='g',
            ),
        )
    ]

    tot: SpeciesValues[float]
    seg: SpeciesValues[np.ndarray]
    tm: ThrustModeValues
    tm2: SpeciesValues[ThrustModeValues]

    @classmethod
    def random(cls, npoints: int) -> ComplexExtras:
        return cls(
            tot=SpeciesValues(
                {Species.CO2: random.uniform(0, 10), Species.H2O: random.uniform(0, 10)}
            ),
            seg=SpeciesValues(
                {
                    Species.CO2: np.random.rand(npoints),
                    Species.H2O: np.random.rand(npoints),
                }
            ),
            tm=ThrustModeValues(
                random.uniform(0, 10),
                random.uniform(0, 10),
                random.uniform(0, 10),
                random.uniform(0, 10),
            ),
            tm2=SpeciesValues(
                {
                    Species.CO2: ThrustModeValues(
                        random.uniform(0, 10),
                        random.uniform(0, 10),
                        random.uniform(0, 10),
                        random.uniform(0, 10),
                    ),
                    Species.H2O: ThrustModeValues(
                        random.uniform(0, 10),
                        random.uniform(0, 10),
                        random.uniform(0, 10),
                        random.uniform(0, 10),
                    ),
                }
            ),
        )


def make_test_trajectory(
    npoints: int,
    seed: int,
    simple_extras: bool = False,
    complex_extras: bool = False,
    nulls: bool = False,
) -> Trajectory:
    # Need a way of getting test trajectories with all their fields populated.
    # The trajectory store doesn't like incomplete data.

    fieldsets = []
    if simple_extras:
        fieldsets.append('simple_extras')
    if complex_extras:
        fieldsets.append('complex_extras')
    if len(fieldsets) == 0:
        fieldsets = None

    t = Trajectory(npoints, name=f'traj_{seed}', fieldsets=fieldsets)
    t.flight_id = seed
    t.fuel_flow = np.random.rand(npoints) * 5000 + 2000
    t.aircraft_mass = np.random.rand(npoints) * 50000 + 100000
    t.fuel_mass = np.random.rand(npoints) * 50000 + 5000
    t.ground_distance = np.linspace(0, 1000, npoints)
    t.altitude = np.linspace(0, 25000, npoints)
    t.flight_level = t.altitude / 100
    t.rate_of_climb = np.random.randn(npoints) * 10
    t.flight_time = np.linspace(0, 3600, npoints)
    t.latitude = np.random.rand(npoints) * 180 - 90
    t.longitude = np.random.rand(npoints) * 360 - 180
    t.azimuth = np.random.rand(npoints) * 360
    t.heading = np.random.rand(npoints) * 360
    t.true_airspeed = np.random.rand(npoints) * 300 + 200
    t.ground_speed = t.true_airspeed + np.random.randn(npoints) * 5
    t.starting_mass = t.aircraft_mass[0]
    t.total_fuel_mass = t.fuel_mass[0] - t.fuel_mass[-1]
    t.n_climb = npoints // 3
    t.n_cruise = npoints // 3
    t.n_descent = npoints - t.n_climb - t.n_cruise
    if simple_extras:
        extra_fields = SimpleExtras.random(npoints)
        t.f1 = extra_fields.f1
        t.f2 = extra_fields.f2
        t.mf = extra_fields.mf
    if complex_extras:
        extra_fields = ComplexExtras.random(npoints)
        t.tot = extra_fields.tot
        t.seg = extra_fields.seg
        t.tm = extra_fields.tm
        t.tm2 = extra_fields.tm2
    if nulls:
        t.flight_id = None
        t.name = None
    return t
