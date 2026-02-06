# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from AEIC.performance.types import ThrustModeValues
from AEIC.trajectories import (
    Dimension,
    Dimensions,
    FieldMetadata,
    FieldSet,
    TrajectoryStore,
)
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
    t.altitude = np.linspace(0, 35000, npoints)
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


def test_init_checking():
    # Missing NetCDF file name when creating or appending.
    with pytest.raises(ValueError):
        _ = TrajectoryStore.open()
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append()

    # Specifying global attributes in non-create modes.
    with pytest.raises(ValueError):
        _ = TrajectoryStore.open(base_file='test.nc', title='Test')
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append(base_file='test.nc', title='Test')

    # TODO: MORE HERE...


def simple_create_ts(
    base_file: Path | str | None = None, title: str | None = None
) -> None:
    with TrajectoryStore.create(base_file=base_file, title=title) as ts:
        ts.add(make_test_trajectory(10, 1))
        ts.add(make_test_trajectory(15, 2))


def simple_check_ts(path: Path | str, title: str, lengths: list[int]):
    with TrajectoryStore.open(base_file=path) as ts_read:
        assert ts_read.global_attributes['title'] == title
        assert len(ts_read) == len(lengths)
        for i in range(len(lengths)):
            assert len(ts_read[i]) == lengths[i]


def test_create_reopen(tmp_path: Path):
    # Create a small TrajectoryStore, save to NetCDF, disabling further
    # appending (closes NetCDF file), reload from NetCDF.

    path = tmp_path / 'test.nc'
    simple_create_ts(base_file=path, title='simple case')

    simple_check_ts(path, 'simple case', [10, 15])


def test_create_append_reopen(tmp_path: Path):
    # Create a TrajectoryStore, save to NetCDF, close, reopen file in append
    # mode, add another trajectory, close NetCDF file and reload from NetCDF.

    path = tmp_path / 'test.nc'
    simple_create_ts(base_file=path, title='append case')

    with TrajectoryStore.append(base_file=path) as ts:
        ts.add(make_test_trajectory(20, 3))

    simple_check_ts(path, 'append case', [10, 15, 20])


@pytest.mark.skip(reason='long test case, enable manually')
def test_create_reopen_large(tmp_path: Path):
    # Create large TrajectoryStore linked with NetCDF file (~13 Gb) for
    # writing, close the NetCDF file, reopen for reading and check contents.

    path = tmp_path / 'test.nc'
    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(1000000):
            ts.add(make_test_trajectory(100, i))
        tstart = datetime.now()
        ts._reindex()
        duration = datetime.now() - tstart
        print(f'Reindexing took {duration.total_seconds()} seconds')

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read) == 1000000
        assert len(ts_read[200000]) == 100
        assert len(ts_read[999999]) == 100


def test_read_nulls(tmp_path: Path):
    # Check re-reading of null values from trajectory store.

    path = tmp_path / 'test.nc'
    with TrajectoryStore.create(base_file=path, title='Nulls') as ts:
        ts.add(make_test_trajectory(10, 1, nulls=True))
        ts.add(make_test_trajectory(15, 2, nulls=True))

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert ts_read.global_attributes['title'] == 'Nulls'
        assert len(ts_read) == 2

        t0 = ts_read[0]
        assert len(t0) == 10
        assert t0.flight_id is None
        assert t0.name == ''
        assert len(t0.fuel_flow) == 10

        t1 = ts_read[1]
        assert len(t1) == 15
        assert t1.flight_id is None
        assert t1.name == ''
        assert len(t1.fuel_flow) == 15


def test_multi_threading(tmp_path: Path):
    result = None

    def worker(idx: int):
        nonlocal result
        path = tmp_path / f'test{idx}.nc'
        try:
            simple_create_ts(base_file=path, title=f'thread {idx}')
            result = 'OK'
        except Exception:
            result = 'FAILED'

    path = tmp_path / 'test.nc'
    simple_create_ts(base_file=path, title='main thread')
    t = threading.Thread(target=worker, args=(1,))
    t.start()
    t.join()
    assert result == 'FAILED'


def test_extra_fields_in_base_nc(tmp_path: Path):
    # Create TrajectoryStore with additional field set saved in base file.
    # (This should result in a file with a "base" group and a "simple_extras"
    # group.)

    path = tmp_path / 'test.nc'
    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(1, 6):
            ts.add(make_test_trajectory(i * 5, i, simple_extras=True))

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read) == 5
        assert ts_read.files[0].fieldsets == {'base', 'simple_extras'}
        assert ts_read[2].f1.shape == (15,)
        assert ts_read[2].mf is not None


def test_extra_fields_in_base_nc_bad(tmp_path: Path):
    # (BAD VERSION OF ABOVE TEST): create TrajectoryStore with additional field
    # set saved in base file. (This should result in a file with a "base" group
    # and a "simple_extras" group.)
    #
    # Add fields only to a sub-set of trajectories. Should result in an error
    # from the data dictionary hash check in TrajectoryStore.add().

    with pytest.raises(ValueError):
        path = tmp_path / 'test.nc'
        with TrajectoryStore.create(base_file=path) as ts:
            for i in range(1, 6):
                t = make_test_trajectory(i * 5, i)
                if i % 2 == 1:
                    t.add_fields(SimpleExtras.random(i * 5))
                ts.add(t)


def test_extra_fields_in_associated_nc(tmp_path: Path):
    # Same thing as last (good) case, except save additional field set in an
    # associated file.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    with TrajectoryStore.create(
        base_file=path,
        associated_files=[(extra_path, ['simple_extras'])],
    ) as ts:
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(SimpleExtras.random(i * 5))
            ts.add(t)

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read) == 5
        assert len(ts_read.files) == 1
        assert ts_read.files[0].fieldsets == {'base'}
        t = ts_read[2]
        assert hasattr(t, 'aircraft_mass')
        assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    with TrajectoryStore.open(
        base_file=path, associated_files=[extra_path]
    ) as ts2_read:
        assert len(ts2_read) == 5
        assert len(ts2_read.files) == 2
        assert ts2_read.files[0].fieldsets == {'base'}
        assert ts2_read.files[1].fieldsets == {'simple_extras'}
        assert ts2_read[2].f1.shape == (15,)
        assert ts2_read[2].mf is not None


def test_extra_fields_in_associated_nc_bad(tmp_path: Path):
    # Same idea as last case, except try to open associated file that doesn't
    # match the base file.

    base1 = tmp_path / 'base1.nc'
    base2 = tmp_path / 'base2.nc'
    extra1 = tmp_path / 'extra1.nc'
    with TrajectoryStore.create(
        base_file=base1,
        associated_files=[(extra1, ['simple_extras'])],
    ) as ts1:
        for i in range(1, 6):
            ts1.add(make_test_trajectory(i * 5, i, simple_extras=True))

    # Create another unrelated pair of files.
    with TrajectoryStore.create(title='Use case 5 (different)', base_file=base2) as ts2:
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(SimpleExtras.random(i * 5))
            ts2.add(t)

    # Try opening a base file with the wrong associated file.
    with pytest.raises(ValueError):
        TrajectoryStore.open(base_file=base2, associated_files=[extra1])


def test_extra_fields_in_associated_nc_with_append(tmp_path: Path):
    # Equivalent of last (good) case with appending to the files in between
    # creating and reading the store.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    with TrajectoryStore.create(
        base_file=path,
        associated_files=[(extra_path, ['simple_extras'])],
    ) as ts:
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(SimpleExtras.random(i * 5))
            ts.add(t)

    with TrajectoryStore.append(base_file=path, associated_files=[extra_path]) as ts2:
        t = make_test_trajectory(10, 100)
        t.add_fields(SimpleExtras.random(10))
        ts2.add(t)

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read) == 6
        assert len(ts_read.files) == 1
        assert ts_read.files[0].fieldsets == {'base'}
        t = ts_read[2]
        assert hasattr(t, 'aircraft_mass')
        assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    with TrajectoryStore.open(
        base_file=path, associated_files=[extra_path]
    ) as ts2_read:
        assert len(ts2_read) == 6
        assert len(ts2_read.files) == 2
        assert ts2_read.files[0].fieldsets == {'base'}
        assert ts2_read.files[1].fieldsets == {'simple_extras'}
        assert ts2_read[2].f1.shape == (15,)
        assert ts2_read[2].mf is not None


def test_save(tmp_path: Path):
    # Test saving TrajectoryStore to a different NetCDF file.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'

    # Create a TrajectoryStore in memory (not linked to a NetCDF file).
    with TrajectoryStore.create() as ts:
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(SimpleExtras.random(i * 5))
            ts.add(t)

        ts.save(base_file=path, associated_files=[(extra_path, ['simple_extras'])])

    with TrajectoryStore.open(base_file=path, associated_files=[extra_path]) as ts_read:
        assert len(ts_read) == 5
        assert len(ts_read.files) == 2
        assert ts_read.files[0].fieldsets == {'base'}
        assert ts_read.files[1].fieldsets == {'simple_extras'}
        assert ts_read[2].f1.shape == (15,)
        assert ts_read[2].mf is not None


def test_create_associated(tmp_path: Path):
    # Create associated file from existing TrajectoryStore.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    with TrajectoryStore.create(base_file=path) as ts:
        ts.add(make_test_trajectory(10, 1))
        ts.add(make_test_trajectory(15, 2))

    with TrajectoryStore.open(base_file=path) as ts_create_assoc:
        ts_create_assoc.create_associated(
            associated_file=extra_path,
            fieldsets=['simple_extras'],
            mapping_function=lambda traj: SimpleExtras.random(len(traj)),
        )

    with TrajectoryStore.open(base_file=path, associated_files=[extra_path]) as ts_read:
        assert len(ts_read) == 2
        assert len(ts_read.files) == 2
        assert ts_read.files[0].fieldsets == {'base'}
        assert ts_read.files[1].fieldsets == {'simple_extras'}
        assert ts_read[1].f1.shape == (15,)
        assert ts_read[1].mf is not None


def test_fieldset_override(tmp_path: Path):
    # For this test, we need to create a base file with some extra fields, then
    # we need to create an associated file with the same extra fields but
    # different values. We should be able to override the base file fields with
    # the associated file fields using the `override` option to
    # `TrajectoryStore.open`.

    path = tmp_path / 'test.nc'
    extra = tmp_path / 'extra.nc'

    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(SimpleExtras.fixed(i * 5, i * 0.1, i * 0.2, 10 * i))
            ts.add(t)

    with TrajectoryStore.open(base_file=path) as ts_create_assoc:
        ts_create_assoc.create_associated(
            associated_file=extra,
            fieldsets=['simple_extras'],
            mapping_function=lambda traj: SimpleExtras.fixed(
                len(traj), 123, 456, 12345
            ),
        )

    # Case 1: just read original base file, which contains both field sets.
    # Should just get the base file values.
    with TrajectoryStore.open(base_file=path) as ts_read1:
        assert len(ts_read1) == 5
        assert len(ts_read1.files) == 1
        assert ts_read1.files[0].fieldsets == {'base', 'simple_extras'}
        assert ts_read1[1].f1[0] == 0.2
        assert ts_read1[1].mf == 20

    # Case 2: read with associated file, without override. Should get the base
    # file values.
    with pytest.warns(
        RuntimeWarning,
        match='FieldSet with name "simple_extras" found in associated NetCDF file',
    ):
        with TrajectoryStore.open(base_file=path, associated_files=[extra]) as ts_read2:
            assert len(ts_read2) == 5
            assert len(ts_read2.files) == 2
            assert ts_read2.files[0].fieldsets == {'base', 'simple_extras'}
            assert ts_read2.files[1].fieldsets == {'simple_extras'}
            assert ts_read2[1].f1[0] == 0.2
            assert ts_read2[1].mf == 20

    # Case 3: read with associated file, *with* override. Should get the
    # associated file values.
    with TrajectoryStore.open(
        base_file=path, associated_files=[extra], override=True
    ) as ts_read3:
        assert len(ts_read3) == 5
        assert len(ts_read3.files) == 2
        assert ts_read3.files[0].fieldsets == {'base', 'simple_extras'}
        assert ts_read3.files[1].fieldsets == {'simple_extras'}
        assert ts_read3[1].f1[0] == 123
        assert ts_read3[1].mf == 12345


def test_basic_merging(tmp_path: Path):
    # Create a number of trajectory stores with names following a pattern.
    paths = []
    for i in range(4):
        path = tmp_path / f'test_{i}.nc'
        paths.append(path)
        with TrajectoryStore.create(base_file=path, title=f'store {i}') as ts:
            for j in range(2):
                t = make_test_trajectory((j + 1) * 5, j + i * 10)
                ts.add(t)

    # Merge the stores into a new store.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(input_stores=paths, output_store=merged_path)

    # Make sure we can't open the merged store for append!
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append(base_file=merged_path)

    # Open the merged store and check contents.
    with TrajectoryStore.open(base_file=merged_path) as ts_merged:
        assert ts_merged.nc_linked is True
        assert len(ts_merged) == 8
        assert ts_merged[0].name == 'traj_0'
        assert ts_merged[7].name == 'traj_31'
        assert ts_merged[4].flight_time.shape == (5,)


def test_pattern_merging(tmp_path: Path):
    # Create a number of trajectory stores with names following a pattern.
    paths = []
    for i in range(10):
        path = tmp_path / f'test_{i:03d}.nc'
        paths.append(path)
        with TrajectoryStore.create(base_file=path, title=f'store {i}') as ts:
            for j in range(2):
                t = make_test_trajectory((j + 1) * 5, j + i * 10)
                ts.add(t)

    # Merge the stores into a new store.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(
        input_stores_pattern=tmp_path / 'test_{index:03d}.nc',
        input_stores_index_range=(0, 9),
        output_store=merged_path,
    )

    # Open the merged store and check contents.
    with TrajectoryStore.open(base_file=merged_path) as ts_merged:
        assert ts_merged.nc_linked is True
        assert len(ts_merged) == 20
        assert ts_merged[0].name == 'traj_0'
        assert ts_merged[7].name == 'traj_31'
        assert ts_merged[4].flight_time.shape == (5,)


def test_merging_with_associated_files(tmp_path: Path):
    # How should this work? If you merge a set of base files that have
    # associated files, you should then be able to merge the associated files
    # and open the "base merged store" with the "associated merged store" to
    # get what you would hope for.

    # TODO: Think about what can go wrong here and add some error checking to
    # the code that opens the merged stores.

    #  1. Create stores with base + associated files.
    base_paths = []
    simple_extra_paths = []
    complex_extra_paths = []
    for j in range(10):
        base_path = tmp_path / f'base{j}.nc'
        simple_extra_path = tmp_path / f'simple{j}.nc'
        complex_extra_path = tmp_path / f'complex{j}.nc'
        with TrajectoryStore.create(
            base_file=base_path,
            associated_files=[
                (simple_extra_path, ['simple_extras']),
                (complex_extra_path, ['complex_extras']),
            ],
        ) as ts:
            for i in range(1, 6):
                t = make_test_trajectory(i * 5, j * 5 + i)
                t.add_fields(SimpleExtras.random(i * 5))
                t.add_fields(ComplexExtras.random(i * 5))
                ts.add(t)
        base_paths.append(base_path)
        simple_extra_paths.append(simple_extra_path)
        complex_extra_paths.append(complex_extra_path)

    #  2. Merge base files.
    merged_base = tmp_path / 'merged_base.aeic-store'
    TrajectoryStore.merge(input_stores=base_paths, output_store=merged_base)

    #  3. Merge associated files.
    merged_simple_associated = tmp_path / 'merged_simple_extra.aeic-store'
    merged_complex_associated = tmp_path / 'merged_complex_extra.aeic-store'
    TrajectoryStore.merge(
        input_stores=simple_extra_paths, output_store=merged_simple_associated
    )
    TrajectoryStore.merge(
        input_stores=complex_extra_paths, output_store=merged_complex_associated
    )

    #  4. Open merged base + merged associated and check contents.
    with TrajectoryStore.open(
        base_file=merged_base,
        associated_files=[merged_simple_associated, merged_complex_associated],
    ) as ts_merged:
        assert ts_merged.nc_linked is True
        assert len(ts_merged) == 50
        for i in range(50):
            assert ts_merged[i].name == f'traj_{i + 1}'
        assert ts_merged[4].flight_time.shape == (25,)
        assert len(ts_merged.files) == 3
        assert ts_merged.files[0].fieldsets == {'base'}
        assert ts_merged.files[1].fieldsets == {'simple_extras'}
        assert ts_merged.files[2].fieldsets == {'complex_extras'}
        assert ts_merged[2].f1.shape == (15,)
        assert ts_merged[2].mf is not None
        _check_complex(ts_merged, repeats=10)


def test_indexing(tmp_path: Path):
    # 1. Create a unique set of flight IDs.
    ntrajs = 100
    seeds = random.sample(range(1000, 10000), ntrajs)

    # 2. Create trajectory store containing those flight IDs.
    path = tmp_path / 'test.nc'
    with TrajectoryStore.create(base_file=path) as ts:
        for s in seeds:
            ts.add(make_test_trajectory(50, s))

    # 3. Reopen the trajectory store.
    with TrajectoryStore.open(base_file=path) as ts_read:
        # 4. Look up trajectories by flight ID.
        for s in seeds:
            traj = ts_read.get_flight(s)
            assert traj is not None
            assert traj.flight_id == s


def test_merged_store_indexing(tmp_path: Path):
    # 1. Create a unique set of flight IDs.
    ntrajs = 1000
    seeds = random.sample(range(10000, 100000), ntrajs)

    # 2. Create individual trajectory stores containing those flight IDs.
    paths = []
    for i in range(10):
        path = tmp_path / f'test{i}.nc'
        paths.append(path)
        with TrajectoryStore.create(base_file=path) as ts:
            for s in seeds[i * 100 : (i + 1) * 100]:
                ts.add(make_test_trajectory(50, s))

    # 3. Merge the stores.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(input_stores=paths, output_store=merged_path)

    # 4. Open the merged store.
    with TrajectoryStore.open(base_file=merged_path) as ts_read:
        # 5. Look up trajectories by flight ID.
        for s in seeds:
            traj = ts_read.get_flight(s)
            assert traj is not None
            assert traj.flight_id == s


def _check_complex(ts_read: TrajectoryStore, repeats: int = 1):
    assert len(ts_read) == 5 * repeats
    for i in range(5):
        assert isinstance(ts_read[i].tot, SpeciesValues)
        for sp in ts_read[i].tot:
            assert isinstance(ts_read[i].tot[sp], float)
        assert isinstance(ts_read[i].seg, SpeciesValues)
        for sp in ts_read[i].seg:
            v = ts_read[i].seg[sp]
            assert isinstance(v, np.ndarray)
            assert v.shape == ((i + 1) * 5,)
        assert isinstance(ts_read[i].tm, ThrustModeValues)
        assert isinstance(ts_read[i].tm2, SpeciesValues)
        for sp in ts_read[i].tm2:
            assert isinstance(ts_read[i].tm2[sp], ThrustModeValues)


def test_complex_extra_fields_in_base_nc(tmp_path: Path):
    # Create TrajectoryStore with additional field set involving complex
    # variables saved in base file. (This should result in a file with a "base"
    # group and a "complex_extras" group.)

    path = tmp_path / 'test.nc'
    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(1, 6):
            ts.add(make_test_trajectory(i * 5, i, complex_extras=True))

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert ts_read.files[0].fieldsets == {'base', 'complex_extras'}
        _check_complex(ts_read)


def test_complex_extra_fields_in_associated_nc(tmp_path: Path):
    # Same thing as last case, except save additional field set in an
    # associated file.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    with TrajectoryStore.create(
        base_file=path,
        associated_files=[(extra_path, ['complex_extras'])],
    ) as ts:
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(ComplexExtras.random(i * 5))
            ts.add(t)

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read) == 5
        assert len(ts_read.files) == 1
        assert ts_read.files[0].fieldsets == {'base'}
        t = ts_read[2]
        assert hasattr(t, 'aircraft_mass')
        assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    with TrajectoryStore.open(
        base_file=path, associated_files=[extra_path]
    ) as ts2_read:
        assert len(ts2_read.files) == 2
        assert ts2_read.files[0].fieldsets == {'base'}
        assert ts2_read.files[1].fieldsets == {'complex_extras'}
        _check_complex(ts2_read)


def test_create_associated_complex(tmp_path: Path):
    # Create associated file from existing TrajectoryStore.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    with TrajectoryStore.create(base_file=path) as ts:
        ts.add(make_test_trajectory(10, 1))
        ts.add(make_test_trajectory(15, 2))

    with TrajectoryStore.open(base_file=path) as ts_create_assoc:
        ts_create_assoc.create_associated(
            associated_file=extra_path,
            fieldsets=['complex_extras'],
            mapping_function=lambda traj: ComplexExtras.random(len(traj)),
        )

    with TrajectoryStore.open(base_file=path, associated_files=[extra_path]) as ts_read:
        assert len(ts_read) == 2
        assert len(ts_read.files) == 2
        assert ts_read.files[0].fieldsets == {'base'}
        assert ts_read.files[1].fieldsets == {'complex_extras'}
        for i in range(2):
            assert isinstance(ts_read[i].tot, SpeciesValues)
            for sp in ts_read[i].tot:
                assert isinstance(ts_read[i].tot[sp], float)
            assert isinstance(ts_read[i].seg, SpeciesValues)
            for sp in ts_read[i].seg:
                v = ts_read[i].seg[sp]
                assert isinstance(v, np.ndarray)
                assert v.shape == (10 + i * 5,)
            assert isinstance(ts_read[i].tm, ThrustModeValues)
            assert isinstance(ts_read[i].tm2, SpeciesValues)
            for sp in ts_read[i].tm2:
                assert isinstance(ts_read[i].tm2[sp], ThrustModeValues)
