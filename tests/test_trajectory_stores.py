import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from AEIC.trajectories.field_sets import FieldMetadata, FieldSet
from AEIC.trajectories.store import TrajectoryStore
from AEIC.trajectories.trajectory import Trajectory


@dataclass
class Extras:
    """Extra fields add to trajectories for testing."""

    FIELD_SETS: ClassVar[list[FieldSet]] = [
        FieldSet(
            'demo',
            f1=FieldMetadata(description='Test 1', units='unit1'),
            f2=FieldMetadata(description='Test 2', units='unit2'),
            mf=FieldMetadata(
                metadata=True,
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
    def random(cls, npoints: int) -> 'Extras':
        return cls(
            f1=np.random.rand(npoints),
            f2=np.random.rand(npoints),
            mf=np.random.randint(0, 100),
        )

    @classmethod
    def fixed(cls, npoints: int, f1: float, f2: float, mf: int) -> 'Extras':
        return cls(f1=np.full(npoints, f1), f2=np.full(npoints, f2), mf=mf)


# Need a way of getting test trajectories with all their fields populated. The
# trajectory store doesn't like incomplete data.


def make_test_trajectory(npoints: int, seed: int, extras: bool = False) -> Trajectory:
    t = Trajectory(npoints, name=f'traj_{seed}', fieldsets=['demo'] if extras else None)
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
    t.flight_level_weight = np.ones(npoints)
    t.starting_mass = t.aircraft_mass[0]
    t.total_fuel_mass = t.fuel_mass[0] - t.fuel_mass[-1]
    t.NClm = npoints // 3
    t.NCrz = npoints // 3
    t.NDes = npoints - t.NClm - t.NCrz
    if extras:
        extra_fields = Extras.random(npoints)
        t.f1 = extra_fields.f1
        t.f2 = extra_fields.f2
        t.mf = extra_fields.mf
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
) -> TrajectoryStore:
    ts = TrajectoryStore.create(base_file=base_file, title=title)
    ts.add(make_test_trajectory(10, 1))
    ts.add(make_test_trajectory(15, 2))
    ts.close()
    return ts


def simple_check_ts(path: Path | str, title: str, lengths: list[int]):
    ts_read = TrajectoryStore.open(base_file=path)
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

    ts2 = TrajectoryStore.append(base_file=path)
    ts2.add(make_test_trajectory(20, 3))
    ts2.close()

    simple_check_ts(path, 'append case', [10, 15, 20])


@pytest.mark.skip(reason='long test case, enable manually')
def test_create_reopen_large(tmp_path: Path):
    # Create large TrajectoryStore linked with NetCDF file (~13 Gb) for
    # writing, close the NetCDF file, reopen for reading and check contents.

    path = tmp_path / 'test.nc'
    ts = TrajectoryStore.create(base_file=path)
    for i in range(1000000):
        ts.add(make_test_trajectory(100, i))
    ts.close()

    ts_read = TrajectoryStore.open(base_file=path)
    assert len(ts_read) == 1000000
    assert len(ts_read[200000]) == 100
    assert len(ts_read[999999]) == 100


def test_multi_threading(tmp_path: Path):
    result = None

    def worker(idx: int):
        nonlocal result
        path = tmp_path / f'test{idx}.nc'
        try:
            _ = simple_create_ts(base_file=path, title=f'thread {idx}')
            result = 'OK'
        except Exception:
            result = 'FAILED'

    path = tmp_path / 'test.nc'
    _ = simple_create_ts(base_file=path, title='main thread')
    t = threading.Thread(target=worker, args=(1,))
    t.start()
    t.join()
    assert result == 'FAILED'


def test_extra_fields_in_base_nc(tmp_path: Path):
    # Create TrajectoryStore with additional field set saved in base file.
    # (This should result in a file with a "base" group and a "demo" group.)

    path = tmp_path / 'test.nc'
    ts = TrajectoryStore.create(base_file=path)
    for i in range(1, 6):
        ts.add(make_test_trajectory(i * 5, i, extras=True))
    ts.close()

    ts_read = TrajectoryStore.open(base_file=path)
    assert len(ts_read) == 5
    assert ts_read.files[0].fieldsets == {'base', 'demo'}
    assert ts_read[2].f1.shape == (15,)
    assert ts_read[2].mf is not None


def test_extra_fields_in_base_nc_bad(tmp_path: Path):
    # (BAD VERSION OF ABOVE TEST): create TrajectoryStore with additional field
    # set saved in base file. (This should result in a file with a "base" group
    # and a "demo" group.)
    #
    # Add fields only to a sub-set of trajectories. Should result in an error
    # from the data dictionary hash check in TrajectoryStore.add().

    with pytest.raises(ValueError):
        path = tmp_path / 'test.nc'
        ts = TrajectoryStore.create(base_file=path)
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, i)
            if i % 2 == 1:
                t.add_fields(Extras.random(i * 5))
            ts.add(t)
        ts.close()


def test_extra_fields_in_associated_nc(tmp_path: Path):
    # Same thing as last (good) case, except save additional field set in an
    # associated file.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    ts = TrajectoryStore.create(
        base_file=path,
        associated_files=[(extra_path, ['demo'])],
    )
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts.add(t)
    ts.close()

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    ts_read = TrajectoryStore.open(base_file=path)
    assert len(ts_read) == 5
    assert len(ts_read.files) == 1
    assert ts_read.files[0].fieldsets == {'base'}
    t = ts_read[2]
    assert hasattr(t, 'aircraft_mass')
    assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    ts2_read = TrajectoryStore.open(base_file=path, associated_files=[extra_path])
    assert len(ts2_read) == 5
    assert len(ts2_read.files) == 2
    assert ts2_read.files[0].fieldsets == {'base'}
    assert ts2_read.files[1].fieldsets == {'demo'}
    assert ts2_read[2].f1.shape == (15,)
    assert ts2_read[2].mf is not None


def test_extra_fields_in_associated_nc_bad(tmp_path: Path):
    # Same idea as last case, except try to open associated file that doesn't
    # match the base file.

    base1 = tmp_path / 'base1.nc'
    base2 = tmp_path / 'base2.nc'
    extra1 = tmp_path / 'extra1.nc'
    ts1 = TrajectoryStore.create(
        base_file=base1,
        associated_files=[(extra1, ['demo'])],
    )
    for i in range(1, 6):
        ts1.add(make_test_trajectory(i * 5, i, extras=True))
    ts1.close()

    # Create another unrelated pair of files.
    ts2 = TrajectoryStore.create(title='Use case 5 (different)', base_file=base2)
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts2.add(t)
    ts2.close()

    # Try opening a base file with the wrong associated file.
    with pytest.raises(ValueError):
        _ = TrajectoryStore.open(base_file=base2, associated_files=[extra1])


def test_extra_fields_in_associated_nc_with_append(tmp_path: Path):
    # Equivalent of last (good) case with appending to the files in between
    # creating and reading the store.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    ts = TrajectoryStore.create(
        base_file=path,
        associated_files=[(extra_path, ['demo'])],
    )
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts.add(t)
    ts.close()

    ts2 = TrajectoryStore.append(base_file=path, associated_files=[extra_path])
    t = make_test_trajectory(10, 100)
    t.add_fields(Extras.random(10))
    ts2.add(t)
    ts2.close()

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    ts_read = TrajectoryStore.open(base_file=path)
    assert len(ts_read) == 6
    assert len(ts_read.files) == 1
    assert ts_read.files[0].fieldsets == {'base'}
    t = ts_read[2]
    assert hasattr(t, 'aircraft_mass')
    assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    ts2_read = TrajectoryStore.open(base_file=path, associated_files=[extra_path])
    assert len(ts2_read) == 6
    assert len(ts2_read.files) == 2
    assert ts2_read.files[0].fieldsets == {'base'}
    assert ts2_read.files[1].fieldsets == {'demo'}
    assert ts2_read[2].f1.shape == (15,)
    assert ts2_read[2].mf is not None


def test_save(tmp_path: Path):
    # Test saving TrajectoryStore to a different NetCDF file.

    # Create a TrajectoryStore in memory (not linked to a NetCDF file).
    ts = TrajectoryStore.create()
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts.add(t)

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    ts.save(base_file=path, associated_files=[(extra_path, ['demo'])])
    ts.close()

    ts_read = TrajectoryStore.open(base_file=path, associated_files=[extra_path])
    assert len(ts_read) == 5
    assert len(ts_read.files) == 2
    assert ts_read.files[0].fieldsets == {'base'}
    assert ts_read.files[1].fieldsets == {'demo'}
    assert ts_read[2].f1.shape == (15,)
    assert ts_read[2].mf is not None


def test_create_associated(tmp_path: Path):
    # Create associated file from existing TrajectoryStore.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    ts = TrajectoryStore.create(base_file=path)
    ts.add(make_test_trajectory(10, 1))
    ts.add(make_test_trajectory(15, 2))
    ts.close()

    ts_create_assoc = TrajectoryStore.open(base_file=path)
    ts_create_assoc.create_associated(
        associated_file=extra_path,
        fieldsets=['demo'],
        mapping_function=lambda traj: Extras.random(len(traj)),
    )
    ts_create_assoc.close()

    ts_read = TrajectoryStore.open(base_file=path, associated_files=[extra_path])
    assert len(ts_read) == 2
    assert len(ts_read.files) == 2
    assert ts_read.files[0].fieldsets == {'base'}
    assert ts_read.files[1].fieldsets == {'demo'}
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

    ts = TrajectoryStore.create(base_file=path)
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.fixed(i * 5, i * 0.1, i * 0.2, 10 * i))
        ts.add(t)
    ts.close()

    ts_create_assoc = TrajectoryStore.open(base_file=path)
    ts_create_assoc.create_associated(
        associated_file=extra,
        fieldsets=['demo'],
        mapping_function=lambda traj: Extras.fixed(len(traj), 123, 456, 12345),
    )

    # Case 1: just read original base file, which contains both field sets.
    # Should just get the base file values.
    ts_read1 = TrajectoryStore.open(base_file=path)
    assert len(ts_read1) == 5
    assert len(ts_read1.files) == 1
    assert ts_read1.files[0].fieldsets == {'base', 'demo'}
    assert ts_read1[1].f1[0] == 0.2
    assert ts_read1[1].mf == 20

    # Case 2: read with associated file, without override. Should get the base
    # file values.
    with pytest.warns(
        RuntimeWarning,
        match='FieldSet with name "demo" found in associated NetCDF file',
    ):
        ts_read2 = TrajectoryStore.open(base_file=path, associated_files=[extra])
        assert len(ts_read2) == 5
        assert len(ts_read2.files) == 2
        assert ts_read2.files[0].fieldsets == {'base', 'demo'}
        assert ts_read2.files[1].fieldsets == {'demo'}
        assert ts_read2[1].f1[0] == 0.2
        assert ts_read2[1].mf == 20

    # Case 3: read with associated file, *with* override. Should get the
    # associated file values.
    ts_read3 = TrajectoryStore.open(
        base_file=path, associated_files=[extra], override=True
    )
    assert len(ts_read3) == 5
    assert len(ts_read3.files) == 2
    assert ts_read3.files[0].fieldsets == {'base', 'demo'}
    assert ts_read3.files[1].fieldsets == {'demo'}
    assert ts_read3[1].f1[0] == 123
    assert ts_read3[1].mf == 12345


def test_basic_merging(tmp_path: Path):
    # Create a number of trajectory stores with names following a pattern.
    paths = []
    tss = []
    for i in range(4):
        path = tmp_path / f'test_{i}.nc'
        paths.append(path)
        ts = TrajectoryStore.create(base_file=path, title=f'store {i}')
        for j in range(2):
            t = make_test_trajectory((j + 1) * 5, j + i * 10)
            ts.add(t)
        ts.close()
        tss.append(ts)

    # Merge the stores into a new store.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(input_stores=paths, output_store=merged_path)

    # Make sure we can't open the merged store for append!
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append(base_file=merged_path)

    # Open the merged store and check contents.
    ts_merged = TrajectoryStore.open(base_file=merged_path)
    assert ts_merged.nc_linked is True
    assert len(ts_merged) == 8
    assert ts_merged[0].name == 'traj_0'
    assert ts_merged[7].name == 'traj_31'
    assert ts_merged[4].flight_time.shape == (5,)
    ts_merged.close()


def test_pattern_merging(tmp_path: Path):
    # Create a number of trajectory stores with names following a pattern.
    paths = []
    tss = []
    for i in range(10):
        path = tmp_path / f'test_{i:03d}.nc'
        paths.append(path)
        ts = TrajectoryStore.create(base_file=path, title=f'store {i}')
        for j in range(2):
            t = make_test_trajectory((j + 1) * 5, j + i * 10)
            ts.add(t)
        ts.close()
        tss.append(ts)

    # Merge the stores into a new store.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(
        input_stores_pattern=tmp_path / 'test_{index:03d}.nc',
        input_stores_index_range=(0, 9),
        output_store=merged_path,
    )

    # Open the merged store and check contents.
    ts_merged = TrajectoryStore.open(base_file=merged_path)
    assert ts_merged.nc_linked is True
    assert len(ts_merged) == 20
    assert ts_merged[0].name == 'traj_0'
    assert ts_merged[7].name == 'traj_31'
    assert ts_merged[4].flight_time.shape == (5,)
    ts_merged.close()


def test_merging_with_associated_files(tmp_path: Path):
    # How should this work? If you merge a set of base files that have
    # associated files, you should then be able to merge the associated files
    # and open the "base merged store" with the "associated merged store" to
    # get what you would hope for.

    # TODO: Think about what can go wrong here and add some error checking to
    # the code that opens the merged stores.

    #  1. Create stores with base + associated files.
    base_paths = []
    extra_paths = []
    for j in range(10):
        base_path = tmp_path / f'base{j}.nc'
        extra_path = tmp_path / f'extra{j}.nc'
        ts = TrajectoryStore.create(
            base_file=base_path,
            associated_files=[(extra_path, ['demo'])],
        )
        for i in range(1, 6):
            t = make_test_trajectory(i * 5, j * 5 + i)
            t.add_fields(Extras.random(i * 5))
            ts.add(t)
        ts.close()
        base_paths.append(base_path)
        extra_paths.append(extra_path)

    #  2. Merge base files.
    merged_base = tmp_path / 'merged_base.aeic-store'
    TrajectoryStore.merge(input_stores=base_paths, output_store=merged_base)

    #  3. Merge associated files.
    merged_associated = tmp_path / 'merged_extra.aeic-store'
    TrajectoryStore.merge(input_stores=extra_paths, output_store=merged_associated)

    #  4. Open merged base + merged associated and check contents.
    ts_merged = TrajectoryStore.open(
        base_file=merged_base, associated_files=[merged_associated]
    )
    assert ts_merged.nc_linked is True
    assert len(ts_merged) == 50
    for i in range(50):
        assert ts_merged[i].name == f'traj_{i + 1}'
    assert ts_merged[4].flight_time.shape == (25,)
    assert len(ts_merged.files) == 2
    assert ts_merged.files[0].fieldsets == {'base'}
    assert ts_merged.files[1].fieldsets == {'demo'}
    assert ts_merged[2].f1.shape == (15,)
    assert ts_merged[2].mf is not None
    ts_merged.close()


def test_indexing(tmp_path: Path):
    # 1. Create a unique set of flight IDs.
    ntrajs = 100
    seeds = random.sample(range(1000, 10000), ntrajs)

    # 2. Create trajectory store containing with those flight IDs.
    path = tmp_path / 'test.nc'
    ts = TrajectoryStore.create(base_file=path)
    for s in seeds:
        ts.add(make_test_trajectory(50, s))
    ts.close()

    # 3. Reopen the trajectory store.
    ts_read = TrajectoryStore.open(base_file=path)

    # 4. Look up trajectories by flight ID.
    for s in seeds:
        traj = ts_read.lookup(s)
        assert traj is not None
        assert traj.flight_id == s
