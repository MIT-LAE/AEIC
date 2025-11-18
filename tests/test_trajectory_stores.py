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


def make_test_trajectory(npoints: int, seed: int) -> Trajectory:
    t = Trajectory(npoints, name=f'traj_{seed}')
    t.fuelFlow = np.random.rand(npoints) * 5000 + 2000
    t.acMass = np.random.rand(npoints) * 50000 + 100000
    t.fuelMass = np.random.rand(npoints) * 50000 + 5000
    t.groundDist = np.linspace(0, 1000, npoints)
    t.altitude = np.linspace(0, 35000, npoints)
    t.FLs = t.altitude / 100
    t.rocs = np.random.randn(npoints) * 10
    t.flightTime = np.linspace(0, 3600, npoints)
    t.latitude = np.random.rand(npoints) * 180 - 90
    t.longitude = np.random.rand(npoints) * 360 - 180
    t.azimuth = np.random.rand(npoints) * 360
    t.heading = np.random.rand(npoints) * 360
    t.tas = np.random.rand(npoints) * 300 + 200
    t.groundSpeed = t.tas + np.random.randn(npoints) * 5
    t.FL_weight = np.ones(npoints)
    t.starting_mass = t.acMass[0]
    t.fuel_mass = t.fuelMass[0] - t.fuelMass[-1]
    t.NClm = npoints // 3
    t.NCrz = npoints // 3
    t.NDes = npoints - t.NClm - t.NCrz
    return t


def test_init_checking():
    # Missing NetCDF file name when creating or appending.
    with pytest.raises(ValueError):
        _ = TrajectoryStore.open()
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append()

    # Specifying global attributes in non-create modes.
    with pytest.raises(ValueError):
        _ = TrajectoryStore.open(nc_file='test.nc', title='Test')
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append(nc_file='test.nc', title='Test')

    # TODO: MORE HERE...


def simple_create_ts(
    nc_file: Path | None = None, title: str | None = None
) -> TrajectoryStore:
    ts = TrajectoryStore.create(nc_file=nc_file, title=title)
    ts.add(make_test_trajectory(10, 1))
    ts.add(make_test_trajectory(15, 2))
    ts.close()
    return ts


def simple_check_ts(path: Path, title: str, lengths: list[int]):
    ts_read = TrajectoryStore.open(nc_file=path)
    assert ts_read.global_attributes['title'] == title
    assert len(ts_read) == len(lengths)
    for i in range(len(lengths)):
        assert len(ts_read[i]) == lengths[i]


def test_create_reopen(tmp_path: Path):
    # Create a small TrajectoryStore, save to NetCDF, disabling further
    # appending (closes NetCDF file), reload from NetCDF.

    path = tmp_path / 'test.nc'
    simple_create_ts(nc_file=path, title='simple case')

    simple_check_ts(path, 'simple case', [10, 15])


def test_create_append_reopen(tmp_path: Path):
    # Create a TrajectoryStore, save to NetCDF, close, reopen file in append
    # mode, add another trajectory, close NetCDF file and reload from NetCDF.

    path = tmp_path / 'test.nc'
    simple_create_ts(nc_file=path, title='append case')

    ts2 = TrajectoryStore.append(nc_file=path)
    ts2.add(make_test_trajectory(20, 3))
    ts2.close()

    simple_check_ts(path, 'append case', [10, 15, 20])


@pytest.mark.skip(reason='long test case, enable manually')
def test_create_reopen_large(tmp_path: Path):
    # Create large TrajectoryStore linked with NetCDF file (~13 Gb) for
    # writing, close the NetCDF file, reopen for reading and check contents.

    path = tmp_path / 'test.nc'
    ts = TrajectoryStore.create(nc_file=path)
    for i in range(1000000):
        ts.add(make_test_trajectory(100, i))
    ts.close()

    ts_read = TrajectoryStore.open(nc_file=path)
    assert len(ts_read) == 1000000
    assert len(ts_read[200000]) == 100
    assert len(ts_read[999999]) == 100


def test_extra_fields_in_base_nc(tmp_path: Path):
    # Create TrajectoryStore with additional field set saved in base file.
    # (This should result in a file with a "base" group and a "demo" group.)

    path = tmp_path / 'test.nc'
    ts = TrajectoryStore.create(nc_file=path)
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts.add(t)
    ts.close()

    ts_read = TrajectoryStore.open(nc_file=path)
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
        ts = TrajectoryStore.create(nc_file=path)
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
        nc_file=path,
        associated_nc_files=[(extra_path, ['demo'])],
    )
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts.add(t)
    ts.close()

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    ts_read = TrajectoryStore.open(nc_file=path)
    assert len(ts_read) == 5
    assert len(ts_read.files) == 1
    assert ts_read.files[0].fieldsets == {'base'}
    t = ts_read[2]
    assert hasattr(t, 'acMass')
    assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    ts2_read = TrajectoryStore.open(nc_file=path, associated_nc_files=[extra_path])
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
        nc_file=base1,
        associated_nc_files=[(extra1, ['demo'])],
    )
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts1.add(t)
    ts1.close()

    # Create another unrelated pair of files.
    ts2 = TrajectoryStore.create(title='Use case 5 (different)', nc_file=base2)
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts2.add(t)
    ts2.close()

    # Try opening a base file with the wrong associated file.
    with pytest.raises(ValueError):
        _ = TrajectoryStore.open(nc_file=base2, associated_nc_files=[extra1])


def test_extra_fields_in_associated_nc_with_append(tmp_path: Path):
    # Equivalent of last (good) case with appending to the files in between
    # creating and reading the store.

    path = tmp_path / 'test.nc'
    extra_path = tmp_path / 'extra.nc'
    ts = TrajectoryStore.create(
        nc_file=path,
        associated_nc_files=[(extra_path, ['demo'])],
    )
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.random(i * 5))
        ts.add(t)
    ts.close()

    ts2 = TrajectoryStore.append(nc_file=path, associated_nc_files=[extra_path])
    t = make_test_trajectory(10, 100)
    t.add_fields(Extras.random(10))
    ts2.add(t)
    ts2.close()

    # Opening just the base NetCDF file (without the associated file) should
    # give trajectories without the extra fields.
    ts_read = TrajectoryStore.open(nc_file=path)
    assert len(ts_read) == 6
    assert len(ts_read.files) == 1
    assert ts_read.files[0].fieldsets == {'base'}
    t = ts_read[2]
    assert hasattr(t, 'acMass')
    assert not hasattr(t, 'f1')

    # Opening with the associated file should give trajectories with the
    # extra fields.
    ts2_read = TrajectoryStore.open(nc_file=path, associated_nc_files=[extra_path])
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
    ts.save(nc_file=path, associated_nc_files=[(extra_path, ['demo'])])
    ts.close()

    ts_read = TrajectoryStore.open(nc_file=path, associated_nc_files=[extra_path])
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
    ts = TrajectoryStore.create(nc_file=path)
    ts.add(make_test_trajectory(10, 1))
    ts.add(make_test_trajectory(15, 2))
    ts.close()

    ts_create_assoc = TrajectoryStore.open(nc_file=path)
    ts_create_assoc.create_associated(
        associated_nc_file=extra_path,
        fieldsets=['demo'],
        mapping_function=lambda traj: Extras.random(len(traj)),
    )
    ts_create_assoc.close()

    ts_read = TrajectoryStore.open(nc_file=path, associated_nc_files=[extra_path])
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

    ts = TrajectoryStore.create(nc_file=path)
    for i in range(1, 6):
        t = make_test_trajectory(i * 5, i)
        t.add_fields(Extras.fixed(i * 5, i * 0.1, i * 0.2, 10 * i))
        ts.add(t)
    ts.close()

    ts_create_assoc = TrajectoryStore.open(nc_file=path)
    ts_create_assoc.create_associated(
        associated_nc_file=extra,
        fieldsets=['demo'],
        mapping_function=lambda traj: Extras.fixed(len(traj), 123, 456, 12345),
    )

    # Case 1: just read original base file, which contains both field sets.
    # Should just get the base file values.
    ts_read1 = TrajectoryStore.open(nc_file=path)
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
        ts_read2 = TrajectoryStore.open(nc_file=path, associated_nc_files=[extra])
        assert len(ts_read2) == 5
        assert len(ts_read2.files) == 2
        assert ts_read2.files[0].fieldsets == {'base', 'demo'}
        assert ts_read2.files[1].fieldsets == {'demo'}
        assert ts_read2[1].f1[0] == 0.2
        assert ts_read2[1].mf == 20

    # Case 3: read with associated file, *with* override. Should get the
    # associated file values.
    ts_read3 = TrajectoryStore.open(
        nc_file=path, associated_nc_files=[extra], override=True
    )
    assert len(ts_read3) == 5
    assert len(ts_read3.files) == 2
    assert ts_read3.files[0].fieldsets == {'base', 'demo'}
    assert ts_read3.files[1].fieldsets == {'demo'}
    assert ts_read3[1].f1[0] == 123
    assert ts_read3[1].mf == 12345
