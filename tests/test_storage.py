# TODO: Remove this when we move to Python 3.14+.
from __future__ import annotations

import random
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from AEIC.config import Config
from AEIC.performance.types import ThrustModeValues
from AEIC.storage import (
    Container,
    Dimension,
    Dimensions,
    FieldMetadata,
    FieldSet,
    access_recorder,
    track_file_accesses,
)
from AEIC.trajectories import TrajectoryStore
from AEIC.trajectories.trajectory import Trajectory
from AEIC.types import SpeciesValues
from tests.subproc import run_in_subprocess
from tests.utils import ComplexExtras, SimpleExtras, make_test_trajectory


def test_init_checking(tmp_path: Path):
    # Missing NetCDF file name when reading or appending. (Required outside of
    # CREATE mode.)
    with pytest.raises(ValueError, match='base_file required'):
        _ = TrajectoryStore.open()
    with pytest.raises(ValueError, match='base_file required'):
        _ = TrajectoryStore.append()

    # Specifying any of the four global-attribute fields in non-CREATE modes.
    for kwargs in (
        {'title': 'T'},
        {'comment': 'C'},
        {'history': 'H'},
        {'source': 'S'},
    ):
        with pytest.raises(ValueError, match='global attributes'):
            _ = TrajectoryStore.open(base_file='test.nc', **kwargs)
        with pytest.raises(ValueError, match='global attributes'):
            _ = TrajectoryStore.append(base_file='test.nc', **kwargs)

    # `override` and `force_fieldset_matches` are READ-only options.
    with pytest.raises(ValueError, match='override may only be specified in READ'):
        _ = TrajectoryStore.create(override=True)
    with pytest.raises(ValueError, match='override may only be specified in READ'):
        _ = TrajectoryStore.append(base_file='test.nc', override=True)
    with pytest.raises(ValueError, match='force_fieldset_matches'):
        _ = TrajectoryStore.create(force_fieldset_matches=True)
    with pytest.raises(ValueError, match='force_fieldset_matches'):
        _ = TrajectoryStore.append(base_file='test.nc', force_fieldset_matches=True)

    # CREATE mode with `base_file=None` cannot also carry associated files —
    # there is no on-disk store to associate them with yet.
    with pytest.raises(
        ValueError, match='associated_files may only be specified in CREATE mode'
    ):
        _ = TrajectoryStore.create(
            associated_files=[(tmp_path / 'extra.nc', ['simple_extras'])]
        )


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


@pytest.mark.forked
def test_create_reopen(tmp_path: Path):
    # Create a small TrajectoryStore, save to NetCDF, disabling further
    # appending (closes NetCDF file), reload from NetCDF. Verifies both
    # structural metadata (title, length, indexing) *and* per-field numeric
    # equality between what was written and what comes back, so a regression
    # that silently permutes axes or rounds values on serialize is caught
    # here rather than passing on hasattr/shape checks alone.
    np.random.seed(0)
    path = tmp_path / 'test.nc'
    originals = [make_test_trajectory(10, 1), make_test_trajectory(15, 2)]
    with TrajectoryStore.create(base_file=path, title='simple case') as ts:
        for t in originals:
            ts.add(t)

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert ts_read.global_attributes['title'] == 'simple case'
        assert len(ts_read) == 2
        for i, original in enumerate(originals):
            loaded = ts_read[i]
            assert isinstance(loaded, Trajectory)
            _assert_trajectories_equal(loaded, original)
        with pytest.raises(IndexError):
            ts_read[10]


@pytest.mark.forked
def test_create_append_reopen(tmp_path: Path):
    # Create a TrajectoryStore, save to NetCDF, close, reopen file in append
    # mode, add another trajectory, close NetCDF file and reload from NetCDF.

    path = tmp_path / 'test.nc'
    simple_create_ts(base_file=path, title='append case')

    with TrajectoryStore.append(base_file=path) as ts:
        ts.add(make_test_trajectory(20, 3))
        ts.sync()

    simple_check_ts(path, 'append case', [10, 15, 20])


@pytest.mark.skip(reason='long test case, enable manually')
@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
def test_multi_threading(tmp_path: Path):
    """TrajectoryStore is documented as single-threaded; constructing one
    from a worker thread after the main thread has used the class must
    raise the guard `RuntimeError` (see store.py:378), not silently
    proceed and corrupt netCDF state.
    """
    captured: BaseException | None = None

    def worker(idx: int):
        nonlocal captured
        try:
            simple_create_ts(
                base_file=tmp_path / f'test{idx}.nc', title=f'thread {idx}'
            )
        except BaseException as e:
            captured = e

    simple_create_ts(base_file=tmp_path / 'test.nc', title='main thread')
    t = threading.Thread(target=worker, args=(1,))
    t.start()
    t.join()

    assert isinstance(captured, RuntimeError), captured
    assert 'multiple TrajectoryStore instances' in str(captured)


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


def basic_merging_create_stores(tmp_path: Path):
    # Create a number of trajectory stores with names following a pattern.
    Config.load()
    paths = []
    for i in range(4):
        path = tmp_path / f'test_{i}.nc'
        paths.append(path)
        with TrajectoryStore.create(base_file=path, title=f'store {i}') as ts:
            for j in range(2):
                t = make_test_trajectory((j + 1) * 5, j + i * 10)
                ts.add(t)


def basic_merging_check_merged_store(merged_path: Path):
    # Open the merged store and check contents.
    Config.load()
    with TrajectoryStore.open(base_file=merged_path) as ts_merged:
        assert ts_merged.nc_linked is True
        assert len(ts_merged) == 8
        assert ts_merged[0].name == 'traj_0'
        assert ts_merged[7].name == 'traj_31'
        assert ts_merged[4].flight_time.shape == (5,)


@pytest.mark.forked
def test_basic_merging(tmp_path: Path):
    # Create a number of trajectory stores with names following a pattern.
    run_in_subprocess(basic_merging_create_stores, tmp_path)
    paths = []
    for i in range(4):
        path = tmp_path / f'test_{i}.nc'
        paths.append(path)

    # Merge the stores into a new store.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(input_stores=paths, output_store=merged_path)

    # Make sure we can't open the merged store for append!
    with pytest.raises(ValueError):
        _ = TrajectoryStore.append(base_file=merged_path)

    # Open the merged store and check contents.
    run_in_subprocess(basic_merging_check_merged_store, merged_path)


# Split test into subprocesses to avoid issues with NetCDF libraries.


def pattern_merge_create_stores(tmp_path, test_data_dir):
    Config.load(data_path_overrides=[test_data_dir])
    for i in range(10):
        path = tmp_path / f'test_{i:03d}.nc'

        with TrajectoryStore.create(base_file=path, title=f'store {i}') as ts:
            for j in range(2):
                t = make_test_trajectory((j + 1) * 5, j + i * 10)
                ts.add(t)


def pattern_merge_check_merged_store(merged_path, test_data_dir):
    Config.load(data_path_overrides=[test_data_dir])
    with TrajectoryStore.open(base_file=merged_path) as ts_merged:
        assert ts_merged.nc_linked is True
        assert len(ts_merged) == 20
        assert ts_merged[0].name == 'traj_0'
        assert ts_merged[7].name == 'traj_31'
        assert ts_merged[4].flight_time.shape == (5,)


def test_pattern_merging(test_data_dir, tmp_path):
    # PART 1 in subprocess
    run_in_subprocess(pattern_merge_create_stores, tmp_path, test_data_dir)

    merged_path = tmp_path / "merged.aeic-store"

    TrajectoryStore.merge(
        input_stores_pattern=tmp_path / 'test_{index:03d}.nc',
        input_stores_index_range=(0, 9),
        output_store=merged_path,
    )

    # PART 2 in subprocess
    run_in_subprocess(pattern_merge_check_merged_store, merged_path, test_data_dir)


# Split test into subprocesses to avoid issues with NetCDF libraries.


def associated_files_merge_create_stores(tmp_path, test_data_dir):
    #  1. Create stores with base + associated files.
    Config.load(data_path_overrides=[test_data_dir])
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


def associated_files_merge_check_merged_stores(
    merged_base, merged_simple_associated, merged_complex_associated, test_data_dir
):
    #  4. Open merged base + merged associated and check contents.
    Config.load(data_path_overrides=[test_data_dir])
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


def test_merging_with_associated_files(tmp_path, test_data_dir):
    # How should this work? If you merge a set of base files that have
    # associated files, you should then be able to merge the associated files
    # and open the "base merged store" with the "associated merged store" to
    # get what you would hope for.

    # TODO: Think about what can go wrong here and add some error checking to
    # the code that opens the merged stores.

    #  1. Create stores with base + associated files.
    run_in_subprocess(associated_files_merge_create_stores, tmp_path, test_data_dir)
    base_paths = []
    simple_extra_paths = []
    complex_extra_paths = []
    for j in range(10):
        base_paths.append(tmp_path / f'base{j}.nc')
        simple_extra_paths.append(tmp_path / f'simple{j}.nc')
        complex_extra_paths.append(tmp_path / f'complex{j}.nc')

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
    run_in_subprocess(
        associated_files_merge_check_merged_stores,
        merged_base,
        merged_simple_associated,
        merged_complex_associated,
        test_data_dir,
    )


@pytest.mark.forked
def test_indexing(tmp_path: Path):
    # Seed both RNGs so the flight-ID sample and the trajectory field values
    # produced by `make_test_trajectory` are reproducible across runs.
    random.seed(0)
    np.random.seed(0)

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


def merged_store_indexing_create_stores(seeds, tmp_path, test_data_dir):
    Config.load(data_path_overrides=[test_data_dir])

    # 2. Create individual trajectory stores containing those flight IDs.
    paths = []
    for i in range(10):
        path = tmp_path / f'test{i}.nc'
        paths.append(path)
        with TrajectoryStore.create(base_file=path) as ts:
            for s in seeds[i * 100 : (i + 1) * 100]:
                ts.add(make_test_trajectory(50, s))


def merged_store_indexing_check_stores(seeds, merged_path, test_data_dir):
    Config.load(data_path_overrides=[test_data_dir])

    # 4. Open the merged store.
    with TrajectoryStore.open(base_file=merged_path) as ts_read:
        # 5. Look up trajectories by flight ID.
        for s in seeds:
            traj = ts_read.get_flight(s)
            assert traj is not None
            assert traj.flight_id == s


def test_merged_store_indexing(tmp_path, test_data_dir):
    # Same rationale as `test_indexing`: seed before sampling flight IDs and
    # generating trajectory fields so the run is reproducible.
    random.seed(1)
    np.random.seed(1)

    # 1. Create a unique set of flight IDs.
    ntrajs = 1000
    seeds = random.sample(range(10000, 100000), ntrajs)

    # 2. Create individual trajectory stores containing those flight IDs.
    run_in_subprocess(
        merged_store_indexing_create_stores, seeds, tmp_path, test_data_dir
    )
    paths = []
    for i in range(10):
        paths.append(tmp_path / f'test{i}.nc')

    # 3. Merge the stores.
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(input_stores=paths, output_store=merged_path)

    # 4. Open the merged store.
    run_in_subprocess(
        merged_store_indexing_check_stores, seeds, merged_path, test_data_dir
    )


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


@pytest.mark.forked
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


@pytest.mark.forked
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


@pytest.mark.forked
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


TEST_FIELDS = FieldSet(
    'test',
    per_trajectory_1=FieldMetadata(
        description='Per-trajectory field #1',
        dimensions=Dimensions(Dimension.TRAJECTORY),
    ),
    per_trajectory_2=FieldMetadata(
        description='Per-trajectory field #2',
        dimensions=Dimensions(Dimension.TRAJECTORY),
        field_type=np.int32,
    ),
    per_point_1=FieldMetadata(description='Per-point field #1'),
    per_point_2=FieldMetadata(description='Per-point field #2'),
)


def test_create_fixed_size_container():
    # Create a fixed-size Container with a specified field set.
    container_fixed = Container(npoints=10, fieldsets=['test'])
    assert len(container_fixed) == 10
    assert container_fixed._extensible is False
    assert container_fixed._capacity == 10


def test_append_to_fixed_size_container():
    # Create a fixed-size Container with a specified field set and try to
    # append to it.
    container_fixed = Container(npoints=10, fieldsets=['test'])
    with pytest.raises(ValueError):
        container_fixed.append(per_point_1=10, per_point_2=20)


def test_create_extensible_container():
    # Create an extensible Container with a specified field set.
    container_extensible = Container(fieldsets=['test'])
    assert len(container_extensible) == 0
    assert container_extensible._extensible is True
    assert container_extensible._capacity == 50


def test_append_to_container_by_keywords():
    # Create extensible container, append points by keyword, check data.
    container_extensible = Container(fieldsets=['test'])
    for i in range(1, 11):
        container_extensible.append(per_point_1=i * 10, per_point_2=i * 20)
    assert len(container_extensible) == 10
    assert container_extensible.per_point_1.tolist() == list(range(10, 110, 10))
    assert container_extensible.per_point_2.tolist() == list(range(20, 220, 20))


def test_append_to_container_by_keywords_bad():
    # Create extensible container, append points by keyword, check data.
    container_extensible = Container(fieldsets=['test'])
    with pytest.raises(ValueError):
        container_extensible.append(per_point_1=10)


def test_append_to_container_by_class():
    # Create extensible container, append points by single point container
    # class (enough to trigger capacity expansion), check data.
    container_extensible = Container(fieldsets=['test'])
    for i in range(1, 71):
        point = container_extensible.make_point()
        point.per_point_1 = i * 10
        point.per_point_2 = i * 20
        container_extensible.append(point)
    assert len(container_extensible) == 70
    assert container_extensible._capacity == 100
    assert container_extensible.per_point_1.tolist() == list(range(10, 710, 10))
    assert container_extensible.per_point_2.tolist() == list(range(20, 1420, 20))


def test_file_access_recorder(tmp_path: Path):
    # Create a FileAccessRecorder, record some accesses, check that they were
    # recorded correctly.
    file1 = tmp_path / 'file1.nc'
    file2 = tmp_path / 'file2.nc'
    file3 = tmp_path / 'file3.nc'
    sqlite_db = tmp_path / 'tmp.sqlite'
    for f in (file1, file2, file3):
        f.touch()

    with track_file_accesses():
        with open(file1):
            pass
        with open(file2):
            pass
        with open(file3):
            pass
        sqlite3.connect(sqlite_db).close()

    assert access_recorder.paths == [
        file1.resolve(),
        file2.resolve(),
        file3.resolve(),
        sqlite_db.resolve(),
    ]


def _assert_trajectories_equal(a: Trajectory, b: Trajectory) -> None:
    """Compare two trajectories field-for-field."""
    assert a._data_dictionary == b._data_dictionary
    assert a._fieldsets == b._fieldsets
    assert len(a) == len(b)
    for name in a._data_dictionary:
        va = a._data.get(name)
        vb = b._data.get(name)
        if isinstance(va, np.ndarray):
            assert isinstance(vb, np.ndarray)
            assert np.array_equal(va[: len(a)], vb[: len(b)])
        elif isinstance(va, SpeciesValues):
            assert isinstance(vb, SpeciesValues)
            assert set(va.keys()) == set(vb.keys())
            for sp in va.keys():
                sub_a = va[sp]
                sub_b = vb[sp]
                if isinstance(sub_a, np.ndarray):
                    assert np.array_equal(sub_a, sub_b)
                elif isinstance(sub_a, ThrustModeValues):
                    for tm in sub_a.keys():
                        assert sub_a[tm] == sub_b[tm]
                else:
                    assert sub_a == sub_b
        elif isinstance(va, ThrustModeValues):
            assert isinstance(vb, ThrustModeValues)
            for tm in va.keys():
                assert va[tm] == vb[tm]
        else:
            assert va == vb


def iter_range_full_check(path: Path):
    Config.load()
    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(1, 11):
            t = make_test_trajectory(i * 5, i)
            t.add_fields(SimpleExtras.random(i * 5))
            t.add_fields(ComplexExtras.random(i * 5))
            ts.add(t)

    with TrajectoryStore.open(base_file=path) as ts_read:
        expected = [ts_read[i] for i in range(len(ts_read))]
        ts_read._trajectories.clear()
        actual = list(ts_read.iter_range(0, len(ts_read)))
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)


def test_iter_range_full(tmp_path: Path):
    # Full-range iter_range on a non-merged store with simple + complex extras
    # should yield trajectories that match __getitem__ field-for-field.
    run_in_subprocess(iter_range_full_check, tmp_path / 'test.nc')


def iter_range_partial_check(path: Path):
    Config.load()
    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(10):
            ts.add(make_test_trajectory(10 + i, i))

    with TrajectoryStore.open(base_file=path) as ts_read:
        # Empty range should yield nothing.
        assert list(ts_read.iter_range(3, 3)) == []

        # Single-element range.
        expected = [ts_read[4]]
        ts_read._trajectories.clear()
        actual = list(ts_read.iter_range(4, 5))
        assert len(actual) == 1
        _assert_trajectories_equal(actual[0], expected[0])

        # Narrow middle range.
        expected = [ts_read[i] for i in range(3, 7)]
        ts_read._trajectories.clear()
        actual = list(ts_read.iter_range(3, 7))
        assert len(actual) == 4
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)

        # Range ending at the last index.
        expected = [ts_read[i] for i in range(7, 10)]
        ts_read._trajectories.clear()
        actual = list(ts_read.iter_range(7, 10))
        assert len(actual) == 3
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)

        # Out-of-range should raise.
        try:
            list(ts_read.iter_range(0, 11))
        except IndexError:
            pass
        else:
            raise AssertionError("Expected IndexError for stop > len")
        try:
            list(ts_read.iter_range(5, 3))
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for start > stop")


def test_iter_range_partial(tmp_path: Path):
    # Partial ranges (including length-1 and ranges starting at non-zero
    # offsets) should match __getitem__ for the corresponding indices.
    run_in_subprocess(iter_range_partial_check, tmp_path / 'test.nc')


def iter_range_merged_check(merged_path: Path):
    Config.load()
    with TrajectoryStore.open(base_file=merged_path) as ts_merged:
        assert len(ts_merged) == 8

        # Verify iter_range across the full merged store matches __getitem__.
        expected = [ts_merged[i] for i in range(len(ts_merged))]
        ts_merged._trajectories.clear()
        actual = list(ts_merged.iter_range(0, len(ts_merged)))
        assert len(actual) == 8
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)

        # Sub-range that straddles a file boundary (size_index = [2,4,6,8]).
        # Range [1, 5) spans files 0, 1, and 2 — critical bisect_right test.
        expected = [ts_merged[i] for i in range(1, 5)]
        ts_merged._trajectories.clear()
        actual = list(ts_merged.iter_range(1, 5))
        assert len(actual) == 4
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)

        # Sub-range aligned exactly with a file boundary on both ends.
        expected = [ts_merged[i] for i in range(2, 6)]
        ts_merged._trajectories.clear()
        actual = list(ts_merged.iter_range(2, 6))
        assert len(actual) == 4
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)

        # Sub-range lying entirely within a single file other than file 0.
        # size_index=[2,4,6,8] → [5, 6) is inside file 2.
        expected = [ts_merged[5]]
        ts_merged._trajectories.clear()
        actual = list(ts_merged.iter_range(5, 6))
        assert len(actual) == 1
        _assert_trajectories_equal(actual[0], expected[0])

        # Batch smaller than the file size to force multiple batches per
        # file and exercise the batching loop.
        expected = [ts_merged[i] for i in range(len(ts_merged))]
        ts_merged._trajectories.clear()
        actual = list(ts_merged.iter_range(0, len(ts_merged), batch_size=1))
        assert len(actual) == 8
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)


def iter_range_merged_merge(tmp_path: Path):
    Config.load()
    paths = [tmp_path / f'test_{i}.nc' for i in range(4)]
    merged_path = tmp_path / 'merged.aeic-store'
    TrajectoryStore.merge(input_stores=paths, output_store=merged_path)


def test_iter_range_merged_store(tmp_path: Path):
    # Critical bisect_right test: iter_range across file boundaries on a
    # merged store built from 4 files of 2 trajectories each (size_index
    # [2, 4, 6, 8]).
    run_in_subprocess(basic_merging_create_stores, tmp_path)
    run_in_subprocess(iter_range_merged_merge, tmp_path)
    run_in_subprocess(iter_range_merged_check, tmp_path / 'merged.aeic-store')


def iter_range_no_cache_pollution_check(path: Path):
    Config.load()
    with TrajectoryStore.create(base_file=path) as ts:
        for i in range(5):
            ts.add(make_test_trajectory(10, i))

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read._trajectories) == 0
        trajectories = list(ts_read.iter_range(0, len(ts_read)))
        assert len(trajectories) == 5
        # Cache should still be empty after iteration.
        assert len(ts_read._trajectories) == 0


def test_iter_range_no_cache_pollution(tmp_path: Path):
    # iter_range must not populate the LRU trajectory cache.
    run_in_subprocess(iter_range_no_cache_pollution_check, tmp_path / 'test.nc')


# --------------- iter_flight_ids ---------------


def iter_flight_ids_basic_check(path: Path):
    Config.load()
    seeds = list(range(10))
    with TrajectoryStore.create(base_file=path) as ts:
        for s in seeds:
            ts.add(make_test_trajectory(10 + s, s))

    with TrajectoryStore.open(base_file=path) as ts_read:
        # Request a subset of flight IDs.
        requested = [1, 3, 5, 7, 9]
        expected = [ts_read.get_flight(fid) for fid in requested]
        ts_read._trajectories.clear()
        actual = list(ts_read.iter_flight_ids(requested))
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)


def test_iter_flight_ids_basic(tmp_path: Path):
    # Subset lookup should return matching trajectories field-for-field.
    run_in_subprocess(iter_flight_ids_basic_check, tmp_path / 'test.nc')


def iter_flight_ids_missing_check(path: Path):
    Config.load()
    seeds = list(range(5))
    with TrajectoryStore.create(base_file=path) as ts:
        for s in seeds:
            ts.add(make_test_trajectory(10, s))

    with TrajectoryStore.open(base_file=path) as ts_read:
        # Mix of existing (0, 2, 4) and non-existing (99, 100) flight IDs.
        requested = [0, 2, 4, 99, 100]
        expected = [ts_read.get_flight(fid) for fid in [0, 2, 4]]
        ts_read._trajectories.clear()
        actual = list(ts_read.iter_flight_ids(requested))
        assert len(actual) == 3
        for a, e in zip(actual, expected):
            _assert_trajectories_equal(a, e)


def test_iter_flight_ids_missing(tmp_path: Path):
    # Non-existent flight IDs should be silently skipped.
    run_in_subprocess(iter_flight_ids_missing_check, tmp_path / 'test.nc')


def iter_flight_ids_empty_check(path: Path):
    Config.load()
    with TrajectoryStore.create(base_file=path) as ts:
        ts.add(make_test_trajectory(10, 0))

    with TrajectoryStore.open(base_file=path) as ts_read:
        actual = list(ts_read.iter_flight_ids([]))
        assert len(actual) == 0


def test_iter_flight_ids_empty(tmp_path: Path):
    # Empty flight ID list should yield nothing.
    run_in_subprocess(iter_flight_ids_empty_check, tmp_path / 'test.nc')


def iter_flight_ids_no_cache_pollution_check(path: Path):
    Config.load()
    with TrajectoryStore.create(base_file=path) as ts:
        for s in range(5):
            ts.add(make_test_trajectory(10, s))

    with TrajectoryStore.open(base_file=path) as ts_read:
        assert len(ts_read._trajectories) == 0
        trajectories = list(ts_read.iter_flight_ids([0, 1, 2, 3, 4]))
        assert len(trajectories) == 5
        # Cache should still be empty after iteration.
        assert len(ts_read._trajectories) == 0


def test_iter_flight_ids_no_cache_pollution(tmp_path: Path):
    # iter_flight_ids must not populate the LRU trajectory cache.
    run_in_subprocess(iter_flight_ids_no_cache_pollution_check, tmp_path / 'test.nc')
