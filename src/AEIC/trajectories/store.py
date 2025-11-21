import bisect
import hashlib
import itertools
import json
import os
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from cachetools import LRUCache
from netCDF4 import Dataset, Dimension, Group, VLType

from .field_sets import FieldSet, HasFieldSets
from .trajectory import BASE_FIELDSET_NAME, Trajectory

# Python doesn't have a simple way of saying "anything that's acceptable as a
# filesystem path", so define a simple type alias instead.

PathType = str | Path


# Types used in associated NetCDF file creation functions.

AssociatedFileCreate = tuple[PathType, list[str]]
AssociatedFileOpen = PathType
AssociatedFiles = list[AssociatedFileCreate] | list[AssociatedFileOpen]


# NOTE: Whenever a NetCDF4 Dataset is opened, the keepweakref parameter must be
# set to avoid the segmentation fault issues described at
# https://github.com/Unidata/netcdf4-python/issues/1444


class AssociatedFileCreateFn(Protocol):
    """The type of functions used to create associated NetCDF files.

    The function is passed the input trajectory plus any additional arguments
    supplied to the `create_associated` method of `TrajectoryStore`, and must
    return an object implementing the HasFieldSets protocol, i.e., an object
    that has a `FIELD_SETS` class attribute giving the FieldSets that should be
    used to save the result data.

    """

    def __call__(self, traj: Trajectory, *args, **kwargs) -> HasFieldSets: ...


class _TrajectoryStoreIterator(Iterator):
    """Private iterator class for TrajectoryStore."""

    def __init__(self, store):
        self._store = store
        self._index = 0

    def __next__(self):
        if self._index < len(self._store):
            item = self._store[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


class TrajectoryCache(LRUCache[int, Trajectory]):
    """Specialization of LRUCache class for trajectory storage.

    This is used to handle the case of in-memory trajectory stores that are not
    connected to a NetCDF file: in that case, we want to prevent evictions from
    the trajectory cache, since there is no file to which we can save evicted
    items. Instead, we just throw an exception if the trajectory cache would
    exceed its assigned size.
    """

    class EvictionOccurred(RuntimeError):
        """Exception raised when an eviction occurs in the trajectory cache."""

        pass

    def __init__(self, *args, **kwargs):
        """Initialize a trajectory cache with LRU eviction."""
        super().__init__(*args, **kwargs)
        self.exception_on_eviction = False

    def popitem(self):
        """Pop item from cache and return it."""
        if self.exception_on_eviction:
            raise self.EvictionOccurred()
        key, value = super().popitem()
        return key, value


class TrajectoryStore:
    """Class representing a set of trajectories stored in NetCDF files.

    Fundamentally, a `TrajectoryStore` is an append-only collection of
    `Trajectory` objects that can be stored in and retrieved from NetCDF files.
    In the simplest file-based use case, a `TrajectoryStore` stores plain
    trajectory data in a single NetCDF file. However, the class offers
    additional functionality that allows additional data fields to be combined
    with trajectory data, either in the same ("base") NetCDF file, or in
    additional ("associated") NetCDF files.

    The intention here is to support a number of different use cases for
    storing trajectory and associated (normally emissions) data.

    Trajectory stores are *not* thread-safe. A program may create and use
    TrajectoryStore values from a single thread only. This single-threaded
    restriction applies both to read-only access and mutation of stores.

    Data stored in a `TrajectoryStore` is divided into "field sets"
    (represented by the `FieldSet` class from the
    `AEIC.trajectories.field_sets` package). A field set is a collection of
    data and metadata fields that are part of a trajectory or data that lives
    alongside a trajectory (emissions data of one sort or another, for
    example). "Data fields" in field sets have values for each point along a
    trajectory: the length of the data values in each of these fields must
    match the length of the trajectory. "Metadata fields" in field sets are
    per-trajectory values: there is one value of each of each of these fields
    for each trajectory. Each field in a field set has a name, a data type and
    associated information used for serialization to and from NetCDF files.

    A `TrajectoryStore` always contains the "base" field set, which holds the
    basic trajectory data (defined as `BASE_FIELDS` in the
    `AEIC.trajectories.trajectory` package). Additional field sets can be
    stored in a `TrajectoryStore` as needed.

    The `TrajectoryStore` class supports three access modes: CREATE, READ and
    APPEND. In addition, it is possible to create additional associated NetCDF
    files associated with an existing `TrajectoryStore` using the
    `create_associated` method.

    A `TrajectoryStore` may be:
     - created purely in-memory, or
     - it may be connected to a single NetCDF file (in which data from all
       field sets will be stored), or
     - it may be connected to a base NetCDF file (holding the "base" field set
       of trajectory data and zero or more additional field sets) and one or
       more associated NetCDF files (storing other non-base field sets).

    Trajectories are managed by a `TrajectoryStore` using an LRU memory cache
    with a (user configurable) fixed size. When a `TrajectoryStore` is
    associated with NetCDF files, entries from the trajectory cache can be
    evicted, since they are stored in external files. An in-memory
    `TrajectoryStore` does not have this flexibility, and if the stored
    trajectories overflow the cache size, requiring an eviction, a
    `TrajectoryCache.EvictionOccurred` exception is raised.

    TODO: Continue this and work out the best way to integrate it into the
    Sphinx docs. This could get quite long.

    """

    class FileMode(str, Enum):
        READ = 'r'
        CREATE = 'w'
        APPEND = 'a'

    @dataclass
    class NcFiles:
        """Internal class used to store information about NetCDF files
        associated with field sets in a TrajectoryStore.

        """

        path: list[Path]
        fieldsets: set[str]
        dataset: list[Dataset]
        traj_dim: list[Dimension]
        groups: dict[str, list[Group]]
        size_index: list[int] | None
        title: str | None = None
        comment: str | None = None
        history: str | None = None
        source: str | None = None
        created: datetime | None = None

    active_in_thread: int | None = None

    # Allowed constructor arguments by mode:
    #
    # (' ' = not allowed, X = required, ? = optional)
    #
    #  R W A
    #  -----------------------
    #    ?    title, comment, history, source: Permitted only for creating a
    #           new NetCDF file.
    #  X ? X  base_file
    #  X X X  mode
    #  ?      override
    #  ? ? ?  *associated_files: must be str for APPEND and READ (field sets
    #           in each file are fixed already) and must be tuple[str,
    #           list[str]] for CREATE (allotment of field sets to associated
    #           files must be specified).
    def __init__(
        self,
        *,
        base_file: str | None = None,
        mode: FileMode = FileMode.READ,
        override: bool | None = None,
        force_fieldset_matches: bool | None = None,
        cache_size_mb: int = 2048,
        associated_files: AssociatedFiles | None = None,
        title: str | None = None,
        comment: str | None = None,
        history: str | None = None,
        source: str | None = None,
    ):
        """Initialize a TrajectoryStore with various file access modes.

        Parameter
        ----------
        base_file : str | None, optional
            Path to the base NetCDF file. Required in READ and APPEND modes.
        mode : TrajectoryStore.FileMode, optional
            File access mode. One of:
            - READ: Open an existing NetCDF file for read-only access.
            - CREATE: Create a new NetCDF file for writing.
            - APPEND: Open an existing NetCDF file for appending new
                trajectories.
            Default is READ.
        override : bool | None, optional
            If True in READ mode, FieldSets from associated files will override
            any FieldSets of the same name in the base file. Default is False.
        force_fieldset_matches : bool | None, optional
            If True in READ mode, FieldSets from NetCDF files that do not match
            the corresponding FieldSet in the registry will be accepted with a
            warning. Default is False. There are no guarantees that things will
            work if this is set to True!
        associated_files : list[PathType] | list[tuple[PathType, list[str]]] | None
            Paths to associated NetCDF files containing additional data or
            metadata fields. Each associated file may be specified as a string
            path or as a tuple of the form (path, [fieldset_names]) where
            fieldset_names is a list of names of FieldSets to load from the
            associated file.
        title : str | None, optional
            Title for the base NetCDF file. (Global NetCDF attribute.)
        comment : str | None, optional
            Comment for the base NetCDF file. (Global NetCDF attribute.)
        history : str | None, optional
            History for the base NetCDF file. (Global NetCDF attribute.)
        source : str | None, optional
            Source for the base NetCDF file. (Global NetCDF attribute.)

        Raises
        ------
        ValueError
            If input arguments are inconsistent with the specified file mode.

        """

        # Check thread activity: must be single-threaded.
        if TrajectoryStore.active_in_thread is not None:
            if TrajectoryStore.active_in_thread != threading.get_ident():
                raise RuntimeError(
                    'TrajectoryStore: multiple TrajectoryStore instances '
                    'active in different threads simultaneously.'
                )
        else:
            TrajectoryStore.active_in_thread = threading.get_ident()

        # File access mode for a TrajectoryStore is fixed: if you need to
        # switch mode, close and reopen the store.
        self.mode = mode

        # Indexable trajectory stores have trajectories that have a numeric
        # flight ID from the mission database. For now, we don't know whether
        # that's the case: when we open a store, we'll probe a trajectory to
        # see if it has a flight ID anbd set this accordingly; similarly, if
        # we're creating creating a new store, we'll decide on indexability
        # when we see the first trajectory added to the store.
        #
        # The index information is stored in a special NetCDF group that we
        # access separately from the main NcFiles mechanism.
        self.indexable = None
        self.index_group: Group | None = None
        self.index_dataset: Dataset | None = None

        # The index can become stale when new items are added to the store. We
        # reindex lazily, either when an index lookup is performed or on a sync
        # or close.
        self.index_stale = False

        # Check input arguments based on file mode.
        global_attributes_ok = mode == self.FileMode.CREATE
        if not global_attributes_ok and (
            title is not None
            or comment is not None
            or history is not None
            or source is not None
        ):
            raise ValueError(
                'global attributes (title, comment, history, '
                'source) may only be specified in CREATE mode'
            )
        self.global_attributes = {}
        if title is not None:
            self.global_attributes['title'] = title
        if comment is not None:
            self.global_attributes['comment'] = comment
        if history is not None:
            self.global_attributes['history'] = history
        if source is not None:
            self.global_attributes['source'] = source

        # Setting base_file=None in CREATE mode creates an empty TrajectoryStore.
        # We can switch to a file-backed store later using the save method, but
        # in the meantime, the trajectory cache is limited in size and cannot
        # evict any items. Adding too many trajectories to an in-memory
        # TrajectoryStore will result in a TrajectoryCache.EvictionOccurred
        # exception.
        base_file_req = mode != self.FileMode.CREATE
        if base_file is None and base_file_req:
            raise ValueError(f'base_file required in file mode {mode}')
        self.base_file = base_file
        check_paths: list[PathType] = []
        if base_file is not None:
            check_paths.append(base_file)

        override_ok = mode == self.FileMode.READ
        if override is not None and not override_ok:
            raise ValueError('override may only be specified in READ mode')
        self.override = override if override is not None else False

        force_fieldset_matches_ok = mode == self.FileMode.READ
        if force_fieldset_matches is not None and not force_fieldset_matches_ok:
            raise ValueError(
                'force_fieldset_matches may only be specified in READ mode'
            )
        self.force_fieldset_matches = (
            force_fieldset_matches if force_fieldset_matches is not None else False
        )

        if associated_files is None:
            associated_files = []
        if mode in (self.FileMode.APPEND, self.FileMode.READ):
            if any(not isinstance(f, PathType) for f in associated_files):
                raise ValueError(
                    'associated_files must be paths in APPEND and READ modes'
                )
            for f in associated_files:
                assert isinstance(f, PathType)
                check_paths.append(f)
        if mode == self.FileMode.CREATE:
            if any(not valid_associated_file_tuple(f) for f in associated_files):
                raise ValueError(
                    'associated_files must be tuple[PathType, list[str]] in CREATE mode'
                )
            for f in associated_files:
                assert isinstance(f, tuple)
                assert isinstance(f[0], PathType)
                check_paths.append(f[0])
        self.associated_files = associated_files
        self.associated_fieldsets: set[str] = set()
        if mode == self.FileMode.CREATE:
            self._calc_associated_fieldsets()

        # Check that file paths exist (for reading) or do not exist (for
        # creation).
        self._check_file_paths(check_paths, mode)

        # Trajectories are stored as an LRU cache indexed by the index of the
        # trajectory. (This is done to handle cases where the trajectory store
        # is very large and cannot be held in memory all at once.)
        #
        # A store created with base_file=None is in-memory only. We can switch to
        # a file-backed store using the save method, but in the meantime, the
        # trajectory cache cannot evict any entries. The custom TrajectoryCache
        # class has a flag to raise an exception on eviction for this use case.
        #
        # NOTE: The LRUCache on which TrajectoryCache is based is not
        # thread-safe, but that's OK because TrajectoryStore isn't thread-safe
        # anyway, because of the threading problems in the underlying NetCDF
        # libraries. Our use cases for AEIC do not require TrajectoryStore to
        # be thread-safe, so we're all good.
        self._trajectories = TrajectoryCache(
            cache_size_mb * 1024 * 1024,
            getsizeof=lambda t: t.nbytes,  # type: ignore
        )
        if self.base_file is None:
            self._trajectories.exception_on_eviction = True

        # List of NetCDF file information structures and mapping from field set
        # names to NetCDF file information. The first entry in self._nc_files is
        # the base NetCDF file; any others are associated files.
        self._nc_files: list[TrajectoryStore.NcFiles] = []
        self._nc: dict[str, TrajectoryStore.NcFiles] = {}

        # If we're opening in CREATE mode, set up for file creation. We can't
        # create the file (or files) until we know what field sets are
        # involved, and we only get to see that when a `Trajectory` is added to
        # the store.
        self._file_creation_pending = (
            mode == self.FileMode.CREATE and base_file is not None
        )

        # Next trajectory index to assign. This is the length of the trajectory
        # dimension. It gets set to the correct value for APPEND mode when we
        # open the existing file.
        self._next_index = 0

        # Whether writing is enabled (CREATE or APPEND mode).
        self._write_enabled = mode in (self.FileMode.CREATE, self.FileMode.APPEND)

        # Open an existing file or files.
        if mode in (self.FileMode.READ, self.FileMode.APPEND):
            if self.merged_store:
                self._open_merged()
            else:
                self._open()

    @classmethod
    def create(cls, *args, **kwargs) -> 'TrajectoryStore':
        """Create a new TrajectoryStore."""
        return cls(mode=TrajectoryStore.FileMode.CREATE, *args, **kwargs)

    @classmethod
    def open(cls, *args, **kwargs) -> 'TrajectoryStore':
        """Open an existing TrajectoryStore read-only."""
        return cls(mode=TrajectoryStore.FileMode.READ, *args, **kwargs)

    @classmethod
    def append(cls, *args, **kwargs) -> 'TrajectoryStore':
        """Open an existing TrajectoryStore for appending."""
        return cls(mode=TrajectoryStore.FileMode.APPEND, *args, **kwargs)

    @property
    def files(self) -> list[NcFiles]:
        """NetCDF file information associated with the trajectory store.

        The base NetCDF file is the first entry in the list; any associated
        files follow.
        """
        return self._nc_files

    def _calc_associated_fieldsets(self) -> None:
        # This is slightly unwieldy because of Python's restrictions on
        # using isinstance with parameterized generics.
        assert isinstance(self.associated_files, list)
        self.associated_fieldsets = set()
        for t in self.associated_files:
            assert isinstance(t, tuple)
            for f in t[1]:
                self.associated_fieldsets.add(f)

    def save(
        self, base_file: PathType, associated_files: AssociatedFiles | None = None
    ):
        """Create NetCDF files for a TrajectoryStore currently not linked to
        one.
        """
        if self.nc_linked:
            raise RuntimeError(
                'TrajectoryStore.save: TrajectoryStore is already linked to a '
                'NetCDF file'
            )

        # Start list of file paths we need to check.
        check_paths: list[PathType] = [base_file]

        # If associated files are provided, check them in the same way as in
        # the constructor.
        if associated_files is None:
            associated_files = []
        if any(not valid_associated_file_tuple(f) for f in associated_files):
            raise ValueError(
                'associated_files must be tuple[PathType, list[str]] in save'
            )
        for f in associated_files:
            assert isinstance(f, tuple)
            assert isinstance(f[0], PathType)
            check_paths.append(f[0])
        self.associated_files = associated_files
        self._calc_associated_fieldsets()

        # Check that file paths do not already exist.
        self._check_file_paths(check_paths, self.FileMode.CREATE)

        # Remember how many trajectories we have to save. (Extracted here
        # because the logic for this will be different once we've opened NetCDF
        # files.)
        trajectories_to_save = len(self)

        # The workflow here is essentially the same as in _create, but we need
        # to enable writing first.
        self.base_file = base_file
        self._write_enabled = True
        self._create()

        # Write trajectories to the newly created files.
        for i in range(trajectories_to_save):
            self._write_trajectory(i)

        # Once the files have been created successfully and the existing data
        # persisted, we can allow evictions from the trajectory cache.
        self._trajectories.exception_on_eviction = False

    def create_associated(
        self,
        associated_file: PathType,
        fieldsets: list[str],
        mapping_function: AssociatedFileCreateFn,
        *args,
        **kwargs,
    ) -> None:
        """Create an associated NetCDF file for additional field sets.

        This maps a function over all trajectories in the TrajectoryStore to
        create new associated data values, which are immediately written to an
        associated NetCDF file.
        """
        # Check that file doesn't already exist.
        p = Path(associated_file)
        if not p.parent.exists():
            raise ValueError(
                f'Parent directory of associated NetCDF file "{p}" does not exist'
            )
        if p.exists():
            raise ValueError(f'Associated NetCDF file "{p}" already exists')

        # Check field sets are known.
        for fs_name in fieldsets:
            if not FieldSet.known(fs_name):
                raise ValueError(
                    f'FieldSet with name "{fs_name}" not found in FieldSet registry'
                )

        # Create file.
        nc_info = self._create_nc_file(
            associated_file,
            set(fieldsets),
            save=False,
            associated_name=self._nc_files[0].path[0],
            associated_hash=self._nc_files[0].dataset[0].id_hash,
        )

        for i in range(len(self)):
            # Create associated data value.
            associated_data = mapping_function(self[i], *args, **kwargs)

            # These are the only checks we're going to do here: the call to
            # _write_data will fail if any of the fields from the field sets
            # are missing or of the wrong type.
            if not hasattr(associated_data, 'FIELD_SETS'):
                raise ValueError(
                    'Result of mapping_function must implement HasFieldSets protocol'
                )
            assoc_field_sets = associated_data.FIELD_SETS
            assert isinstance(assoc_field_sets, list)
            if len(assoc_field_sets) != len(fieldsets):
                raise ValueError(
                    'Result of mapping_function must contain all field sets '
                    'specified for associated NetCDF file'
                )
            if set(fs.fieldset_name for fs in assoc_field_sets) != set(fieldsets):
                raise ValueError(
                    'Field sets in mapping_function result must match field sets '
                    'specified for associated NetCDF file'
                )

            # Write data for field set to associated file.
            self._write_data(
                single_nc_file=nc_info,
                index=i,
                fieldsets=fieldsets,
                data=associated_data,
            )

        nc_info.dataset[0].close()

    @property
    def nc_linked(self) -> bool:
        """Is a NetCDF file (or files) associated with the trajectory store?"""
        return len(self._nc_files) != 0

    def close(self):
        """Close any open NetCDF files associated with the trajectory store."""
        if self.indexable and self.index_stale:
            self._reindex()
        for nc in self._nc_files:
            for ds in nc.dataset:
                ds.close()
        if self.index_dataset is not None:
            self.index_dataset.close()
        self._nc.clear()
        self._nc_files.clear()
        self.index_dataset = None
        self.index_group = None

    def sync(self):
        """Synchronize any pending writes to the NetCDF file or files.

        Note that this does not necessarily make the NetCDF files readable by
        another application because of NetCDF4's finalization behavior. To
        ensure complete finalization, call close() instead.
        """
        if not self._write_enabled:
            raise RuntimeError('Cannot sync TrajectoryStore not opened in write mode')
        if self.indexable and self.index_stale:
            self._reindex()
        for nc in self._nc_files:
            nc.dataset[0].sync()
        if self.index_dataset is not None:
            self.index_dataset.sync()

    def __len__(self):
        """Count number of trajectories in store."""
        if self.nc_linked:
            # Normally, use the base field set for length calculations.
            # Sometimes we need the length of a store that doesn't contain the
            # base field set (this happens when merging associated files, for
            # example), so we pick another field set that we do have.
            check_fs = BASE_FIELDSET_NAME
            if check_fs not in self._nc:
                check_fs = next(iter(self._nc))
            return sum(len(d) for d in self._nc[check_fs].traj_dim)
        return len(self._trajectories)

    def __getitem__(self, idx) -> Trajectory:
        """Retrieve a trajectory by numeric index (in order of addition)."""
        if idx in self._trajectories:
            return self._trajectories[idx]
        if self.nc_linked:
            self._load_trajectory(idx)
        if idx not in self._trajectories:
            raise IndexError('Trajectory index out of range')
        return self._trajectories[idx]

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterator over trajectories in store in index order."""
        return _TrajectoryStoreIterator(self)

    def add(self, trajectory: Trajectory) -> int:
        """Add a trajectory to the store and return its index."""
        if not self._write_enabled:
            raise RuntimeError(
                'Cannot add trajectory to TrajectoryStore not opened in write mode'
            )
        if len(self._trajectories) > 0:
            proto = next(iter(self._trajectories.values()))
            if hash(trajectory.data_dictionary) != hash(proto.data_dictionary):
                raise ValueError(
                    'All trajectories in a TrajectoryStore must have the same '
                    'data fields'
                )

        # Decide on whether or not we can index the store.
        has_flight_id = (
            hasattr(trajectory, 'flight_id') and trajectory.flight_id is not None
        )
        if self.indexable is not None and has_flight_id != self.indexable:
            raise ValueError(
                'All trajectories in an indexable TrajectoryStore must have '
                'flight_id field, and non-indexable stores must not have it'
            )
        if self.indexable is None:
            self.indexable = has_flight_id

        # Maintain count of trajectories in store for indexing.
        saved_index = self._next_index
        self._trajectories[saved_index] = trajectory
        self._next_index += 1

        # If this is the first trajectory added to the store, we might need to
        # create the NetCDF files.
        if self._file_creation_pending:
            self._create()
            self._file_creation_pending = False

        # Write the trajectory data to the output NetCDF file.
        self._write_trajectory(saved_index)

        if self.indexable:
            self.index_stale = True

        return saved_index

    @staticmethod
    def merge(
        output_store: PathType,
        input_stores: list[PathType] | None = None,
        input_stores_pattern: PathType | None = None,
        input_stores_index_range: tuple[int, int] | None = None,
        title: str | None = None,
        comment: str | None = None,
        history: str | None = None,
        source: str | None = None,
    ) -> None:
        # Parameter checks and normalization.
        if input_stores is not None and input_stores_pattern is not None:
            raise ValueError(
                'Specify either input_stores or input_stores_pattern, not both'
            )
        if input_stores_pattern is not None and input_stores_index_range is None:
            raise ValueError(
                'When specifying input_stores_pattern, '
                'input_stores_index_range must also be specified'
            )
        if input_stores_pattern is not None:
            assert input_stores_index_range is not None
            input_stores = [
                Path(str(input_stores_pattern).format(index=i))
                for i in range(
                    input_stores_index_range[0], input_stores_index_range[1] + 1
                )
            ]
        assert input_stores is not None
        for p in input_stores:
            if not Path(p).exists():
                raise ValueError(f'Input TrajectoryStore file "{p}" does not exist')
            if Path(p).suffix != '.nc':
                raise ValueError(f'Merge input "{p}" is not a NetCDF file')
        if not str(output_store).endswith('.aeic-store'):
            raise ValueError(
                'Output TrajectoryStore file must have ".aeic-store" extension'
            )
        if Path(output_store).exists():
            raise ValueError(
                f'Output TrajectoryStore file "{output_store}" already exists'
            )

        # Create output directory.
        os.mkdir(output_store)

        # Collect metadata and check that the field sets match.
        store_data = []
        fieldset_names: set[str] | None = None
        index_groups = []
        for input_store in input_stores:
            p = Path(input_store)
            ts = TrajectoryStore.open(base_file=p)
            if fieldset_names is None:
                fieldset_names = set(ts._nc.keys())
            if fieldset_names != set(ts._nc.keys()):
                raise ValueError(
                    'All input TrajectoryStore files must have the same field sets'
                )
            store_data.append((p.name, len(ts)))
            index_groups.append(ts.index_group)

        # Check indexability consistency.
        indexable = all(g is not None for g in index_groups)
        if indexable != any(g is not None for g in index_groups):
            raise ValueError('Either all or none of the input stores must be indexable')

        # Move input stores to output directory.
        for input_store in input_stores:
            p = Path(input_store)
            dest = Path(output_store) / p.name
            os.rename(p, dest)

        # Create merged index.
        if indexable:
            index_dataset = Dataset(
                Path(output_store) / '_index.nc', 'w', keepweakref=True
            )
            index_dataset.createDimension('trajectory', None)
            index_group = index_dataset.createGroup('_index')
            index_group.createVariable('flight_id', np.int64, ('trajectory',))
            index_group.createVariable('trajectory_index', np.int64, ('trajectory',))
            flight_ids = []
            trajectory_indexes = []
            index_offset = 0
            for input_store in input_stores:
                ts = TrajectoryStore.open(
                    base_file=Path(output_store) / Path(input_store).name
                )
                assert ts.index_group is not None
                vs = ts.index_group.variables
                flight_ids += list(vs['flight_id'][:])
                trajectory_indexes += [
                    idx + index_offset for idx in vs['trajectory_index'][:]
                ]
                index_offset += len(ts)
            id_pairs = sorted(zip(trajectory_indexes, flight_ids), key=lambda x: x[1])
            index_group.variables['flight_id'][:] = [id for _, id in id_pairs]
            index_group.variables['trajectory_index'][:] = [idx for idx, _ in id_pairs]

        # Write metadata JSON file to output directory.
        data = dict(stores=store_data, created=datetime.now(tz=UTC).isoformat())
        if title is not None:
            data['title'] = title
        if comment is not None:
            data['comment'] = comment
        if history is not None:
            data['history'] = history
        if source is not None:
            data['source'] = source
        with open(Path(output_store) / 'metadata.json', 'w') as f:
            json.dump(data, f)

    def _load_trajectory(self, index: int) -> None:
        """Load a trajectory at the given index from the NetCDF file(s)."""
        data = {}
        npoints: int | None = None

        # Handle field sets one by one.
        for fs_name in self._nc:
            # Look up the field set in the field set registry.
            fs = FieldSet.from_registry(fs_name)

            # File information for the field set and the NetCDF group for
            # variables in the field set.
            nc_files = self._nc[fs_name]
            file_index = 0
            group_index = index
            if nc_files.size_index is not None:
                file_index = bisect.bisect_left(nc_files.size_index, index + 1)
                if file_index >= len(nc_files.size_index):
                    return
                group_index = index - nc_files.size_index[file_index]
            group = nc_files.groups[fs_name][file_index]

            # Write per-point data as variable-length arrays.
            for name, field in fs.items():
                if name not in group.variables:
                    raise ValueError(
                        f'Data field "{name}" does not exist in NetCDF file'
                    )
                data[name] = group.variables[name][group_index]
                if not field.metadata and npoints is None:
                    npoints = len(data[name])

        # Construct the return trajectory.
        assert npoints is not None
        traj = Trajectory(npoints=npoints)
        for fs_name in self._nc:
            if fs_name != BASE_FIELDSET_NAME:
                traj.add_fields(FieldSet.from_registry(fs_name))
        for k, v in data.items():
            setattr(traj, k, v)

        self._trajectories[index] = traj

    def _write_trajectory(self, index: int) -> None:
        """Write a trajectory at the given index to the NetCDF file(s)."""

        # Look up the trajectory to write and write to files.
        self._write_data(traj=self._trajectories[index], index=index)

    def _write_data(
        self,
        *,
        index: int,
        traj: Trajectory | None = None,
        data: Any | None = None,
        single_nc_file: 'TrajectoryStore.NcFiles | None' = None,
        fieldsets: list[str] | None = None,
    ) -> None:
        """Write a trajectory at the given index to the NetCDF file(s)."""

        # NOTE: I thought about implementing some sort of batching here, but I
        # think we might be able to rely on the internal caching that the
        # NetCDF4 library does.

        # Check: should be one or the other.
        if traj is None and data is None:
            raise ValueError('Either traj or data must be provided')

        # In the normal case, we handle all the field sets in the store. When
        # creating an associated NetCDF file, we only handle the specified
        # field set.
        if fieldsets is None:
            fieldsets = list(self._nc.keys())

        # Handle field sets one by one.
        for fs_name in fieldsets:
            # File information for the field set and the NetCDF group for
            # variables in the field set.
            fs = FieldSet.from_registry(fs_name)
            nc_file = single_nc_file or self._nc[fs_name]
            group = nc_file.groups[fs_name][0]

            # Write per-point data as variable-length arrays.
            for name in group.variables:
                field = fs[name]
                if not field.metadata:
                    val = None
                    if traj is not None:
                        val = traj.data[name]
                        if val is None and hasattr(traj, name):
                            val = getattr(traj, name)
                    if data is not None:
                        val = getattr(data, name)
                    if val is None:
                        raise ValueError(
                            f'Per-point data field "{name}" is None at index {index}'
                        )
                    group.variables[name][index] = val
                else:
                    val = None
                    if traj is not None:
                        val = traj.metadata[name]
                        if val is None and hasattr(traj, name):
                            val = getattr(traj, name)
                    if data is not None:
                        val = getattr(data, name)
                    if val is None and field.required:
                        raise ValueError(
                            f'Metadata field "{name}" is None at index {index}'
                        )
                    if val is not None:
                        group.variables[name][index] = val

    def _open(self):
        """Open an existing NetCDF file (or files) for reading/appending
        trajectories.

        There is one NetCDF group per field set, and more than one field set
        may be stored in each NetCDF file.
        """

        # Open the NetCDF4 dataset from base file.
        assert self.base_file is not None
        base_nc_file = self._open_nc_file(self.base_file)
        self.global_attributes = {}
        if base_nc_file.title is not None:
            self.global_attributes['title'] = base_nc_file.title
        if base_nc_file.comment is not None:
            self.global_attributes['comment'] = base_nc_file.comment
        if base_nc_file.history is not None:
            self.global_attributes['history'] = base_nc_file.history
        if base_nc_file.source is not None:
            self.global_attributes['source'] = base_nc_file.source
        self.global_attributes['created'] = base_nc_file.created

        # Check that the field sets in the base NetCDF file exist in the field
        # set registry. There is a 1-to-1 relation between NetCDF groups and
        # field sets.
        for fs_name in base_nc_file.fieldsets:
            if not FieldSet.known(fs_name):
                raise ValueError(
                    f'FieldSet with name "{fs_name}" found in base NetCDF file '
                    f'not found in FieldSet registry'
                )

            # Field set from registry.
            fs = FieldSet.from_registry(fs_name)

            # Field set constructed from NetCDF group for comparison.
            netcdf_fs = FieldSet.from_netcdf_group(base_nc_file.groups[fs_name][0])

            # Now we can check that the fields in the field set exist in the
            # relevant NetCDF group and that they have the right types and
            # shapes.
            if hash(fs) != hash(netcdf_fs):
                raise ValueError(
                    f'FieldSet with name "{fs_name}" in base NetCDF file is '
                    f'incompatible with FieldSet in registry'
                )

        # Save file information under fieldset names in _nc dictionary.
        self._nc_files.append(base_nc_file)
        for fs_name in base_nc_file.fieldsets:
            self._nc[fs_name] = base_nc_file

        # Set the next trajectory index for APPEND mode.
        if self.mode == self.FileMode.APPEND:
            self._next_index = len(base_nc_file.traj_dim[0])

        # Set up index information.
        if '_index' in base_nc_file.dataset[0].groups:
            self.index_group = base_nc_file.dataset[0].groups['_index']
            self.indexable = True

        # Open any associated NetCDF files.
        for name in self.associated_files:
            assert isinstance(name, PathType)
            assocated_file = self._open_nc_file(name, check_associated=base_nc_file)
            self._nc_files.append(assocated_file)
            for fs_name in assocated_file.fieldsets:
                if fs_name not in self._nc or self.override:
                    self._nc[fs_name] = assocated_file
                else:
                    warnings.warn(
                        f'FieldSet with name "{fs_name}" found in associated '
                        f'NetCDF file "{name}" already exists in base file',
                        RuntimeWarning,
                    )

    def _open_merged(self):
        """Open an existing merged store file for reading trajectories."""

        # Open the NetCDF4 dataset from base file.
        assert self.base_file is not None
        base_nc_file = self._open_merged_store(self.base_file)
        self.global_attributes = {}
        if base_nc_file.title is not None:
            self.global_attributes['title'] = base_nc_file.title
        if base_nc_file.comment is not None:
            self.global_attributes['comment'] = base_nc_file.comment
        if base_nc_file.history is not None:
            self.global_attributes['history'] = base_nc_file.history
        if base_nc_file.source is not None:
            self.global_attributes['source'] = base_nc_file.source
        self.global_attributes['created'] = base_nc_file.created

        # Check that the field sets in the base NetCDF file exist in the field
        # set registry. There is a 1-to-1 relation between NetCDF groups and
        # field sets.
        for fs_name in base_nc_file.fieldsets:
            if not FieldSet.known(fs_name):
                raise ValueError(
                    f'FieldSet with name "{fs_name}" found in base NetCDF file '
                    f'not found in FieldSet registry'
                )

            # Field set from registry.
            fs = FieldSet.from_registry(fs_name)

            for g in base_nc_file.groups[fs_name]:
                # Field set constructed from NetCDF group for comparison.
                netcdf_fs = FieldSet.from_netcdf_group(g)

                # Now we can check that the fields in the field set exist in
                # the relevant NetCDF group and that they have the right types
                # and shapes.
                if hash(fs) != hash(netcdf_fs):
                    raise ValueError(
                        f'FieldSet with name "{fs_name}" in base NetCDF file is '
                        f'incompatible with FieldSet in registry'
                    )

        # Save file information under fieldset names in _nc dictionary.
        self._nc_files.append(base_nc_file)
        for fs_name in base_nc_file.fieldsets:
            self._nc[fs_name] = base_nc_file

        # Set the next trajectory index for APPEND mode.
        if self.mode == self.FileMode.APPEND:
            # Shouldn't happen.
            raise RuntimeError(
                'Appending to merged TrajectoryStore files is not supported'
            )

        # Open any associated NetCDF files.
        for name in self.associated_files:
            assert isinstance(name, PathType)
            assocated_file = self._open_merged_store(
                name, check_associated=base_nc_file
            )
            self._nc_files.append(assocated_file)
            for fs_name in assocated_file.fieldsets:
                if fs_name not in self._nc or self.override:
                    self._nc[fs_name] = assocated_file
                else:
                    warnings.warn(
                        f'FieldSet with name "{fs_name}" found in associated '
                        f'NetCDF file "{name}" already exists in base file',
                        RuntimeWarning,
                    )

    def _create(self):
        """Create a new NetCDF file (or files) for writing trajectories.

        There is one NetCDF group per field set, and more than one field set
        may be stored in each NetCDF file.

        This function is only called once the first trajectory is added to the
        store because we need to know what field sets are being stored.
        """

        # We cannot create the NetCDF file until we know what field sets are
        # involved. For that we need at least one trajectory.
        assert len(self._trajectories) > 0

        # Get a prototype trajectory: all trajectories in the store must have
        # the same field sets, so it doesn't matter which one we take.
        proto = next(iter(self._trajectories.values()))

        # Determine the field sets stored in the base NetCDF file (those not in
        # associated files).
        base_nc_fieldsets = proto.fieldsets - self.associated_fieldsets

        # Create the base NetCDF file.
        assert self.base_file is not None
        self._create_nc_file(self.base_file, base_nc_fieldsets)

        # Create the associated NetCDF files.
        base_nc_file = self._nc[BASE_FIELDSET_NAME]
        for associated_file in self.associated_files:
            # Another case of clumsy typing due to Python's restrictions on
            # isinstance.
            assert isinstance(associated_file, tuple)
            nc_file, fieldsets = associated_file
            self._create_nc_file(
                nc_file,
                set(fieldsets),
                associated_name=base_nc_file.path[0],
                associated_hash=base_nc_file.dataset[0].id_hash,
            )

    def _check_file_paths(self, paths: list[PathType], mode: FileMode) -> None:
        # Ensure all input paths are distinct.
        resolved_paths = [Path(p).resolve() for p in paths]
        if len(resolved_paths) != len(set(resolved_paths)):
            raise ValueError('Input NetCDF file paths must be distinct')

        if mode == TrajectoryStore.FileMode.CREATE:
            # In CREATE mode, ensure no input files already exist and that the
            # parent directories do exist.
            for p in resolved_paths:
                if p.exists():
                    raise ValueError(f'Input file {p} already exists')
                if not p.parent.exists():
                    raise ValueError(f'Parent directory of file {p} does not exist')
        else:
            # In READ and APPEND modes, ensure all input files exist.
            for p in resolved_paths:
                if not p.exists():
                    raise ValueError(f'Input file {p} does not exist')

            # Checks related to merged stores:
            #  - If any input paths are files, then all must be files (normal
            #    store).
            #  - If any input paths are directories, then all must be
            #    directories (merged store).
            #  - For a merged store, all input directories must have a name
            #    suffix of ".aeic-store".
            #  - For a merged store, only reading is allowed (no appending).
            #  - For a merged store, all directories must contain a
            #    metadata.json file and the same number of NetCDF files.
            self.merged_store = any(p.is_dir() for p in resolved_paths)
            normal = any(p.is_file() for p in resolved_paths)
            if self.merged_store and normal:
                raise ValueError('Invalid mix of files and directories in input paths')
            if self.merged_store:
                if mode != TrajectoryStore.FileMode.READ:
                    raise ValueError('Merged stores may only be opened in READ mode')
                num_files: int | None = None
                for p in resolved_paths:
                    if not str(p.name).endswith('.aeic-store'):
                        raise ValueError(
                            f'Merged store directory "{p}" does not have '
                            f'".aeic-store" suffix'
                        )
                    metadata_file = p / 'metadata.json'
                    if not metadata_file.exists():
                        raise ValueError(
                            f'Metadata file missing from merged store directory "{p}"'
                        )
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    nfiles = len(metadata.get('stores', []))
                    if num_files is None:
                        num_files = nfiles
                    elif nfiles != num_files:
                        raise ValueError(
                            'All merged store directories must contain '
                            'the same number of NetCDF files'
                        )

    def _open_nc_file(
        self,
        nc_file: PathType,
        check_associated: 'TrajectoryStore.NcFiles | None' = None,
    ) -> 'TrajectoryStore.NcFiles':
        # Ensure input file exists.
        nc_file = Path(nc_file).resolve()
        if not nc_file.exists():
            raise ValueError(f'Input file {nc_file} does not exist')

        # Open NetCDF4 dataset in read or append mode.
        dataset = Dataset(
            nc_file,
            mode='r' if self.mode == self.FileMode.READ else 'a',
            keepweakref=True,
        )

        # Retrieve trajectory dimension.
        traj_dim = dataset.dimensions['trajectory']

        # Retrieve global attributes.
        title = getattr(dataset, 'title', None)
        comment = getattr(dataset, 'comment', None)
        history = getattr(dataset, 'history', None)
        source = getattr(dataset, 'source', None)
        created_str = getattr(dataset, 'created', None)
        created = None
        if created_str is not None:
            created = datetime.fromisoformat(created_str)

        # Retrieve additional global attributes for consistency checking.
        fieldset_names = dataset.fieldset_names
        if isinstance(fieldset_names, str):
            fieldset_names = [fieldset_names]
        fieldset_hashes = dataset.fieldset_hashes
        if isinstance(fieldset_hashes, str):
            fieldset_hashes = [fieldset_hashes]
        id_hash = dataset.id_hash
        associated_name = getattr(dataset, 'associated_name', None)
        associated_hash = getattr(dataset, 'associated_hash', None)

        # Retrieve groups for each field set.
        groups = {k: [g] for k, g in dataset.groups.items() if not k[0] == '_'}

        # Check consistency.
        if set(groups.keys()) != set(fieldset_names):
            raise ValueError(
                f'Field set names in global attribute do not match NetCDF groups '
                f'in file {nc_file}'
            )
        for fs_name, fs_hash in zip(fieldset_names, fieldset_hashes):
            fs = FieldSet.from_registry(fs_name)
            if fs.digest != fs_hash:
                if not self.force_fieldset_matches:
                    raise ValueError(
                        f'Field set hash for field set "{fs_name}" in file {nc_file} '
                        f'does not match hash of FieldSet in registry'
                    )
                warnings.warn(
                    f'Field set hash for field set "{fs_name}" in file {nc_file} '
                    f'does not match hash of FieldSet in registry, but '
                    f'force_fieldset_matches is True so continuing anyway',
                    RuntimeWarning,
                )
        m = hashlib.md5()
        for h in fieldset_hashes:
            m.update(h.encode('utf-8'))
        if id_hash != m.hexdigest():
            raise ValueError(
                f'id_hash in NetCDF file {nc_file} does not match calculated '
                f'hash from field set hashes'
            )
        if check_associated is not None:
            if associated_name is None or associated_hash is None:
                raise ValueError(
                    f'Associated file attributes missing from NetCDF file {nc_file}'
                )
            if associated_hash != check_associated.dataset[0].id_hash:
                raise ValueError(
                    f'Associated file hash in NetCDF file {nc_file} does not match '
                    f'hash of base file {check_associated.path}'
                )

        return TrajectoryStore.NcFiles(
            path=[nc_file],
            fieldsets=set(fieldset_names),
            dataset=[dataset],
            traj_dim=[traj_dim],
            groups=groups,
            size_index=[len(traj_dim)],
            title=title,
            comment=comment,
            history=history,
            source=source,
            created=created,
        )

    def _reindex(self):
        if not self.indexable or not self.index_stale:
            return
        gs = self._nc[BASE_FIELDSET_NAME].groups[BASE_FIELDSET_NAME]
        flight_ids = []
        for g in gs:
            flight_ids += list(g.variables['flight_id'][:])
        id_pairs = sorted(enumerate(flight_ids), key=lambda x: x[1])
        assert self.index_group is not None
        self.index_group.variables['flight_id'][:] = [id for _, id in id_pairs]
        self.index_group.variables['trajectory_index'][:] = [idx for idx, _ in id_pairs]
        self.index_stale = False

    def lookup(self, flight_id: int) -> Trajectory | None:
        """Lookup a trajectory by flight ID."""
        if not self.indexable:
            raise RuntimeError('Cannot lookup by flight_id in non-indexable store')

        # Reindex lazily if needed.
        if self.index_stale:
            self._reindex()

        assert self.index_group is not None
        flight_ids = self.index_group.variables['flight_id'][:]
        traj_idxs = self.index_group.variables['trajectory_index'][:]
        idx = np.searchsorted(flight_ids, flight_id)
        if idx >= len(flight_ids) or flight_ids[idx] != flight_id:
            return None
        return self[traj_idxs[idx]]

    def _open_merged_store(
        self,
        store_dir: PathType,
        check_associated: 'TrajectoryStore.NcFiles | None' = None,
    ) -> 'TrajectoryStore.NcFiles':
        # Ensure input file exists.
        store_dir = Path(store_dir).resolve()
        if not store_dir.exists():
            raise ValueError(f'Input store {store_dir} does not exist')

        # Read metadata file.
        metadata_file = store_dir / 'metadata.json'
        if not metadata_file.exists():
            raise ValueError(f'Metadata file missing from merged store {store_dir}')
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get paths to NetCDF files.
        nc_files = [store_dir / s[0] for s in metadata.get('stores', [])]
        if len(nc_files) == 0:
            raise ValueError(f'No stores listed in metadata file {metadata_file}')

        # Open NetCDF4 dataset in read or append mode.
        dataset = [Dataset(nc_file, mode='r', keepweakref=True) for nc_file in nc_files]

        # Retrieve trajectory dimension.
        traj_dim = [ds.dimensions['trajectory'] for ds in dataset]

        # Retrieve global attributes.
        title = metadata.get('title', None)
        comment = metadata.get('comment', None)
        history = metadata.get('history', None)
        source = metadata.get('source', None)
        created_str = metadata.get('created', None)
        created = None
        if created_str is not None:
            created = datetime.fromisoformat(created_str)

        # Retrieve additional global attributes for consistency checking.
        def get_list(ds: Dataset, attr: str) -> list[str]:
            val = getattr(ds, attr, None)
            if isinstance(val, str):
                return [val]
            assert isinstance(val, list)
            return val

        fieldset_names = get_list(dataset[0], 'fieldset_names')
        fieldset_hashes = get_list(dataset[0], 'fieldset_hashes')
        id_hash = dataset[0].id_hash
        associated_name = getattr(dataset[0], 'associated_name', None)
        associated_hash = getattr(dataset[0], 'associated_hash', None)

        # Retrieve groups for each field set.
        groups = {}
        for k in dataset[0].groups.keys():
            if k[0] != '_':
                groups[k] = [ds.groups[k] for ds in dataset]

        # Check consistency.
        # TODO: Abstract this stuff, share with _open and check for all files.
        if set(groups.keys()) != set(fieldset_names):
            raise ValueError(
                f'Field set names in global attribute do not match NetCDF groups '
                f'in store {store_dir}'
            )
        for fs_name, fs_hash in zip(fieldset_names, fieldset_hashes):
            fs = FieldSet.from_registry(fs_name)
            if fs.digest != fs_hash:
                if not self.force_fieldset_matches:
                    raise ValueError(
                        f'Field set hash for field set "{fs_name}" '
                        f'in store {store_dir} does not match '
                        'hash of FieldSet in registry'
                    )
                warnings.warn(
                    f'Field set hash for field set "{fs_name}" in '
                    f'store {store_dir} does not match hash of '
                    'FieldSet in registry, but '
                    'force_fieldset_matches is True so continuing anyway',
                    RuntimeWarning,
                )
        m = hashlib.md5()
        for h in fieldset_hashes:
            m.update(h.encode('utf-8'))
        if id_hash != m.hexdigest():
            raise ValueError(
                f'id_hash in store {store_dir} does not match calculated '
                f'hash from field set hashes'
            )
        if check_associated is not None:
            if associated_name is None or associated_hash is None:
                raise ValueError(
                    f'Associated file attributes missing from store {store_dir}'
                )
            if associated_hash != check_associated.dataset[0].id_hash:
                raise ValueError(
                    f'Associated file hash in NetCDF store {store_dir} does '
                    f'not match hash of base file {check_associated.path}'
                )

        # Open index file.
        index_file = store_dir / '_index.nc'
        if index_file.exists():
            index_dataset = Dataset(index_file, mode='r', keepweakref=True)
            self.index_group = index_dataset.groups['_index']
            self.indexable = True

        return TrajectoryStore.NcFiles(
            path=nc_files,
            fieldsets=set(fieldset_names),
            dataset=dataset,
            traj_dim=traj_dim,
            groups=groups,
            size_index=list(itertools.accumulate([len(td) for td in traj_dim])),
            title=title,
            comment=comment,
            history=history,
            source=source,
            created=created,
        )

    def _create_nc_file(
        self,
        nc_file: str | Path,
        fieldsets: set[str],
        associated_name: Path | None = None,
        associated_hash: str | None = None,
        save: bool = True,
    ) -> 'TrajectoryStore.NcFiles':
        # Ensure output directory exists.
        nc_file = Path(nc_file).resolve()
        if not nc_file.parent.exists():
            raise ValueError(f'Output directory {nc_file.parent} does not exist')

        # Ensure file does not already exist.
        if nc_file.exists():
            raise ValueError(f'Output file {nc_file} already exists')

        # Create NetCDF4 file in write mode.
        dataset = Dataset(nc_file, mode='w', format='NETCDF4', keepweakref=True)

        # The only dimension we need is the trajectory. All per-point data is
        # stored using NetCDF4 variable-length arrays.
        traj_dim = dataset.createDimension('trajectory', None)

        # Set up basic global attributes.
        if len(self.global_attributes) > 0:
            if 'title' in self.global_attributes:
                dataset.title = self.global_attributes['title']
            if 'comment' in self.global_attributes:
                dataset.comment = self.global_attributes['comment']
            if 'history' in self.global_attributes:
                dataset.history = self.global_attributes['history']
            if 'source' in self.global_attributes:
                dataset.source = self.global_attributes['source']

        # Set up creation time global attribute.
        if save and 'created' not in self.global_attributes:
            t = datetime.now(UTC).astimezone().isoformat()
            self.global_attributes['created'] = t
            dataset.created = self.global_attributes['created']

        # Set up global attribute containing field set names.
        fs_names = sorted(
            list(fieldsets), key=lambda s: '0000' + s if s == 'base' else s
        )
        dataset.fieldset_names = fs_names

        # Set up global attribute containing hex digest hashes of field sets.
        fs_hashes = [FieldSet.from_registry(name).digest for name in fs_names]
        dataset.fieldset_hashes = fs_hashes

        #  Calculate and set id_hash attribute from hashes of field sets.
        m = hashlib.md5()
        for h in fs_hashes:
            m.update(h.encode('utf-8'))
        dataset.id_hash = m.hexdigest()

        # Set up associated file attributes, if applicable.
        if associated_name is not None:
            dataset.associated_name = str(associated_name)
        if associated_hash is not None:
            dataset.associated_hash = associated_hash

        # Each named field set gets its own group in the NetCDF file.
        groups = {}

        # Variable length types for floating point and integer per-point data.
        # (NetCDF4 handles variable-length strings natively so we add Python's
        # string type here.)
        vl_types: dict[type, VLType | type] = {str: str}
        for fs_name in fieldsets:
            fs = FieldSet.from_registry(fs_name)
            for field in fs.values():
                if not field.metadata and field.field_type not in vl_types:
                    try:
                        vl_types[field.field_type] = dataset.createVLType(
                            field.field_type, f'{field.field_type.__name__}_vlen'
                        )
                    except TypeError:
                        raise ValueError(
                            f'Unsupported field type {field.field_type} '
                            f'for variable-length array in field set "{fs_name}"'
                        )

        # Iterate over provided field set names.
        for fs_name in fieldsets:
            # Look up the field set in the field set registry. By the time we
            # reach here, the field sets should have already been created,
            # which stores them in the registry.
            fs = FieldSet.from_registry(fs_name)

            # Create the field set's group.
            g = dataset.createGroup(fs_name)
            groups[fs_name] = [g]

            # For each variable (per-point or per-trajectory) in the field set,
            for field_name, metadata in fs.items():
                if not metadata.metadata:
                    # if it's a per-point variable, create a variable-length
                    # array of the apprropriate scalar type, indexed by the
                    # trajectory dimension.
                    vl_type = vl_types.get(metadata.field_type, None)
                    if vl_type is None:
                        raise ValueError(
                            f'Unsupported field type {metadata.field_type} '
                            f'for variable "{field_name}" in field set "{fs_name}"'
                        )
                    v = g.createVariable(field_name, vl_type, ('trajectory',))
                else:
                    # if it's a per-trajectory variable, create a simple
                    # variable with a normal NetCDF type, indexed by the
                    # trajectory dimension.
                    v = g.createVariable(
                        field_name, metadata.field_type, ('trajectory',)
                    )

                # Add metadata to variable.
                v.description = metadata.description
                v.units = metadata.units
                v.required = 'true' if metadata.required else 'false'

                # The variable `v` doesn't need to be saved anywhere special
                # now; it can be accessed from its group.

        if self.indexable and self.index_group is None:
            self.index_group = dataset.createGroup('_index')
            self.index_group.createVariable('flight_id', np.int64, ('trajectory',))
            self.index_group.createVariable(
                'trajectory_index', np.int64, ('trajectory',)
            )

        # Save information about file for lookup by field set name. This is how
        # we find the file and group for a given field set later on.
        #
        # To get from a variable (attribute) name to its field set, we look it
        # up in the trajectory's data dictionary, getting a FieldSet: the name
        # of the FieldSet is the name of the NetCDF group where we find the
        # variable.
        file_info = self.NcFiles(
            path=[nc_file],
            fieldsets=fieldsets,
            dataset=[dataset],
            traj_dim=[traj_dim],
            size_index=None,
            groups=groups,
        )
        if save:
            self._nc_files.append(file_info)
            for fs_name in fieldsets:
                self._nc[fs_name] = file_info
        return file_info


def valid_associated_file_tuple(t) -> bool:
    """Check if t is a tuple[str, list[str]]."""
    if not isinstance(t, tuple) or len(t) != 2:
        return False
    if not isinstance(t[0], PathType) or not isinstance(t[1], list):
        return False
    if any(not isinstance(name, str) for name in t[1]):
        return False
    return True
