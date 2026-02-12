# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import bisect
import gc
import hashlib
import itertools
import json
import os
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Protocol

import netCDF4 as nc4
import numpy as np
from cachetools import LRUCache

from AEIC.performance.types import ThrustMode, ThrustModeValues
from AEIC.types import Species, SpeciesValues

from .dimensions import Dimension
from .field_sets import FieldMetadata, FieldSet, HasFieldSets
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

    **Trajectory stores are not thread-safe. A program may create and use
    TrajectoryStore values from a single thread only. This single-threaded
    restriction applies both to read-only access and mutation of stores.**

    **Field sets**

    Data stored in a `TrajectoryStore` is divided into "field sets"
    (represented by the `FieldSet` class from the
    `AEIC.trajectories.field_sets` package). A field set is a collection of
    pointwise and per-trajectory data fields that are part of a trajectory or
    data that lives alongside a trajectory (emissions data of one sort or
    another, for example). "Pointwise data fields" in field sets have values
    for each point along a trajectory: the length of the data values in each of
    these fields must match the length of the trajectory. "Per-trajectory data
    fields" in field sets are per-trajectory values: there is one value of each
    of these fields for each trajectory. Each field in a field set has a name,
    a data type and associated information used for serialization to and from
    NetCDF files.

    A `TrajectoryStore` always contains the "base" field set, which holds the
    basic trajectory data (defined as `BASE_FIELDS` in the
    `AEIC.trajectories.trajectory` package). Additional field sets can be
    stored in a `TrajectoryStore` as needed.

    The set of field sets contained in a `TrajectoryStore` is determined when
    the first trajectory is added to the store. All subsequent trajectories
    added must have the same field sets.

    **Access modes and NetCDF files**

    The `TrajectoryStore` class supports three access modes: CREATE, READ and
    APPEND, accessed using the `create`, `open` and `append` class methods. In
    addition, it is possible to create additional associated NetCDF files
    associated with an existing `TrajectoryStore` using the `create_associated`
    class method. In general, `TrajectoryStore` instances should be managed
    using a `with` context manager to ensure that files are closed properly
    when no longer needed. This is particularly important because of the tricky
    finalization semantics of the underlying NetCDF and HDF5 libraries.

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

    **Base and associated files**

    Every `TrajectoryStore` has a "base" NetCDF file (or files, for a merged
    store) that contains the base field set and possibly additional field sets.
    Additional field sets may be stored in separate "associated" NetCDF files.
    Each associated file may contain one or more field sets.

    When opening a `TrajectoryStore` in READ or APPEND mode, a list of
    associated files may be provided along with the base file. Trajectories
    retrieved from the resulting store will have data from all field sets
    stored in the base and associated files.

    When creating a new `TrajectoryStore` in CREATE mode, a list of associated
    files may be provided along with the base file. In this case, the list
    must specify which field sets are to be stored in each associated file.
    When the first trajectory is added to the store, the base and associated
    files are created and the field sets are distributed among them as
    specified.

    Associated files may be created from an existing `TrajectoryStore` using
    the `create_associated` class method. This is essentially a mapping
    operation, taking each trajectory in the store, applying a user-supplied
    function to generate new data values, and storing the resulting data in a
    new associated NetCDF file. The intended use case here is for the
    calculation of data like emissions that are associated with trajectories
    but are not part of the trajectory data itself, and so may be calculated
    separately.

    When an associated file is created, metadata is stored within the file to
    link it to the base file from which it was created. This linkage is checked
    when opening a `TrajectoryStore` with associated files to ensure that the
    associated files correspond to the base file.

    **Merged stores**

    To simplify management of large trajectory data sets that may be split
    across multiple NetCDF files, the `TrajectoryStore` class supports "merged"
    stores. A merged store is stored as a directory containing multiple NetCDF
    files, each of which contains a subset of the trajectories in the store,
    along with a JSON metadata file and possible a separate NetCDF file for the
    store's flight ID index.

    A merged store is created using the `merge` class method, which takes as
    input a list of existing NetCDF files and the name of a directory in which
    to create the merged store. The resulting merged store may then be opened
    in READ mode like any other `TrajectoryStore`. Merged stores must have
    extension ".aeic-store".

    Merged stores may be created from both base files and associated files.

    **Trajectory access and mission database indexing**

    Trajectories can be retrieved from a trajectory store simply by indexing
    the store with the integer index of the trajectory within the store.
    Indexes are assigned in order from zero in the order of insertion of
    trajectories. Indexes into merged stores run consecutively from zero across
    all the constituent NetCDF files composing the merged store.

    In addition, if all trajectories in the store have a `flight_id` field
    (representing the mission database flight ID for the trajectory), the store
    can be indexed by flight ID as well. In this case, the store is said to be
    "indexable". If a store is indexable, the `flight_id` field of each
    trajectory must be unique within the store. A trajectory can be retrieved
    by flight ID using the `get_flight` method.

    """

    class FileMode(StrEnum):
        READ = 'r'
        CREATE = 'w'
        APPEND = 'a'

    @dataclass
    class NcFiles:
        """Internal class used to store information about NetCDF files
        associated with field sets in a TrajectoryStore.
        """

        path: list[Path]
        """Paths to NetCDF files: multiple to support merged stores."""

        fieldsets: set[str]
        """Field sets stored in the NetCDF files."""

        dataset: list[nc4.Dataset]
        """Netcdf4 Dataset objects for the files: multiple to support merged
        stores."""

        traj_dim: list[nc4.Dimension]
        """Trajectory dimension objects for the files: multiple to support
        merged stores."""

        traj_var: list[nc4.Variable]
        """Trajectory variable objects for the files: multiple to support
        merged stores."""

        species: list[Species] | None
        """Species included in the species dimension in the files, if any."""

        groups: dict[str, list[nc4.Group]]
        """Mapping from field set names to NetCDF groups in the files: groups
        are multiple for each field set name to support merged stores."""

        size_index: list[int] | None
        """Cumulative trajectory count through the NetCDF files represented
        here: not used for single NetCDF stores, but for merged stored, used to
        find the underlying NetCDF file containing a given trajectory index."""

        title: str | None = None
        """Title global attribute value."""

        comment: str | None = None
        """Comment global attribute value."""

        history: str | None = None
        """History global attribute value."""

        source: str | None = None
        """Source global attribute value."""

        created: datetime | None = None
        """Creation time global attribute value."""

    active_in_thread: int | None = None
    """Thread ID of active TrajectoryStore instance, if any. Multi-threaded
    access is not allowed. This attribute is used to check for this."""

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
    #  ? ? ?  *associated_files: must be list[str] for APPEND and READ (field
    #         sets in each file are fixed already) and must be list[tuple[str,
    #         list[str]]] for CREATE (allotment of field sets to associated
    #         files must be specified).
    def __init__(
        self,
        *,
        base_file: PathType | None = None,
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
            - READ: Open an existing store for read-only access.
            - CREATE: Create a new store for writing.
            - APPEND: Open an existing store for appending new trajectories.
            Default is READ.
        override : bool | None, optional
            If true in READ mode, field sets from associated files will override
            any field sets of the same name in the base file. Default is False.
        force_fieldset_matches : bool | None, optional
            If True in READ mode, field sets from NetCDF files that do not match
            the corresponding FieldSet in the registry will be accepted with a
            warning. Default is False. There are no guarantees that things will
            work if this is set to True!
        associated_files : list[PathType] | list[tuple[PathType, list[str]]] | None
            Paths to associated NetCDF files containing additional pointwise or
            per-trajectory data fields. Each associated file may be specified as
            a string path or as a tuple of the form (path, [fieldset_names]) where
            fieldset_names is a list of names of field sets to load from the
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
        # see if it has a flight ID and set this accordingly; similarly, if
        # we're creating a new store, we'll decide on indexability when we see
        # the first trajectory added to the store.
        #
        # The index information is stored in a special NetCDF group that we
        # access separately from the main NcFiles mechanism.
        self.indexable = None
        self.index_group: nc4.Group | None = None
        self.index_dataset: nc4.Dataset | None = None

        # The index can become stale when new items are added to the store. We
        # reindex lazily, either when an index lookup is performed or on a sync
        # or close.
        self.index_stale = False

        # Default values for other attributes.
        self.global_attributes = {}
        self.merged_store = False

        # Check that all constructor arguments are consistent.
        self._check_constructor_arguments(
            mode,
            base_file,
            override,
            force_fieldset_matches,
            associated_files,
            title,
            comment,
            history,
            source,
        )

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
            try:
                if self.merged_store:
                    self._open_merged()
                else:
                    self._open()
            except Exception:
                self.close()
                raise

    @classmethod
    def create(cls, *args, **kwargs) -> TrajectoryStore:
        """Create a new TrajectoryStore.

        This class method is the preferred way to create a new trajectory
        store. Call this method in preference to calling the class constructor
        directly.
        """
        return cls(mode=TrajectoryStore.FileMode.CREATE, *args, **kwargs)

    @classmethod
    def open(cls, *args, **kwargs) -> TrajectoryStore:
        """Open an existing TrajectoryStore read-only.

        This class method is the preferred way to open an existing trajectory
        store for reading. Call this method in preference to calling the class
        constructor directly.
        """
        return cls(mode=TrajectoryStore.FileMode.READ, *args, **kwargs)

    @classmethod
    def append(cls, *args, **kwargs) -> TrajectoryStore:
        """Open an existing TrajectoryStore for appending.

        This class method is the preferred way to open an existing trajectory
        store for appending. Call this method in preference to calling the
        class constructor directly.
        """
        return cls(mode=TrajectoryStore.FileMode.APPEND, *args, **kwargs)

    @property
    def files(self) -> list[NcFiles]:
        """NetCDF file information associated with the trajectory store.

        The base NetCDF file is the first entry in the list; any associated
        files follow.
        """
        return self._nc_files

    def save(
        self, base_file: PathType, associated_files: AssociatedFiles | None = None
    ):
        """Create NetCDF files for a TrajectoryStore currently not linked to
        one.

        This method is used when an in-memory `TrajectoryStore` needs to be
        persisted to disk. This method can *only* be called on a
        `TrajectoryStore` created in CREATE mode with `base_file=None`.
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
        self.associated_files = associated_files if associated_files is not None else []
        check_paths += self._check_associated_files('save')
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

        # We need to call the mapping once to see what chemical species we need
        # to put in the species dimension int he new NetCDF file.
        nc_info = None

        for i in range(len(self)):
            # Create associated data value.
            associated_data = mapping_function(self[i], *args, **kwargs)

            # Create file: we can only do this once we have one result from the
            # mapping function, because we need to determine the species we
            # need to allow in the NetCDF files' species dimension.
            if nc_info is None:
                species = set()
                for fs_name in fieldsets:
                    fs = FieldSet.from_registry(fs_name)
                    for f, metadata in fs.fields.items():
                        if Dimension.SPECIES in metadata.dimensions:
                            species.update(getattr(associated_data, f).keys())

                nc_info = self._create_nc_file(
                    associated_file,
                    set(fieldsets),
                    sorted(species),
                    save=False,
                    associated_name=self._nc_files[0].path[0],
                    associated_hash=self._nc_files[0].dataset[0].id_hash,
                )

            # These are the only checks we're going to do here: the call to
            # _write_data will fail if any of the fields from the field sets
            # are missing or of the wrong type.
            if not isinstance(associated_data, HasFieldSets):
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

        assert nc_info is not None
        nc_info.dataset[0].close()

    @property
    def nc_linked(self) -> bool:
        """Is a NetCDF file (or files) associated with the trajectory store?"""
        return len(self._nc_files) != 0

    def close(self):
        """Close any open NetCDF files associated with the trajectory store."""

        # If we have an index and it's stale, reindex before closing to ensure
        # consistency.
        if self.indexable and self.index_stale:
            self._reindex()

        # If there is a separate index Dataset (also for merged stores), close
        # it too.
        self.index_group = None
        if self.index_dataset is not None:
            self.index_dataset.close()
            self.index_dataset = None

        # Close each NetCDF4 Dataset associated with each of the open stores
        # (there will be multiple Datasets if we're using a merged store).
        for nc in self._nc_files:
            for ds in nc.dataset:
                ds.close()

        # Clear out everything to do with NetCDF4 Datasets we had open.
        self._nc.clear()
        self._nc_files.clear()

        # Try to force finalization of NetCDF4 objects.
        gc.collect()

    def sync(self):
        """Synchronize any pending writes to the NetCDF file or files.

        Note that this does not necessarily make the NetCDF files readable by
        another application because of NetCDF4's finalization behavior. To
        ensure complete finalization, call close() instead.
        """
        if not self._write_enabled:
            raise RuntimeError('Cannot sync TrajectoryStore not opened in write mode')

        # If we have an index and it's stale, reindex as part of the sync.
        if self.indexable and self.index_stale:
            self._reindex()

        # Sync each NetCDF4 Dataset associated with each of the open stores
        # (there will be multiple Datasets if we're using a merged store).
        for nc in self._nc_files:
            for ds in nc.dataset:
                ds.sync()

        # If there is a separate index Dataset (also for merged stores), sync
        # it too.
        if self.index_dataset is not None:
            self.index_dataset.sync()

    def add(self, trajectory: Trajectory) -> int:
        """Add a trajectory to the store and return its index."""
        if not self._write_enabled:
            raise RuntimeError(
                'Cannot add trajectory to TrajectoryStore not opened in write mode'
            )

        # As soon as we've added one trajectory to the store, we have fixed the
        # data schema, which we check for each new trajectory.
        if len(self._trajectories) > 0:
            proto = next(iter(self._trajectories.values()))
            if hash(trajectory) != hash(proto):
                raise ValueError(
                    'All trajectories in a TrajectoryStore must have the same '
                    'data fields'
                )

        # Decide on whether or not we can index the store, checking consistency
        # on this decision with each trajectory we add.
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

        # Whenever we add a trajectory, the trajectory index is no longer up to
        # date. For efficiency, we do not reindex immediately, deferring either
        # to close or sync of the store, or to when an index lookup is
        # requested.
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
        """Merge multiple `TrajectoryStore` files into a single merged store.

        The merging is done simply by copying the individual NetCDF files into
        a directory and creating a metadata JSON file in the same directory to
        record the metadata about the merged store and its constituent files.
        (If the store is indexed, a separate NetCDF file is also created to
        hold the merged index.) The resulting merged store can then be opened
        in READ mode like any other `TrajectoryStore`. Merged stores must have
        extension `.aeic-store`."""

        # Check parameters and normalize input stores list.
        input_stores = TrajectoryStore._check_merge_arguments(
            output_store, input_stores, input_stores_pattern, input_stores_index_range
        )

        # Create output directory.
        os.mkdir(output_store)

        # Collect metadata and check that the field sets match.
        store_data = []
        fieldset_names: set[str] | None = None
        index_groups = []
        assert input_stores is not None
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
            TrajectoryStore._create_merged_store_index(output_store, input_stores)

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

    def get_flight(self, flight_id: int) -> Trajectory | None:
        """Lookup a trajectory by flight ID."""
        if not self.indexable:
            raise RuntimeError('Cannot lookup by flight_id in non-indexable store')

        # Reindex lazily if needed.
        if self.index_stale:
            self._reindex()

        # The flight IDs and trajectory indexes are stored in parallel in the
        # index group.
        assert self.index_group is not None
        flight_ids = self.index_group.variables['flight_id'][:]
        traj_idxs = self.index_group.variables['trajectory_index'][:]

        # Binary search in the flight IDs.
        idx = bisect.bisect_left(flight_ids, flight_id)

        # Check that we really found the flight ID.
        if idx >= len(flight_ids) or flight_ids[idx] != flight_id:
            return None

        # Look up the trajectory index in the same position as the flight ID we
        # found and return the trajectory in that position.
        return self[traj_idxs[idx]]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, tb):
        self.close()
        # Returning False propagates exceptions.
        return False

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

            # Return the sum of the trajectory dimension lengths in all of the
            # NetCDF4 Datasets associated with the field set we're using to
            # measure the length. (There will be multiple Datasets if we're
            # using a merged store.)
            return sum(len(d) for d in self._nc[check_fs].traj_dim)
        return len(self._trajectories)

    def __getitem__(self, idx) -> Trajectory:
        """Retrieve a trajectory by numeric index (in order of addition)."""

        # If the requested index is in the LRU trajectory cache, return it
        # immediately.
        if idx in self._trajectories:
            return self._trajectories[idx]

        # Otherwise, if the store is linked to external NetCDF files, attempt
        # to load the requested trajectory into the cache.
        if self.nc_linked:
            self._load_trajectory(idx)

        # Load failed or the index is unknown.
        if idx not in self._trajectories:
            raise IndexError('Trajectory index out of range')

        # Return the trajectory from the cache.
        return self._trajectories[idx]

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterator over trajectories in store in index order."""
        return _TrajectoryStoreIterator(self)

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
        base_nc_fieldsets = proto.X_fieldsets - self.associated_fieldsets

        # Create the base NetCDF file.
        assert self.base_file is not None
        species = proto.species
        self._create_nc_file(self.base_file, base_nc_fieldsets, species)

        # Create the associated NetCDF files. The `associated_name` and
        # `associated_hash` arguments are used to record the link between the
        # associated files and the base file.
        base_nc_file = self._nc[BASE_FIELDSET_NAME]
        for associated_file in self.associated_files:
            # Another case of clumsy typing due to Python's restrictions on
            # isinstance.
            assert isinstance(associated_file, tuple)
            nc_file, fieldsets = associated_file
            self._create_nc_file(
                nc_file,
                set(fieldsets),
                species,
                associated_name=base_nc_file.path[0],
                associated_hash=base_nc_file.dataset[0].id_hash,
            )

    def _retrieve_nc_species_values(self, dataset: nc4.Dataset) -> list[Species] | None:
        """Retrieve species values from a NetCDF Dataset's species dimension.

        If the Dataset does not have a species dimension, return None.
        """
        if 'species' not in dataset.dimensions:
            return None

        var = dataset.variables.get('species', None)
        if var is None:
            raise ValueError('Dataset has species dimension but no species variable')

        return [
            Species[var[i].decode('utf-8') if isinstance(var[i], bytes) else var[i]]
            for i in range(len(var))
        ]

    def _create_nc_file(
        self,
        nc_file: str | Path,
        fieldsets: set[str],
        species: list[Species],
        associated_name: Path | None = None,
        associated_hash: str | None = None,
        save: bool = True,
    ) -> TrajectoryStore.NcFiles:
        # Ensure output directory exists.
        nc_file = Path(nc_file).resolve()
        if not nc_file.parent.exists():
            raise ValueError(f'Output directory {nc_file.parent} does not exist')

        # Ensure file does not already exist.
        if nc_file.exists():
            raise ValueError(f'Output file {nc_file} already exists')

        # Create NetCDF4 file in write mode.
        dataset = nc4.Dataset(nc_file, mode='w', format='NETCDF4', keepweakref=True)

        # Create dimensions and any required coordinate variables.
        traj_dim, traj_var = _create_dimensions(dataset, fieldsets, species)

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

        # Set up global attribute containing field set names. (We mess with the
        # sort order here to make sure that the "base" field set comes first.)
        fs_names = sorted(
            list(fieldsets), key=lambda s: '0000' + s if s == 'base' else s
        )
        dataset.fieldset_names = fs_names

        # Set up global attribute containing hex digest hashes of field sets.
        # These are used for making the overall ID hash for the store, which is
        # used for linking from associated files.
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
        vl_types = _create_vl_types(dataset, fieldsets)

        # Iterate over provided field set names.
        for fs_name in fieldsets:
            # Look up the field set in the field set registry. By the time we
            # reach here, the field sets should have already been created,
            # which stores them in the registry.
            fs = FieldSet.from_registry(fs_name)

            # Create the field set's group. (We need a list of groups here for
            # the merged store case.)
            g = dataset.createGroup(fs_name)
            groups[fs_name] = [g]

            # For each variable in the field set:
            for field_name, metadata in fs.items():
                # Determine the NetCDF variable type: if it's a per-point
                # variable, look up the appropriate variable length type
                # created earlier.
                field_type = metadata.field_type
                if Dimension.POINT in metadata.dimensions:
                    field_type = vl_types.get(metadata.field_type, None)
                    if field_type is None:
                        raise ValueError(
                            f'Unsupported field type {metadata.field_type} '
                            f'for variable "{field_name}" in field set "{fs_name}"'
                        )

                # Create a variable of the appropriate NetCDF type, indexed by
                # the calculated dimension set.
                v = g.createVariable(field_name, field_type, metadata.dimensions.netcdf)

                # Add metadata to variable.
                v.description = metadata.description
                v.units = metadata.units
                v.required = 'true' if metadata.required else 'false'
                if metadata.default is not None:
                    v.default = metadata.default

                # The variable `v` doesn't need to be saved anywhere special
                # now; it can be accessed via its group.

        # If we're indexing, we need to create the index group and variables.
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
            traj_var=[traj_var],
            species=species,
            size_index=None,
            groups=groups,
        )
        if save:
            self._nc_files.append(file_info)
            for fs_name in fieldsets:
                self._nc[fs_name] = file_info
        return file_info

    def _open(self):
        """Open an existing NetCDF file (or files) for reading/appending
        trajectories.

        There is one NetCDF group per field set, and more than one field set
        may be stored in each NetCDF file.
        """

        # Open the NetCDF4 dataset from base file.
        assert self.base_file is not None
        base_nc_file = self._open_nc_file(self.base_file)
        self._base_open_checks(base_nc_file)

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
            associated_file = self._open_nc_file(name, check_associated=base_nc_file)
            self._associated_open_checks(name, associated_file)

    def _open_nc_file(
        self,
        nc_file: PathType,
        check_associated: TrajectoryStore.NcFiles | None = None,
    ) -> TrajectoryStore.NcFiles:
        """Open a single NetCDF file.

        This method opens a single NetCDF file that is either a base file or an
        associated file. It performs consistency checks on the contents of the
        file and returns a `NcFiles` object with information about the file."""

        # Ensure input file exists.
        nc_file = Path(nc_file).resolve()
        if not nc_file.exists():
            raise ValueError(f'Input file {nc_file} does not exist')

        # Open NetCDF4 dataset in read or append mode.
        dataset = nc4.Dataset(
            nc_file,
            mode='r' if self.mode == self.FileMode.READ else 'a',
            keepweakref=True,
        )

        # Retrieve trajectory dimension and variable.
        traj_dim = dataset.dimensions['trajectory']
        traj_var = dataset.variables['trajectory']

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

        # Retrieve groups for each field set. Group names with a leading
        # underscore are reserved for internal use, in particular for the
        # flight ID index.
        groups = {k: [g] for k, g in dataset.groups.items() if not k[0] == '_'}

        # Retrieve values of species dimension, if any.
        species = self._retrieve_nc_species_values(dataset)

        # Check consistency.

        # The groups in the file should match the field sets we're claiming are
        # in the file.
        if set(groups.keys()) != set(fieldset_names):
            raise ValueError(
                f'Field set names in global attribute do not match NetCDF groups '
                f'in file {nc_file}'
            )

        # The field sets in the file should match the definitions in the
        # registry.
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

        # The ID hash is made up of all the field set hashes in the file.
        m = hashlib.md5()
        for h in fieldset_hashes:
            m.update(h.encode('utf-8'))
        if id_hash != m.hexdigest():
            raise ValueError(
                f'id_hash in NetCDF file {nc_file} does not match calculated '
                f'hash from field set hashes'
            )

        # For an associated file, check that the base file it's associated with
        # matches what we expect.
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

        # Here, the `path`, `dataset`, `traj_dim`, `traj_var` and `size_index`
        # fields are lists to support merged stores. In this case, we have a
        # single file, so we put the values into singleton lists.
        return TrajectoryStore.NcFiles(
            path=[nc_file],
            fieldsets=set(fieldset_names),
            dataset=[dataset],
            traj_dim=[traj_dim],
            traj_var=[traj_var],
            species=species,
            groups=groups,
            size_index=[len(traj_dim)],
            title=title,
            comment=comment,
            history=history,
            source=source,
            created=created,
        )

    def _open_merged(self):
        """Open an existing merged store file for reading trajectories."""

        # Open the NetCDF4 dataset from base file.
        assert self.base_file is not None
        base_nc_file = self._open_merged_store(self.base_file)
        self._base_open_checks(base_nc_file)

        # Set the next trajectory index for APPEND mode.
        if self.mode == self.FileMode.APPEND:
            # Shouldn't happen.
            raise RuntimeError(
                'Appending to merged TrajectoryStore files is not supported'
            )

        # Open index file.
        index_file = Path(self.base_file) / '_index.nc'
        if index_file.exists():
            self.index_dataset = nc4.Dataset(index_file, mode='r', keepweakref=True)
            self.index_group = self.index_dataset.groups['_index']
            self.indexable = True

        # Open any associated NetCDF files.
        for name in self.associated_files:
            assert isinstance(name, PathType)
            associated_file = self._open_merged_store(
                name, check_associated=base_nc_file
            )
            self._associated_open_checks(name, associated_file)

    def _open_merged_store(
        self,
        store_dir: PathType,
        check_associated: TrajectoryStore.NcFiles | None = None,
    ) -> TrajectoryStore.NcFiles:
        """Open a merged store.

        This method opens multiple NetCDF files in a merged store directory
        that are either a base store or an associated store. It performs
        consistency checks on the contents of the store and returns a `NcFiles`
        object with information about the files.
        """

        # Ensure input file exists.
        store_dir = Path(store_dir).resolve()
        if not store_dir.exists():
            raise ValueError(f'Input store {store_dir} does not exist')

        # Read metadata file. This contains information about the files
        # composing the merged store, as well as store level global attribute
        # metadata values.
        metadata_file = store_dir / 'metadata.json'
        if not metadata_file.exists():
            raise ValueError(f'Metadata file missing from merged store {store_dir}')
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get paths to NetCDF files.
        nc_files = [store_dir / s[0] for s in metadata.get('stores', [])]
        if len(nc_files) == 0:
            raise ValueError(f'No stores listed in metadata file {metadata_file}')

        # Open NetCDF4 datasets in read mode for each NetCDF file.
        dataset = [
            nc4.Dataset(nc_file, mode='r', keepweakref=True) for nc_file in nc_files
        ]

        # Retrieve trajectory dimensions and variables.
        traj_dim = [ds.dimensions['trajectory'] for ds in dataset]
        traj_var = [ds.variables['trajectory'] for ds in dataset]

        # Retrieve global attributes from JSON data (created when merged store
        # is created).
        title = metadata.get('title', None)
        comment = metadata.get('comment', None)
        history = metadata.get('history', None)
        source = metadata.get('source', None)
        created_str = metadata.get('created', None)
        created = None
        if created_str is not None:
            created = datetime.fromisoformat(created_str)

        # Retrieve additional global attributes for consistency checking.
        def get_list(ds: nc4.Dataset, attr: str) -> list[str]:
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

        # Check consistency.
        for ds in dataset:
            check_groups = {}
            for k in ds.groups.keys():
                if k[0] != '_':
                    check_groups[k] = [ds.groups[k] for ds in dataset]
            if set(check_groups.keys()) != set(fieldset_names):
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

        # Retrieve groups for each field set: take from first dataset, since
        # we've checked consistency across all datasets already.
        groups = {}
        for k in dataset[0].groups.keys():
            if k[0] != '_':
                groups[k] = [ds.groups[k] for ds in dataset]

        # Retrieve species actually used in the NetCDF files.
        species = self._retrieve_nc_species_values(dataset[0])

        return TrajectoryStore.NcFiles(
            path=nc_files,
            fieldsets=set(fieldset_names),
            dataset=dataset,
            traj_dim=traj_dim,
            traj_var=traj_var,
            species=species,
            groups=groups,
            size_index=list(itertools.accumulate([len(td) for td in traj_dim])),
            title=title,
            comment=comment,
            history=history,
            source=source,
            created=created,
        )

    def _base_open_checks(self, base_nc_file: NcFiles):
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

                # Now we can check that the fields in the field set exist in the
                # relevant NetCDF group and that they have the right types and
                # shapes.
                for n in fs.fields:
                    if n not in netcdf_fs.fields:
                        raise ValueError(
                            f'Field "{n}" in FieldSet with name "{fs_name}" not '
                            f'found in NetCDF file'
                        )
                    if fs.fields[n] != netcdf_fs.fields[n]:
                        raise ValueError(
                            f'Field "{n}" in FieldSet with name "{fs_name}" is '
                            f'incompatible with field in NetCDF file'
                        )
                if hash(fs) != hash(netcdf_fs):
                    raise ValueError(
                        f'FieldSet with name "{fs_name}" in base NetCDF file is '
                        f'incompatible with FieldSet in registry'
                    )

        # Save file information under fieldset names in _nc dictionary.
        self._nc_files.append(base_nc_file)
        for fs_name in base_nc_file.fieldsets:
            self._nc[fs_name] = base_nc_file

    def _associated_open_checks(self, name: PathType, associated_file: NcFiles):
        self._nc_files.append(associated_file)
        for fs_name in associated_file.fieldsets:
            if fs_name not in self._nc or self.override:
                self._nc[fs_name] = associated_file
            else:
                warnings.warn(
                    f'FieldSet with name "{fs_name}" found in associated '
                    f'NetCDF file "{name}" already exists in base file',
                    RuntimeWarning,
                )

    def _reindex(self):
        """Regenerate flight ID index."""

        # NOTE: Takes about 1.5s on a store with 1 million trajectories.

        if not self.indexable or not self.index_stale:
            return

        # Get the NetCDF4 groups for the base field set.
        gs = self._nc[BASE_FIELDSET_NAME].groups[BASE_FIELDSET_NAME]

        # Extract flight IDs from all trajectories in the order of the files in
        # the store (for a merged store, there may be more than one file).
        flight_ids = []
        for g in gs:
            flight_ids += list(g.variables['flight_id'][:])

        # Pair up the flight IDs with the indexes of the trajectories in the
        # store and sort the pairs into flight ID order so that we can binary
        # search the index.
        id_pairs = sorted(enumerate(flight_ids), key=lambda x: x[1])

        # Save the index information into the index group.
        assert self.index_group is not None
        self.index_group.variables['flight_id'][:] = [id for _, id in id_pairs]
        self.index_group.variables['trajectory_index'][:] = [idx for idx, _ in id_pairs]

        # We've just remade the index, so it's definitely not stale.
        self.index_stale = False

    @staticmethod
    def _create_merged_store_index(
        output_store: PathType, input_stores: list[PathType]
    ):
        index_dataset = nc4.Dataset(
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
        index_dataset.close()

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

            # If this is a merged store, find the right file and index into the
            # right group in that file.
            if nc_files.size_index is not None:
                file_index = bisect.bisect_left(nc_files.size_index, index + 1)
                if file_index >= len(nc_files.size_index):
                    return
                group_index = index - nc_files.size_index[file_index]
            group = nc_files.groups[fs_name][file_index]

            # Read data from NetCDF variables.
            for name, field in fs.items():
                if name not in group.variables:
                    raise ValueError(
                        f'Data field "{name}" does not exist in NetCDF file'
                    )
                val = self._read_from_nc_var(
                    group.variables[name],
                    group_index,
                    name,
                    field,
                    nc_files.species or [],
                )
                data[name] = val
                if Dimension.POINT in field.dimensions and npoints is None:
                    if Dimension.SPECIES in field.dimensions:
                        # Get number of points from arbitrary entry in the
                        # SpeciesValues dictionary here.
                        npoints = len(next(iter(data[name].values())))
                    else:
                        # Data should be a simple Numpy array here.
                        npoints = len(data[name])

        # Construct the return trajectory.
        assert npoints is not None
        traj = Trajectory(npoints=npoints)
        for fs_name in self._nc:
            if fs_name != BASE_FIELDSET_NAME:
                traj.add_fields(FieldSet.from_registry(fs_name))
        for k, v in data.items():
            setattr(traj, k, v)

        # Save the trajectory we've just loaded into the cache.
        self._trajectories[index] = traj

    def _write_trajectory(self, index: int) -> None:
        """Write a trajectory at the given index to the NetCDF file(s)."""

        # Look up the trajectory to write and write to files. We use a
        # `_write_data` helper function so that we can also use `_write_data`
        # for creating associated files.
        self._write_data(traj=self._trajectories[index], index=index)

    def _write_data(
        self,
        *,
        index: int,
        traj: Trajectory | None = None,
        data: Any | None = None,
        single_nc_file: TrajectoryStore.NcFiles | None = None,
        fieldsets: list[str] | None = None,
    ) -> None:
        """Write a trajectory at the given index to the NetCDF file(s)."""

        # Check: should be one or the other. We're either saving a real
        # trajectory (called from `_write_trajectory`) or writing data to an
        # associated file (called from `create_associated`).
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

            for name in group.variables:
                field = fs[name]
                var = group.variables[name]

                # Get data either from the trajectory or the "associated data"
                # we've been passed.
                val = None
                if traj is not None:
                    val = getattr(traj, name)
                elif data is not None:
                    val = getattr(data, name)

                self._write_to_nc_var(var, index, name, field, val)
                nc_file.traj_var[0][index] = index

    def _write_to_nc_var(
        self,
        var: nc4.Variable,
        index: int,
        name: str,
        field: FieldMetadata,
        val: Any,
    ) -> None:
        """Write a value to a NetCDF variable at the given index."""

        # Handle missing values.
        if val is None:
            if field.required:
                raise ValueError(f'Data field "{name}" is None at index {index}')
            return

        # Save data to NetCDF variable, handling species and thrust modes. At
        # this point, we assume that all the types are correct, since these
        # will have been checked earlier. Numpy array values are saved as
        # variable length types of the appropriate base type.
        has_sp = Dimension.SPECIES in field.dimensions
        has_tm = Dimension.THRUST_MODE in field.dimensions
        match (has_sp, has_tm):
            case (False, False):
                # float, np.ndarray
                var[index] = val
            case (False, True):
                # ThrustModeValues
                for ti, tm in enumerate(ThrustMode):
                    var[index, ti] = val[tm]
            case (True, False):
                # SpeciesValues[float], SpeciesValues[np.ndarray]
                for si, sp in enumerate(Species):
                    if sp in val:
                        var[index, si] = val[sp]
            case (True, True):
                # SpeciesValues[ThrustModeValues]
                for si, sp in enumerate(Species):
                    for ti, tm in enumerate(ThrustMode):
                        if sp in val and tm in val[sp]:
                            var[index, si, ti] = val[sp][tm]

    def _read_from_nc_var(
        self,
        var: nc4.Variable,
        index: int,
        name: str,
        field: FieldMetadata,
        species: list[Species],
    ) -> Any:
        """Read a value from a NetCDF variable at the given index."""

        # Make sure the netCDF4 package doesn't return masked arrays.
        var.set_auto_mask(False)

        # Read data from NetCDF variable, handling species and thrust modes.
        match (
            Dimension.SPECIES in field.dimensions,
            Dimension.THRUST_MODE in field.dimensions,
            Dimension.POINT in field.dimensions,
        ):
            case (False, False, False):
                # float
                if var[index] == var.get_fill_value():
                    return None
                return var[index]
            case (False, False, True):
                # np.ndarray
                if all(var[index] == var.get_fill_value()):
                    return None
                return var[index]
            case (True, False, False) | (True, False, True):
                # SpeciesValues[float] | SpeciesValues[np.ndarray]
                return SpeciesValues(
                    {sp: var[index, si] for si, sp in enumerate(species)}
                )
            case (False, True, False):
                # ThrustModeValues
                return ThrustModeValues(
                    {tm: var[index, ti] for ti, tm in enumerate(ThrustMode)}
                )
            case (True, True, False):
                # SpeciesValues[ThrustModeValues]
                return SpeciesValues[ThrustModeValues](
                    {
                        sp: ThrustModeValues(
                            {tm: var[index, si, ti] for ti, tm in enumerate(ThrustMode)}
                        )
                        for si, sp in enumerate(species)
                    }
                )
            case _:
                raise ValueError(f'Invalid combination of dimensions for field {name}')

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

    def _check_constructor_arguments(
        self,
        mode: FileMode,
        base_file: PathType | None = None,
        override: bool | None = None,
        force_fieldset_matches: bool | None = None,
        associated_files: AssociatedFiles | None = None,
        title: str | None = None,
        comment: str | None = None,
        history: str | None = None,
        source: str | None = None,
    ):
        """Check constructor input arguments based on file mode."""

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

        self.associated_files = associated_files if associated_files is not None else []
        if mode in (self.FileMode.APPEND, self.FileMode.READ):
            if any(not isinstance(f, PathType) for f in self.associated_files):
                raise ValueError(
                    'associated_files must be paths in APPEND and READ modes'
                )
            for f in self.associated_files:
                assert isinstance(f, PathType)
                check_paths.append(f)
        if mode == self.FileMode.CREATE:
            if base_file is None and len(self.associated_files) > 0:
                raise ValueError(
                    'associated_files may only be specified in CREATE mode '
                    'if base_file is also specified'
                )
            check_paths += self._check_associated_files('CREATE mode')
        self.associated_fieldsets: set[str] = set()
        if mode == self.FileMode.CREATE:
            self._calc_associated_fieldsets()

        # Check that file paths exist (for reading) or do not exist (for
        # creation).
        self._check_file_paths(check_paths, mode)

    @staticmethod
    def _check_merge_arguments(
        output_store: PathType,
        input_stores: list[PathType] | None = None,
        input_stores_pattern: PathType | None = None,
        input_stores_index_range: tuple[int, int] | None = None,
    ) -> list[PathType]:
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
        return input_stores

    @staticmethod
    def _valid_associated_file_tuple(t) -> bool:
        """Check if t is a tuple[str, list[str]]."""
        if not isinstance(t, tuple) or len(t) != 2:
            return False
        if not isinstance(t[0], PathType) or not isinstance(t[1], list):
            return False
        if any(not isinstance(name, str) for name in t[1]):
            return False
        return True

    def _calc_associated_fieldsets(self) -> None:
        """Determine field sets in associated files."""

        # This is slightly unwieldy because of Python's restrictions on
        # using isinstance with parameterized generics.
        assert isinstance(self.associated_files, list)
        self.associated_fieldsets = set()
        for t in self.associated_files:
            assert isinstance(t, tuple)
            for f in t[1]:
                self.associated_fieldsets.add(f)

    def _check_associated_files(
        self,
        label: str,
    ) -> list[PathType]:
        """Check associated files data for correctness, returning a list of
        paths to check."""

        if self.associated_files is None:
            return []
        if any(
            not TrajectoryStore._valid_associated_file_tuple(f)
            for f in self.associated_files
        ):
            raise ValueError(
                f'associated_files must be tuple[PathType, list[str]] in {label}'
            )
        check_paths: list[PathType] = []
        for f in self.associated_files:
            assert isinstance(f, tuple)
            assert isinstance(f[0], PathType)
            check_paths.append(f[0])
        return check_paths


def _create_dimensions(
    dataset: nc4.Dataset, fieldsets: set[str], species: list[Species]
) -> tuple[nc4.Dimension, nc4.Variable]:
    # The only dimension we need to keep track of explicitly is the trajectory.
    # All per-point data is stored using NetCDF4 variable-length arrays and
    # species and thrust mode dimensions are of fixed size.
    traj_dim = dataset.createDimension('trajectory', None)

    # Coordinate variable for trajectory dimension.
    traj_var = dataset.createVariable('trajectory', np.int64, ('trajectory',))

    # Create species and thrust mode dimensions and coordinate variables if
    # required.

    def create_enum_dimension(
        name: str, enum_type: type[Enum], values: list | None = None
    ):
        dataset.createDimension(
            name, len(values) if values is not None else len(enum_type)
        )
        v = dataset.createVariable(name, str, name)
        v.set_auto_mask(False)
        v.set_always_mask(False)
        for idx, m in enumerate(values if values is not None else enum_type):
            dataset.variables[name][idx] = m.name

    species_dim_created: bool = False
    thrust_mode_dim_created: bool = False
    for fs_name in fieldsets:
        fs = FieldSet.from_registry(fs_name)
        dims = fs.dimensions
        if Dimension.SPECIES in dims and not species_dim_created:
            # Create species dimension using only species in data to be saved,
            # not all defined species.
            create_enum_dimension('species', Species, species)
            species_dim_created = True
        if Dimension.THRUST_MODE in dims and not thrust_mode_dim_created:
            create_enum_dimension('thrust_mode', ThrustMode)
            thrust_mode_dim_created = True

    return traj_dim, traj_var


def _create_vl_types(
    dataset: nc4.Dataset, fieldsets: set[str]
) -> dict[type, nc4.VLType | type]:
    # Variable length types for floating point and integer per-point data.
    # (NetCDF4 handles variable-length strings natively so we add Python's
    # string type here.)
    vl_types: dict[type, nc4.VLType | type] = {str: str}
    for fs_name in fieldsets:
        fs = FieldSet.from_registry(fs_name)
        for field in fs.values():
            if Dimension.POINT in field.dimensions and field.field_type not in vl_types:
                try:
                    vl_types[field.field_type] = dataset.createVLType(
                        field.field_type, f'{field.field_type.__name__}_vlen'
                    )
                except TypeError:
                    raise ValueError(
                        f'Unsupported field type {field.field_type} '
                        f'for variable-length array in field set "{fs_name}"'
                    )

    return vl_types
