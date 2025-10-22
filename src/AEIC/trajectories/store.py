import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from cachetools import LRUCache
from netCDF4 import Dataset, Dimension, Group, VLType

from .field_sets import FieldSet
from .trajectory import BASE_FIELDSET_NAME, Trajectory

# Python doesn't have a simple way of saying "anything that's acceptable as a
# filesystem path", so define a simple type alias instead.

PathType = str | Path


AssociatedFileCreate = tuple[PathType, list[str]]
AssociatedFileOpen = PathType
AssociatedFiles = list[AssociatedFileCreate] | list[AssociatedFileOpen]


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

    This is a complicated and flexible class that needs a big documentation
    comment here.
    TODO: Write it!
    """

    class FileMode(str, Enum):
        READ = 'r'
        CREATE = 'w'
        APPEND = 'a'

    @dataclass
    class NcFile:
        """Internal class used to store information about NetCDF files
        associated with a TrajectoryStore."""

        path: Path
        fieldsets: set[str]
        dataset: Dataset
        traj_dim: Dimension
        groups: dict[str, Group]
        title: str | None = None
        comment: str | None = None
        history: str | None = None
        source: str | None = None
        created: datetime | None = None

    # Allowed constructor arguments by mode:
    #
    # (' ' = not allowed, X = required, ? = optional)
    #
    #  R W A
    #  -----------------------
    #    ?    title, comment, history, source: Permitted only for creating a
    #           new NetCDF file.
    #  X ? X  nc_file
    #  X X X  mode
    #  ?      fieldsets
    #  ?      override
    #  ? ? ?  *associated_nc_files: must be str for APPEND and READ (field sets
    #           in each file are fixed already) and must be tuple[str,
    #           list[str]] for CREATE (allotment of field sets to associated
    #           files must be specified).
    def __init__(
        self,
        /,
        nc_file: str | None = None,
        mode: FileMode = FileMode.READ,
        override: bool | None = None,
        cache_size_mb: int = 2048,
        associated_nc_files: AssociatedFiles | None = None,
        title: str | None = None,
        comment: str | None = None,
        history: str | None = None,
        source: str | None = None,
    ):
        """Initialize a TrajectoryStore with various file access modes.

        Parameter
        ----------
        nc_file : str | None, optional
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
        associated_nc_files : list[PathType] | list[tuple[PathType, list[str]]] | None
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
        self.mode = mode

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

        # Setting nc_file=None in CREATE mode creates an empty TrajectoryStore.
        # We can switch to a file-backed store later using the save method, but
        # in the meantime, the trajectory cache is limited in size and cannot
        # evict any items. Adding too many trajectories to an in-memory
        # TrajectoryStore will result in a TrajectoryCache.EvictionOccurred
        # exception.
        nc_file_req = mode != self.FileMode.CREATE
        if nc_file is None and nc_file_req:
            raise ValueError(f'nc_file required in file mode {mode}')
        self.nc_file = nc_file

        override_ok = mode == self.FileMode.READ
        if override is not None and not override_ok:
            raise ValueError('override may only be specified in READ mode')
        self.override = override if override is not None else False
        if override is not None:
            raise NotImplementedError(
                'TrajectoryStore.__init__: override not yet implemented'
            )
        # TODO: Implement field set override behavior.

        if associated_nc_files is None:
            associated_nc_files = []
        if mode in (self.FileMode.APPEND, self.FileMode.READ):
            if any(not isinstance(f, PathType) for f in associated_nc_files):
                raise ValueError(
                    'associated_nc_files must be paths in APPEND and READ modes'
                )
        if mode == self.FileMode.CREATE:
            if any(not valid_associated_file_tuple(f) for f in associated_nc_files):
                raise ValueError(
                    'associated_nc_files must be '
                    'tuple[PathType, list[str]] in CREATE mode'
                )
        self.associated_nc_files = associated_nc_files
        self.associated_fieldsets: set[str] = set()
        if mode == self.FileMode.CREATE:
            # This is slightly unwieldy because of Python's restrictions on
            # using isinstance with parameterized generics.
            assert isinstance(associated_nc_files, list)
            self.associated_fieldsets = set()
            for t in associated_nc_files:
                assert isinstance(t, tuple)
                for f in t[1]:
                    self.associated_fieldsets.add(f)

        # Trajectories are stored as an LRU cache indexed by the index of the
        # trajectory. (This is done to handle cases where the trajectory store
        # is very large and cannot be held in memory all at once.)
        #
        # A store created with nc_file=None is in-memory only. We can switch to
        # a file-backed store using the save method, but in the meantime, the
        # trajectory cache cannot evict any entries. The custom TrajectoryCache
        # class has a flag to raise an exception on eviction for this use case.
        # TODO: THIS IS NOT THREAD-SAFE! ADD NECESSARY LOCKING.
        self._trajectories = TrajectoryCache(
            cache_size_mb * 1024 * 1024,
            getsizeof=lambda t: t.nbytes,  # type: ignore
        )
        if self.nc_file is None:
            self._trajectories.exception_on_eviction = True

        # List of NetCDF file information structures and mapping from field set
        # names to NetCDF file information. The first entry in self._nc_files is
        # the base NetCDF file; any others are associated files.
        self._nc_files: list[TrajectoryStore.NcFile] = []
        self._nc: dict[str, TrajectoryStore.NcFile] = {}

        # If we're opening in CREATE mode, set up for file creation. We can't
        # create the file (or files) until we know what field sets are
        # involved, and we only get to see that when a `Trajectory` is added to
        # the store.
        self._file_creation_pending = mode == self.FileMode.CREATE

        # Next trajectory index to assign. This is the length of the trajectory
        # dimension. It gets set to the correct value for APPEND mode when we
        # open the existing file.
        self._next_index = 0

        # Whether writing is enabled (CREATE or APPEND mode).
        self._write_enabled = mode in (self.FileMode.CREATE, self.FileMode.APPEND)

        # Open an existing file or files.
        if mode in (self.FileMode.READ, self.FileMode.APPEND):
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
    def files(self) -> list[NcFile]:
        """NetCDF file information associated with the trajectory store.

        The base NetCDF file is the first entry in the list; any associated
        files follow.
        """
        return self._nc_files

    def save(self, nc_file: str):
        """Create a NetCDF files for a TrajectoryStore currently not linked to
        one."""
        # TODO: Implement this method.
        raise NotImplementedError('TrajectoryStore.save not yet implemented')

        # Once the files have been created successfully and the existing data
        # persisted, we can allow evictions from the trajectory cache.
        self._trajectories.exception_on_eviction = False

    def create_associated(self, nc_file: str, fieldsets: list[str]):
        """Create an associated NetCDF file for additional field sets."""
        # TODO: Make the signature here more correct and think about
        # implementation: this is essentially a map over trajectories in the
        # trajectory store.
        ...

    @property
    def nc_linked(self) -> bool:
        """Is a NetCDF file (or files) associated with the trajectory store?"""
        return len(self._nc_files) != 0

    def close(self):
        """Close any open NetCDF files associated with the trajectory store."""
        for nc in self._nc_files:
            nc.dataset.close()
        self._nc.clear()
        self._nc_files.clear()

    def sync(self):
        """Synchronize any pending writes to the NetCDF file or files.

        Note that this does not necessarily make the NetCDF files readable by
        another application because of NetCDF4's finalization behavior. To
        ensure complete finalization, call close() instead.
        """
        for nc in self._nc_files:
            nc.dataset.sync()

    def __len__(self):
        """Count number of trajectories in store."""
        if self.nc_linked:
            return len(self._nc[BASE_FIELDSET_NAME].traj_dim)
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

        return saved_index

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
            nc_file = self._nc[fs_name]
            group = nc_file.groups[fs_name]

            # Write per-point data as variable-length arrays.
            for name, field in fs.items():
                if name not in group.variables:
                    raise ValueError(
                        f'Data field "{name}" does not exist in NetCDF file'
                    )
                data[name] = group.variables[name][index]
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

        # I thought about implementing some sort of batching here, but I think
        # we might be able to rely on the internal caching that the NetCDF4
        # library does.

        # Look up the trajectory to write.
        traj = self._trajectories[index]

        # Handle field sets one by one.
        for fs_name in self._nc:
            # File information for the field set and the NetCDF group for
            # variables in the field set.
            nc_file = self._nc[fs_name]
            group = nc_file.groups[fs_name]

            # Write per-point data as variable-length arrays.
            for name in group.variables:
                field = traj.data_dictionary[name]
                if not field.metadata:
                    if traj.data[name] is None:
                        raise ValueError(
                            f'Per-point data field "{name}" is None for trajectory '
                            f'at index {index}'
                        )
                    group.variables[name][index] = traj.data[name]
                else:
                    if traj.metadata[name] is None:
                        raise ValueError(
                            f'Metadata field "{name}" is None for trajectory '
                            f'at index {index}'
                        )
                    group.variables[name][index] = traj.metadata[name]

    def _open(self):
        """Open an existing NetCDF file (or files) for reading/appending
        trajectories.

        There is one NetCDF group per field set, and more than one field set
        may be stored in each NetCDF file.
        """

        # Open the NetCDF4 dataset from base file.
        assert self.nc_file is not None
        base_nc_file = self._open_nc_file(self.nc_file)
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
            netcdf_fs = FieldSet.from_netcdf_group(base_nc_file.groups[fs_name])

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
            self._next_index = len(base_nc_file.traj_dim)

        # Open any associated NetCDF files.
        for name in self.associated_nc_files:
            assert isinstance(name, PathType)
            assocated_file = self._open_nc_file(name, check_associated=base_nc_file)
            self._nc_files.append(assocated_file)
            for fs_name in assocated_file.fieldsets:
                if fs_name in self._nc:
                    raise ValueError(
                        f'FieldSet with name "{fs_name}" found in associated '
                        f'NetCDF file "{name}" already exists in base file'
                    )
                self._nc[fs_name] = assocated_file

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
        assert self.nc_file is not None
        self._create_nc_file(self.nc_file, base_nc_fieldsets)

        # Create the associated NetCDF files.
        base_nc_file = self._nc[BASE_FIELDSET_NAME]
        for associated_file in self.associated_nc_files:
            # Another case of clumsy typing due to Python's restrictions on
            # isinstance.
            assert isinstance(associated_file, tuple)
            nc_file, fieldsets = associated_file
            self._create_nc_file(
                nc_file,
                set(fieldsets),
                associated_name=base_nc_file.path,
                associated_hash=base_nc_file.dataset.id_hash,
            )

    def _open_nc_file(
        self,
        nc_file: PathType,
        check_associated: 'TrajectoryStore.NcFile | None' = None,
    ) -> 'TrajectoryStore.NcFile':
        # Ensure input file exists.
        nc_file = Path(nc_file).resolve()
        if not nc_file.exists():
            raise ValueError(f'Input file {nc_file} does not exist')

        # Open NetCDF4 dataset in read or append mode.
        dataset = Dataset(nc_file, mode='r' if self.mode == self.FileMode.READ else 'a')

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
        groups = dataset.groups

        # Check consistency.
        if set(groups.keys()) != set(fieldset_names):
            raise ValueError(
                f'Field set names in global attribute do not match NetCDF groups '
                f'in file {nc_file}'
            )
        for fs_name, fs_hash in zip(fieldset_names, fieldset_hashes):
            fs = FieldSet.from_registry(fs_name)
            # TODO: Add force option to make this into a warning instead of
            # error.
            if fs.digest != fs_hash:
                raise ValueError(
                    f'Field set hash for field set "{fs_name}" in file {nc_file} '
                    f'does not match hash of FieldSet in registry'
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
            if associated_hash != check_associated.dataset.id_hash:
                raise ValueError(
                    f'Associated file hash in NetCDF file {nc_file} does not match '
                    f'hash of base file {check_associated.path}'
                )

        return TrajectoryStore.NcFile(
            path=nc_file,
            fieldsets=set(fieldset_names),
            dataset=dataset,
            traj_dim=traj_dim,
            groups=groups,
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
    ) -> None:
        # Ensure output directory exists.
        nc_file = Path(nc_file).resolve()
        if not nc_file.parent.exists():
            raise ValueError(f'Output directory {nc_file.parent} does not exist')

        # Ensure file does not already exist.
        if nc_file.exists():
            raise ValueError(f'Output file {nc_file} already exists')

        # Create NetCDF4 file in write mode.
        dataset = Dataset(nc_file, mode='w', format='NETCDF4')

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
        self.global_attributes['created'] = datetime.now(UTC).astimezone().isoformat()
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
            groups[fs_name] = g

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

                # The variable `v` doesn't need to be saved anywhere special
                # now; it can be accessed from its group.

        # Save information about file for lookup by field set name. This is how
        # we find the file and group for a given field set later on.
        #
        # To get from a variable (attribute) name to its field set, we look it
        # up in the trajectory's data dictionary, getting a FieldSet: the name
        # of the FieldSet is the name of the NetCDF group where we find the
        # variable.
        file_info = self.NcFile(
            path=nc_file,
            fieldsets=fieldsets,
            dataset=dataset,
            traj_dim=traj_dim,
            groups=groups,
        )
        self._nc_files.append(file_info)
        for fs_name in fieldsets:
            self._nc[fs_name] = file_info


def valid_associated_file_tuple(t) -> bool:
    """Check if t is a tuple[str, list[str]]."""
    if not isinstance(t, tuple) or len(t) != 2:
        return False
    if not isinstance(t[0], PathType) or not isinstance(t[1], list):
        return False
    if any(not isinstance(name, str) for name in t[1]):
        return False
    return True
