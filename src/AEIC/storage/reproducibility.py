# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import functools
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from AEIC.config import Config, config


@dataclass
class ReproducibilityData:
    """Container for information about the software environment and accessed
    files during simulations, to enable reproduction of results and
    debugging."""

    python_version: str
    """Python version string, e.g. '3.13.12'."""

    software_version: str
    """Version string for the AEIC software, ideally including Git metadata if
       available, e.g. 'v1.2.0', 'v1.2.3-4-gabcdef-dirty' or 'unknown'."""

    git_commit: str | None
    """Git commit hash if available, or None if not in a Git repository."""

    git_branch: str | None
    """Git branch name if available, or None if not in a Git repository."""

    git_dirty: bool
    """True if there are uncommitted changes in the Git repository, False if
       clean or not in a Git repository."""

    sample_fraction: float | None
    """Optional sample fraction used during simulations, if applicable."""

    sample_seed: int | None
    """Optional random seed used during simulations, if applicable."""

    config: str
    """String representation of the AEIC configuration used during
       simulations."""

    files: list[Path] = field(default_factory=list)
    """List of file paths accessed during simulations, captured by the
       AccessRecorder."""

    def __or__(self, other: ReproducibilityData) -> ReproducibilityData:
        """Combine two ReproducibilityData instances by merging their file lists
        and ensuring other fields match (or are compatible)."""
        if self.python_version != other.python_version:
            raise ValueError('ReproducibilityData: Python version mismatch')
        if self.software_version != other.software_version:
            raise ValueError('ReproducibilityData: software version mismatch')
        if self.git_commit != other.git_commit:
            raise ValueError('ReproducibilityData: Git commit mismatch')
        if self.git_branch != other.git_branch:
            raise ValueError('ReproducibilityData: Git branch mismatch')
        if self.git_dirty != other.git_dirty:
            raise ValueError('ReproducibilityData: Git dirty state mismatch')
        if self.config != other.config:
            raise ValueError('ReproducibilityData: configuration mismatch')
        if self.sample_fraction != other.sample_fraction:
            raise ValueError('ReproducibilityData: sample fraction mismatch')
        if self.sample_seed != other.sample_seed:
            raise ValueError('ReproducibilityData: sample seed mismatch')

        combined_files = sorted(set(self.files) | set(other.files))
        return ReproducibilityData(
            python_version=self.python_version,
            software_version=self.software_version,
            git_commit=self.git_commit,
            git_branch=self.git_branch,
            git_dirty=self.git_dirty,
            config=self.config,
            sample_fraction=self.sample_fraction,
            sample_seed=self.sample_seed,
            files=combined_files,
        )

    @classmethod
    def union(cls, *data_list: ReproducibilityData) -> ReproducibilityData:
        """Combine a list of ReproducibilityData instances by merging their file
        lists and ensuring other fields match (or are compatible)."""
        if not data_list:
            raise ValueError('ReproducibilityData.union: empty data list')

        return functools.reduce(lambda a, b: a | b, data_list)

    @classmethod
    def build(
        cls, sample_fraction: float | None = None, sample_seed: int | None = None
    ) -> ReproducibilityData:
        python_version = (
            f'{sys.version_info.major}.'
            f'{sys.version_info.minor}.'
            f'{sys.version_info.micro}'
        )

        def simplify(p: Path) -> Path:
            if p.is_relative_to(config.default_data_path):
                return Path('...') / p.relative_to(config.default_data_path)
            return p

        files = [simplify(f) for f in access_recorder.paths]

        return cls(
            python_version=python_version,
            software_version=VERSION,
            git_commit=GIT_COMMIT,
            git_branch=GIT_BRANCH,
            git_dirty=GIT_DIRTY,
            files=files,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            config=Config.get().model_dump_json(),
        )


@dataclass
class AccessRecorder:
    """Utility to record file accesses during simulations, using Python's audit
    hooks."""

    _paths: set[Path] = field(default_factory=set)
    enabled: bool = False

    # TODO: Maybe just monkey-patch open(), sqlite3.connect() and
    # netCDF4.Dataset instead of using an audit hook?
    def __post_init__(self):
        sys.addaudithook(self._hook)

    def reset(self):
        """Clear the recorded paths."""
        self._paths.clear()

    def start(self):
        """Enable recording of file accesses."""
        self.enabled = True

    def stop(self):
        """Disable recording of file accesses."""
        self.enabled = False

    @property
    def paths(self) -> list[Path]:
        """Return the list of recorded file paths, sorted for consistency."""
        return sorted(self._paths)

    def _hook(self, event, args):
        if not self.enabled:
            return

        try:
            match event:
                case 'open':
                    path, mode, *_ = args
                    skip = (
                        any(path.endswith(s) for s in ['.pyc', '.py', '.so'])
                        or any(path.startswith(p) for p in ['/etc/', '/usr/'])
                        or path.find('site-packages') >= 0
                    )
                    if 'r' in mode and not skip:
                        self._paths.add(Path(path).resolve())

                case 'sqlite3.connect':
                    self._paths.add(Path(args[0]).resolve())

        except Exception:
            pass


access_recorder = AccessRecorder()
"""Single global instance of AccessRecorder to be used for tracking file
accesses during simulations."""


@contextmanager
def track_file_accesses():
    """Context manager to enable tracking of file accesses during simulations.
    Usage:

    ```python
    with track_file_accesses():
        run_simulations()
    ```

    After the block, accessed files can be retrieved from
    `access_recorder.paths`."""

    access_recorder.start()
    try:
        yield
    finally:
        access_recorder.stop()


def _in_git_repo() -> bool:
    """Check if the current file is within a Git repository by looking for a
    .git directory in the current and parent directories."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / '.git').exists():
            return True
    return False


def _git(cmd: list[str]) -> str | None:
    """Run Git command and return output, or None if not in a Git repository or
    on error."""
    if not _in_git_repo():
        return None
    try:
        return subprocess.check_output(
            ['git', *cmd],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _compute_version() -> str:
    """Determine the version string for the AEIC package, preferring Git metadata if
    available, and falling back to installed package metadata."""

    # Prefer Git if available.
    describe = _git(['describe', '--tags', '--dirty', '--always', '--long'])
    if describe:
        return describe

    # Fall back to installed package metadata.
    try:
        return pkg_version('AEIC')
    except PackageNotFoundError:
        return 'unknown'


# At top-level to capture information once at import time.
GIT_COMMIT = _git(['rev-parse', 'HEAD'])
GIT_BRANCH = _git(['rev-parse', '--abbrev-ref', 'HEAD'])
GIT_DIRTY = _git(['status', '--porcelain']) != ''
VERSION = _compute_version()
