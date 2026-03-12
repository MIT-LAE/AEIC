# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path


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

    files: list[Path] = field(default_factory=list)
    """List of file paths accessed during simulations, captured by the
       AccessRecorder."""

    @classmethod
    def build(cls) -> ReproducibilityData:
        python_version = (
            f'{sys.version_info.major}.'
            f'{sys.version_info.minor}.'
            f'{sys.version_info.micro}'
        )

        return cls(
            python_version=python_version,
            software_version=VERSION,
            git_commit=GIT_COMMIT,
            git_branch=GIT_BRANCH,
            git_dirty=GIT_DIRTY,
            files=access_recorder.paths,
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
