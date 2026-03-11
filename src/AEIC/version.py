from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path


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


def _compute_git_info():
    describe = _git(['describe', '--tags', '--dirty', '--always', '--long'])
    commit = _git(['rev-parse', 'HEAD'])
    branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'])
    dirty = bool(_git(['status', '--porcelain']))

    return {
        'describe': describe,
        'commit': commit,
        'branch': branch,
        'dirty': dirty,
    }


GIT_INFO = _compute_git_info()
VERSION = _compute_version()
