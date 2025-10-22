import os

import httpx
from pyproj import Geod
from tqdm import tqdm

GEOD = Geod(ellps="WGS84")


def file_location(path: str) -> str:
    """Get path to a file, checking local and data directories."""

    # Try local file first.
    if os.path.exists(path):
        return path

    # Try in the data directory.
    path = data_file_path(path)
    if path is not None and os.path.exists(path):
        return path

    raise FileNotFoundError(f"File {path} not found in local or data directory.")


def data_file_path(path: str) -> str:
    """Get the full path to a file within the data directory."""

    data_dir = os.environ.get('AEIC_DATA_DIR')
    if data_dir is None:
        raise ValueError("AEIC_DATA_DIR environment variable is not set.")

    return os.path.join(data_dir, path)


def ensure_data_directory(dir: str) -> None:
    """Ensure that the given directory exists within the data directory."""

    data_dir = os.environ.get('AEIC_DATA_DIR')
    if data_dir is None:
        return

    os.makedirs(os.path.join(data_dir, dir), exist_ok=True)


def download(url: str, dest_path: str) -> None:
    """Download a file from a URL to the given destination path."""

    ensure_data_directory(os.path.dirname(dest_path))

    with httpx.stream("GET", url) as response:
        total = int(response.headers["Content-Length"])

        with tqdm(
            total=total, unit_scale=True, unit_divisor=1024, unit="B"
        ) as progress:
            num_bytes_downloaded = response.num_bytes_downloaded
            with open(dest_path, "wb") as download_file:
                for chunk in response.iter_bytes():
                    download_file.write(chunk)
                    progress.update(
                        response.num_bytes_downloaded - num_bytes_downloaded
                    )
                    num_bytes_downloaded = response.num_bytes_downloaded
