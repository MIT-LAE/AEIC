from pathlib import Path

import httpx
from tqdm import tqdm


def download(url: str, dest_path: Path | str) -> None:
    """Download a file from a URL to the given destination path."""

    dest_path = dest_path if isinstance(dest_path, Path) else Path(dest_path)
    if not dest_path.parent.exists():
        raise FileNotFoundError(
            f'Download destination directory {dest_path.parent} does not exist.'
        )

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
