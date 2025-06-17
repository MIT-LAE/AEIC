import os


def file_location(path: str) -> str:
    # Try local file first.
    if os.path.exists(path):
        return path

    # Try in the data directory.
    data_dir = os.environ.get('AEIC_DATA_DIR')
    if data_dir is not None:
        path = os.path.join(data_dir, path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"File {path} not found in local or data directory.")