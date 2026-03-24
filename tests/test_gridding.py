from AEIC.config import config
from AEIC.gridding.grid import Grid


def test_grid_load():
    # Load the grid and check values.
    grid = Grid.load(config.file_location('grids/basic-1x1.toml'))
    assert grid.longitude.resolution == 1.0
    assert grid.longitude.offset == -0.5
    assert grid.latitude.resolution == 1.0
    assert grid.latitude.offset == -0.5
    assert grid.altitude.mode == 'height'
    assert grid.altitude.resolution == 500.0
    assert grid.altitude.offset == 0.0
    assert grid.altitude.levels is None
