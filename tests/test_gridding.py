from AEIC.config import config
from AEIC.gridding.grid import Grid


def test_height_grid_load():
    # Load a grid with heights in the vertical and check values.
    grid = Grid.load(config.file_location('grids/basic-1x1.toml'))
    assert grid.longitude.resolution == 1.0
    assert grid.longitude.offset == 0.5
    assert grid.latitude.resolution == 1.0
    assert grid.latitude.offset == 0.5
    assert grid.altitude.mode == 'height'
    assert grid.altitude.resolution == 500.0
    assert grid.altitude.bottom == 0.0
    assert not hasattr(grid.altitude, 'levels')


def test_pressure_grid_load():
    # Load a grid with ISA pressures in the vertical and check values.
    grid = Grid.load(config.file_location('grids/basic-1x1-isa-pressure.toml'))
    assert grid.longitude.resolution == 1.0
    assert grid.longitude.offset == 0.5
    assert grid.latitude.resolution == 1.0
    assert grid.latitude.offset == 0.5
    assert grid.altitude.mode == 'isa_pressure'
    assert grid.altitude.levels == [1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0]
    assert grid.altitude.bottom == 1000.0
    assert grid.altitude.top == 100.0
