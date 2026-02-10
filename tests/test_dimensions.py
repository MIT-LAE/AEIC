import pytest

from AEIC.trajectories.dimensions import Dimension, Dimensions


def test_dimension_from_name():
    assert Dimension.from_dim_name('trajectory') == Dimension.TRAJECTORY
    assert Dimension.from_dim_name('point') == Dimension.POINT
    assert Dimension.from_dim_name('thrust_mode') == Dimension.THRUST_MODE
    with pytest.raises(ValueError):
        Dimension.from_dim_name('invalid')


def test_dimensions_creation():
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.POINT)
    assert Dimension.TRAJECTORY in dims
    assert Dimension.POINT in dims
    assert Dimension.THRUST_MODE not in dims

    with pytest.raises(ValueError):
        Dimensions(Dimension.POINT)

    with pytest.raises(ValueError):
        Dimensions(Dimension.TRAJECTORY, Dimension.POINT, Dimension.THRUST_MODE)


def test_dimensions_length():
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.POINT)
    assert len(dims) == 2
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.THRUST_MODE)
    assert len(dims) == 2
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.POINT, Dimension.POINT)
    assert len(dims) == 2
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.SPECIES, Dimension.POINT)
    assert len(dims) == 3


def test_dimensions_equality():
    dims1 = Dimensions(Dimension.TRAJECTORY, Dimension.POINT)
    dims2 = Dimensions(Dimension.POINT, Dimension.TRAJECTORY)
    dims3 = Dimensions(Dimension.TRAJECTORY, Dimension.THRUST_MODE)
    assert dims1 == dims2
    assert dims1 != dims3
    assert dims1 != 'not a Dimensions object'
    assert str(dims1) == 'Dimensions(TP)'
    assert repr(dims3) == 'Dimensions(TM)'


def test_dimensions_add():
    dims = Dimensions(Dimension.TRAJECTORY)
    dims = dims.add(Dimension.POINT)
    assert Dimension.POINT in dims
    with pytest.raises(ValueError):
        dims.add(Dimension.THRUST_MODE)
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.THRUST_MODE)
    with pytest.raises(ValueError):
        dims.add(Dimension.POINT)


def test_dimensions_ordered():
    dims = Dimensions(Dimension.POINT, Dimension.TRAJECTORY, Dimension.SPECIES)
    assert dims.ordered == [Dimension.TRAJECTORY, Dimension.SPECIES, Dimension.POINT]


def test_dimensions_abbrevs():
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.POINT, Dimension.SPECIES)
    assert dims.abbrev == 'TSP'
    dims = Dimensions.from_abbrev('TS')
    assert dims == Dimensions(Dimension.TRAJECTORY, Dimension.SPECIES)
    with pytest.raises(ValueError):
        Dimensions.from_abbrev('TSX')
