import pytest

from AEIC.storage import Dimension, Dimensions


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


def test_dimensions_str_repr():
    dims_tp = Dimensions(Dimension.TRAJECTORY, Dimension.POINT)
    dims_tm = Dimensions(Dimension.TRAJECTORY, Dimension.THRUST_MODE)
    assert str(dims_tp) == 'Dimensions(TP)'
    assert repr(dims_tm) == 'Dimensions(TM)'


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


def test_dimensions_netcdf():
    # POINT is excluded — it's the implicit per-trajectory axis in the
    # on-disk layout — and the remaining names come back in standard
    # Dimension-enum order regardless of construction order.
    assert Dimensions(Dimension.TRAJECTORY, Dimension.POINT).netcdf == ('trajectory',)
    assert Dimensions(
        Dimension.POINT, Dimension.SPECIES, Dimension.TRAJECTORY
    ).netcdf == ('trajectory', 'species')
    assert Dimensions(
        Dimension.THRUST_MODE, Dimension.SPECIES, Dimension.TRAJECTORY
    ).netcdf == ('trajectory', 'species', 'thrust_mode')


def test_dimensions_abbrevs():
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.POINT, Dimension.SPECIES)
    assert dims.abbrev == 'TSP'
    dims = Dimensions.from_abbrev('TS')
    assert dims == Dimensions(Dimension.TRAJECTORY, Dimension.SPECIES)
    with pytest.raises(ValueError):
        Dimensions.from_abbrev('TSX')


def test_dimension_dim_name():
    assert Dimension.TRAJECTORY.dim_name == 'trajectory'
    assert Dimension.POINT.dim_name == 'point'
    assert Dimension.SPECIES.dim_name == 'species'
    assert Dimension.THRUST_MODE.dim_name == 'thrust_mode'


def test_dimensions_remove():
    dims = Dimensions(Dimension.TRAJECTORY, Dimension.POINT, Dimension.SPECIES)
    reduced = dims.remove(Dimension.SPECIES)
    assert reduced == Dimensions(Dimension.TRAJECTORY, Dimension.POINT)
    # Removing a dimension that isn't present is a no-op (no raise).
    assert dims.remove(Dimension.THRUST_MODE) == dims
    # Removing the trajectory dimension violates the constructor invariant
    # and must raise rather than silently produce an invalid Dimensions.
    with pytest.raises(ValueError):
        dims.remove(Dimension.TRAJECTORY)


def test_dimensions_from_dim_names():
    dims = Dimensions.from_dim_names('trajectory', 'point')
    assert dims == Dimensions(Dimension.TRAJECTORY, Dimension.POINT)
    dims = Dimensions.from_dim_names('trajectory', 'species', 'thrust_mode')
    assert dims == Dimensions(
        Dimension.TRAJECTORY, Dimension.SPECIES, Dimension.THRUST_MODE
    )
    with pytest.raises(ValueError):
        Dimensions.from_dim_names('trajectory', 'not_a_dim')
