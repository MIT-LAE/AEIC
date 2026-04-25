import functools

import numpy as np
import pytest

import AEIC.trajectories.builders as tb
from AEIC.emissions.emission import compute_emissions
from AEIC.trajectories import TrajectoryStore
from AEIC.types import SpeciesValues


@pytest.mark.forked
def test_emissions_storage(tmp_path, sample_missions, performance_model, fuel):
    fname = tmp_path / 'test_trajectories_with_emissions.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore.create(base_file=fname)

    original_totals = []
    for mis in sample_missions:
        traj = builder.fly(performance_model, mis)
        emissions = compute_emissions(performance_model, fuel, traj)
        original_totals.append(dict(emissions.total_emissions))
        traj.add_fields(emissions)
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore.open(base_file=fname)
    assert len(ts_loaded) == len(ts)
    for i, expected in enumerate(original_totals):
        loaded = ts_loaded[i].total_emissions
        assert set(loaded) == set(expected)
        for sp, value in expected.items():
            np.testing.assert_allclose(loaded[sp], value)


@pytest.mark.forked
def test_separate_emissions(tmp_path, sample_missions, performance_model, fuel):
    path = tmp_path / 'test_trajectories.nc'
    emissions_path = tmp_path / 'test_emissions.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    expected_totals = []
    expected_indices = []
    with TrajectoryStore.create(base_file=path) as ts:
        for mis in sample_missions:
            traj = builder.fly(performance_model, mis)
            emissions = compute_emissions(performance_model, fuel, traj)
            expected_totals.append(dict(emissions.total_emissions))
            expected_indices.append(dict(emissions.trajectory_indices))
            ts.add(traj)

    with TrajectoryStore.open(base_file=path) as ts_create_assoc:
        ts_create_assoc.create_associated(
            associated_file=emissions_path,
            fieldsets=['emissions'],
            mapping_function=functools.partial(
                compute_emissions, performance_model, fuel
            ),
        )

    with TrajectoryStore.open(
        base_file=path, associated_files=[emissions_path]
    ) as ts_read:
        assert len(ts_read) == len(sample_missions)
        assert len(ts_read.files) == 2
        assert ts_read.files[0].fieldsets == {'base'}
        assert ts_read.files[1].fieldsets == {'emissions'}
        for i in range(len(sample_missions)):
            loaded_totals = ts_read[i].total_emissions
            assert isinstance(loaded_totals, SpeciesValues)
            assert set(loaded_totals) == set(expected_totals[i])
            for sp, value in expected_totals[i].items():
                np.testing.assert_allclose(loaded_totals[sp], value)

            loaded_indices = ts_read[i].trajectory_indices
            assert isinstance(loaded_indices, SpeciesValues)
            assert set(loaded_indices) == set(expected_indices[i])
            for sp, expected in expected_indices[i].items():
                assert isinstance(loaded_indices[sp], np.ndarray)
                assert len(loaded_indices[sp]) == len(ts_read[i].fuel_flow)
                np.testing.assert_allclose(loaded_indices[sp], expected)
