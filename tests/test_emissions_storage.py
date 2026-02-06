import functools

import numpy as np

import AEIC.trajectories.builders as tb
from AEIC.emissions.emission import compute_emissions
from AEIC.trajectories import TrajectoryStore
from AEIC.types import SpeciesValues


def test_emissions_storage(tmp_path, sample_missions, performance_model, fuel):
    fname = tmp_path / 'test_trajectories_with_emissions.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore.create(base_file=fname)

    for mis in sample_missions:
        traj = builder.fly(performance_model, mis)
        emissions = compute_emissions(performance_model, fuel, traj)
        traj.add_fields(emissions)
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore.open(base_file=fname)
    assert len(ts_loaded) == len(ts)
    traj = ts_loaded[0]
    assert hasattr(traj, 'total_emissions')


def test_separate_emissions(tmp_path, sample_missions, performance_model, fuel):
    path = tmp_path / 'test_trajectories.nc'
    emissions_path = tmp_path / 'test_emissions.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    with TrajectoryStore.create(base_file=path) as ts:
        for mis in sample_missions:
            traj = builder.fly(performance_model, mis)
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
            assert isinstance(ts_read[i].total_emissions, SpeciesValues)
            for sp in ts_read[i].total_emissions:
                assert isinstance(ts_read[i].total_emissions[sp], float)
            assert isinstance(ts_read[i].trajectory_indices, SpeciesValues)
            for sp in ts_read[i].trajectory_indices:
                v = ts_read[i].trajectory_indices[sp]
                assert isinstance(v, np.ndarray)
                assert len(v) == len(ts_read[i].fuel_flow)
