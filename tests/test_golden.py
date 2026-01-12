import AEIC.trajectories.builders as tb
from AEIC.trajectories import TrajectoryStore


def test_trajectory_simulation_golden(
    test_data_dir, sample_missions, performance_model
):
    comparison_fname = test_data_dir / 'golden/test_trajectories_golden.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    comparison_ts = TrajectoryStore.open(base_file=comparison_fname)

    for idx, mis in enumerate(sample_missions):
        traj = builder.fly(performance_model, mis)
        comparison_traj = comparison_ts[idx]
        assert traj.approx_eq(comparison_traj)

    comparison_ts.close()
