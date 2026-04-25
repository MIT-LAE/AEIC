import pytest

import AEIC.trajectories.builders as tb
from AEIC.trajectories import TrajectoryStore


@pytest.mark.forked
def test_trajectory_simulation_matches_golden_snapshot(
    test_data_dir, sample_missions, performance_model
):
    """Regression sentinel against a SUT self-snapshot.

    The golden NetCDF is produced by `scripts/make_golden_test_data.py`,
    which runs the *current* SUT and freezes its output, so this test
    only verifies non-drift from a prior SUT state — not independent
    correctness. A legitimate improvement (fixed bug, better numerics)
    will fail this test identically to a real regression and the
    expected response is to regenerate the golden, not to debug the
    SUT. Independent correctness lives elsewhere
    (`test_matlab_verification`, `test_trajectory_simulation_*`,
    `test_emission_functions.py` notebook-cited cases).
    """
    comparison_fname = test_data_dir / 'golden/test_trajectories_golden.nc'

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    comparison_ts = TrajectoryStore.open(base_file=comparison_fname)

    failed = []
    for idx, mis in enumerate(sample_missions):
        traj = builder.fly(performance_model, mis)
        comparison_traj = comparison_ts[idx]
        if not traj.approx_eq(comparison_traj):
            failed.append(traj.name)

    comparison_ts.close()

    assert not failed, f'Trajectory simulation mismatch for: {failed}'
