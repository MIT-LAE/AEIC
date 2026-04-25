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
    sample_traj = None
    for idx, mis in enumerate(sample_missions):
        traj = builder.fly(performance_model, mis)
        if sample_traj is None and mis.aircraft_type == '738':
            sample_traj = traj
        comparison_traj = comparison_ts[idx]
        if not traj.approx_eq(comparison_traj):
            failed.append(traj.name)

    comparison_ts.close()

    assert not failed, f'Trajectory simulation mismatch for: {failed}'

    # Unit-convention tripwire: the snapshot comparison only catches drift
    # from a prior SUT state — if someone changes a unit (m → km, kg → t,
    # m/s → kt) and regenerates the golden, the snapshot silently passes
    # with physically wrong numbers. Pin physical-envelope bounds on a
    # known 738 trajectory so a unit shift fails loudly here.
    assert sample_traj is not None
    assert 50_000 < float(sample_traj.aircraft_mass[0]) < 90_000  # kg
    assert 5_000 < float(sample_traj.altitude.max()) < 15_000  # m
    assert 100_000 < float(sample_traj.ground_distance.max()) < 20_000_000  # m
    assert 30 < float(sample_traj.true_airspeed.max()) < 350  # m/s
