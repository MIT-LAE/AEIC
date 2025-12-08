import numpy as np

import AEIC.trajectories.builders as tb
from AEIC.missions import Mission
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import FieldMetadata, FieldSet, TrajectoryStore
from AEIC.utils.files import file_location

test_fields = FieldSet(
    'test_fields',
    test_field1=FieldMetadata(
        field_type=np.float32, description='A test field 1', units='unit1'
    ),
    test_field2=FieldMetadata(
        field_type=np.int32, description='A test field 2', units='unit2'
    ),
)


def test_trajectory_simulation_1(tmp_path):
    fname = tmp_path / 'test_trajectories.nc'
    performance_model_file = file_location('IO/default_config.toml')
    missions_file = file_location('missions/sample_missions_10.toml')

    perf = PerformanceModel(performance_model_file)
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore.create(base_file=fname)
    missions = Mission.from_toml(missions_file)

    for mis in missions:
        traj = builder.fly(perf, mis)
        traj.add_fields(test_fields)
        traj.test_field1 = np.random.rand(len(traj))
        traj.test_field2 = np.random.randint(0, 100, size=len(traj))
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore.open(base_file=fname)
    assert len(ts_loaded) == len(ts)
    # TODO: Test that additional fields are correctly saved and loaded.
