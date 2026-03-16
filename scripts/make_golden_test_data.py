"""Create golden test data files."""

import tomllib
from pathlib import Path

import AEIC.trajectories.builders as tb
from AEIC.config import Config, config
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel
from AEIC.trajectories import TrajectoryStore

TEST_DIR = Path(__file__).parent.parent.parent.parent / 'tests'
TEST_DATA_DIR = TEST_DIR / 'data'

# We need to add the test data directory as a data path override so that the
# golden test data is created with stable inputs. In particular, in normal
# operation airports data is downloaded from an external source, and this can
# change!
Config.load(data_path_overrides=[TEST_DATA_DIR])


def make_test_trajectories():
    # Create test trajectories for sample missions using sample performance
    # model and save to a NetCDF file.

    performance_model = PerformanceModel.load(
        config.file_location('performance/sample_performance_model.toml')
    )

    missions_file = config.file_location('missions/sample_missions_10.toml')
    with open(missions_file, 'rb') as f:
        mission_dict = tomllib.load(f)
    sample_missions = Mission.from_toml(mission_dict)

    output_file = TEST_DATA_DIR / 'golden/test_trajectories_golden.nc'
    output_file.unlink(missing_ok=True)
    ts = TrajectoryStore.create(base_file=output_file)

    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    for mis in sample_missions:
        traj = builder.fly(performance_model, mis)
        ts.add(traj)

    ts.close()


def run():
    make_test_trajectories()


if __name__ == '__main__':
    run()
