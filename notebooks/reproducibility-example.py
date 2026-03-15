# %load_ext autoreload
# %autoreload 2

import tomllib

import AEIC.trajectories.builders as tb
from AEIC.config import Config, config
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel
from AEIC.storage import track_file_accesses
from AEIC.trajectories import TrajectoryStore

with track_file_accesses():
    # Default configuration.
    Config.load()

    # Mission data.
    missions_file = config.file_location('missions/sample_missions_10.toml')
    with open(missions_file, 'rb') as f:
        mission_dict = tomllib.load(f)
    sample_missions = Mission.from_toml(mission_dict)

    # Performance model.
    pm_file = config.file_location('performance/sample_performance_model.toml')
    pm = PerformanceModel.load(pm_file)

    # Trajectory builder.
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    with TrajectoryStore.create(base_file='sample_trajectories.nc') as store:
        store.add_comment(
            'This is a sample trajectory store created for testing reproducibility.'
        )
        store.add_comment(
            'This store contains trajectories generated from '
            'a fixed set of missions and a performance model.'
        )
        store.add_comment(
            'Simulations were run using the LegacyBuilder with iterate_mass=False.'
        )
        for mission in sample_missions:
            trajectory = builder.fly(pm, mission)
            store.add(trajectory)


with TrajectoryStore.open(base_file='sample_trajectories.nc') as store:
    print(store.reproducibility_data)
    print(store.comments)
