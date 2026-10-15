"""Create golden test data files."""

import tomllib
from pathlib import Path

import AEIC.trajectories.builders as tb
from AEIC.commands.make_performance_model import resolve_lto
from AEIC.config import Config, config
from AEIC.missions import Mission
from AEIC.parsers.ptf_reader import PTFData
from AEIC.performance.model_builder import build_legacy_model, write_performance_model
from AEIC.performance.models import PerformanceModel
from AEIC.trajectories import TrajectoryStore

TEST_DIR = Path(__file__).parent.parent / 'tests'
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


def make_legacy_perf_toml():
    # Create the reference legacy performance toml file using sample performance
    # PTF data. If any part of the performance TOML writer is updated this
    # reference file should be re-generated.

    output_file = TEST_DATA_DIR / 'performance' / 'legacy_golden.toml'
    output_file.unlink(missing_ok=True)

    sample_ptf = TEST_DATA_DIR / 'verification' / 'legacy' / 'legacy_performance.PTF'
    sample_edb = config.file_location('engines/sample_edb.xlsx')

    lto = resolve_lto('edb', sample_edb, '01P11CM121', (0.07, 0.30, 0.85, 1.0), None)
    ptf_data = PTFData.load(str(sample_ptf))

    model = build_legacy_model(
        ptf_data,
        lto,
        aircraft_class='narrow',
        number_of_engines=2,
        maximum_payload=22422,
        apu_name='APU 131-9',
    )
    write_performance_model(output_file, model)


def run():
    make_test_trajectories()
    make_legacy_perf_toml()


if __name__ == '__main__':
    run()
