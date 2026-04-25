import tomllib

import pandas as pd
import pytest

import AEIC.trajectories.builders as tb
from AEIC.emissions import compute_emissions
from AEIC.missions import Mission
from AEIC.performance.models import PerformanceModel
from AEIC.types import Fuel
from AEIC.verification.legacy import LegacyTrajectory, process_matlab_csvs
from AEIC.verification.metrics import out_of_tolerance

TRAJ_FIELDS = [
    'ground_distance',
    'latitude',
    'longitude',
    'altitude',
    'fuel_flow',
    'aircraft_mass',
    'azimuth',
    'true_airspeed',
    'rate_of_climb',
]

TRAJ_FIELD_UNITS = {
    'ground_distance': 'm',
    'altitude': 'm',
    'fuel_flow': 'kg s⁻¹',
    'aircraft_mass': 'kg',
    'true_airspeed': 'm s⁻¹',
    'rate_of_climb': 'm s⁻¹',
}

COMPARISON_FIELDS = TRAJ_FIELDS + ['trajectory_indices']

# Fields whose final-point comparison is skipped (`Trajectory.compare`
# drops the last index for these). At touchdown the legacy MATLAB
# trajectory and the new SUT diverge by construction on TAS — the
# legacy file carries an extra post-landing time point that the new
# builder doesn't synthesize the same way — so a strict per-point
# match on the last sample is a guaranteed false positive. Drop the
# tail point for `true_airspeed` only; ground-distance / altitude /
# fuel-flow agree at the endpoint and shouldn't be skipped.
SKIP_FINAL_POINT_FIELDS = {'true_airspeed'}


@pytest.mark.config_updates(use_weather=False)
def test_matlab_verification(test_data_dir) -> None:
    # Set up paths to test data.
    data_dir = test_data_dir / 'verification/legacy'
    legacy_dir = data_dir / 'matlab-output'
    missions_file = data_dir / 'missions.toml'
    fuel_file = data_dir / 'fuel.toml'
    perf_path = data_dir / 'performance-model.toml'

    # Load test data: performance model, missions, fuel file.
    pm = PerformanceModel.load(perf_path)
    with open(missions_file, 'rb') as fp:
        mission_dict = tomllib.load(fp)
    missions = Mission.from_toml(mission_dict)
    with open(fuel_file, 'rb') as fp:
        fuel = Fuel.model_validate(tomllib.load(fp))

    # Create a single trajectory builder to fly all missions.
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    failed = []

    for mission in missions:
        # Load legacy data for mission and convert to "new AEIC" format. (This
        # includes unit conversions.)
        legacy_traj_in = LegacyTrajectory(legacy_dir / f'{mission.label}.csv')
        legacy_traj = legacy_traj_in.trajectory()

        # Simulate mission and compute emissions with new AEIC.
        new_traj = builder.fly(pm, mission)
        new_traj.add_fields(compute_emissions(pm, fuel, new_traj))

        # For comparison, we do *not* interpolate the new AEIC trajectory
        # onto the same time points as the legacy trajectory. The match
        # should be close enough that we can compare corresponding points
        # along the trajectories. The number of points in the trajectories
        # should match exactly.

        # Compute comparison metrics.
        # dict[str, ComparisonMetrics | SpeciesValues[ComparisonMetrics]]
        metrics = legacy_traj.compare(
            new_traj, COMPARISON_FIELDS, SKIP_FINAL_POINT_FIELDS
        )

        # Record any metrics that are outside tolerance.
        bad_metrics = out_of_tolerance(metrics, mape_pct_tol=0.25)
        if len(bad_metrics) > 0:
            failed.append((mission.label, bad_metrics))

    if len(failed) > 0:
        print('Missions with metrics outside tolerance:')
        for mission_id, bad_metrics in failed:
            print(f'  {mission_id}:')
            for m in bad_metrics:
                print(f'    {m}')

    assert len(failed) == 0, 'Missions with metrics outside tolerance'


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_process_matlab_csvs_per_mission_split(tmp_path):
    """`process_matlab_csvs` is the entry point that turns the two raw
    MATLAB outputs (one trajectory CSV, one emissions CSV) into the
    per-mission combined files in `matlab-output/`. It is unexercised
    by any other test, so the consistency check at line 58
    (`tdf.t[:-1] != edf.t`) and the per-key groupby split would corrupt
    every downstream verification run silently if either regressed.

    Pin the happy path: two missions in the inputs, two combined CSV
    files emitted, each with the trajectory's tail-point dropped on the
    emissions side and the right `(depart, arrive, AC)` rows.
    """
    traj_csv = tmp_path / 'traj.csv'
    emis_csv = tmp_path / 'emis.csv'
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    # Two missions: BOS→LAX (738) with 3 traj points + 2 emissions
    # points (the post-landing tail point is trajectory-only), and
    # JFK→ORD (320) with 2 + 1.
    traj_rows = [
        {
            'airportDepart': 'BOS',
            'airportArrive': 'LAX',
            'AC': '738',
            't': 0.0,
            'alt': 0,
        },
        {
            'airportDepart': 'BOS',
            'airportArrive': 'LAX',
            'AC': '738',
            't': 60.0,
            'alt': 1000,
        },
        {
            'airportDepart': 'BOS',
            'airportArrive': 'LAX',
            'AC': '738',
            't': 120.0,
            'alt': 0,
        },
        {
            'airportDepart': 'JFK',
            'airportArrive': 'ORD',
            'AC': '320',
            't': 0.0,
            'alt': 0,
        },
        {
            'airportDepart': 'JFK',
            'airportArrive': 'ORD',
            'AC': '320',
            't': 60.0,
            'alt': 500,
        },
    ]
    emis_rows = [
        {
            'airportDepart': 'BOS',
            'airportArrive': 'LAX',
            'AC': '738',
            't': 0.0,
            'EI_CO2': 3.16,
        },
        {
            'airportDepart': 'BOS',
            'airportArrive': 'LAX',
            'AC': '738',
            't': 60.0,
            'EI_CO2': 3.16,
        },
        {
            'airportDepart': 'JFK',
            'airportArrive': 'ORD',
            'AC': '320',
            't': 0.0,
            'EI_CO2': 3.16,
        },
    ]
    _write_csv(traj_csv, traj_rows)
    _write_csv(emis_csv, emis_rows)

    process_matlab_csvs(traj_csv, emis_csv, out_dir)

    bos_lax = pd.read_csv(out_dir / 'BOS_LAX_738.csv')
    jfk_ord = pd.read_csv(out_dir / 'JFK_ORD_320.csv')
    # The trajectory tail-point survives on the trajectory side; the
    # merged dataframe has the trajectory length (3 / 2) with NaN on
    # the emissions side at the tail.
    assert len(bos_lax) == 3
    assert len(jfk_ord) == 2
    # Time columns survived the merge in sorted order.
    assert list(bos_lax['t']) == [0.0, 60.0, 120.0]
    assert list(jfk_ord['t']) == [0.0, 60.0]
    # Per-mission key columns are dropped from both inputs.
    for col in ('airportDepart', 'airportArrive', 'AC'):
        assert col not in bos_lax.columns
        assert col not in jfk_ord.columns
    # Both source dataframes contributed columns; tail-point EI_CO2 is
    # NaN because emissions only had two of the three trajectory points.
    assert 'alt' in bos_lax.columns
    assert 'EI_CO2' in bos_lax.columns
    assert pd.isna(bos_lax['EI_CO2'].iloc[2])
    assert bos_lax['EI_CO2'].iloc[0] == pytest.approx(3.16)


def test_process_matlab_csvs_rejects_inconsistent_time_columns(tmp_path):
    """The `(tdf.t[:-1] != edf.t).any()` consistency check must fire when
    the trajectory and emissions time columns disagree. Without this
    test, a regression that swallowed the mismatch would produce
    silently-misaligned merged CSVs.
    """
    traj_csv = tmp_path / 'traj.csv'
    emis_csv = tmp_path / 'emis.csv'
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    _write_csv(
        traj_csv,
        [
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 0.0,
                'alt': 0,
            },
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 60.0,
                'alt': 1000,
            },
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 120.0,
                'alt': 0,
            },
        ],
    )
    # Note `t=999.0` instead of `0.0` on the emissions row → mismatch.
    _write_csv(
        emis_csv,
        [
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 999.0,
                'EI_CO2': 3.16,
            },
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 60.0,
                'EI_CO2': 3.16,
            },
        ],
    )
    with pytest.raises(ValueError, match='Time columns do not match'):
        process_matlab_csvs(traj_csv, emis_csv, out_dir)


def test_process_matlab_csvs_rejects_missing_out_dir(tmp_path):
    """Output directory must exist before invocation — the function
    raises `FileNotFoundError` rather than silently creating it. Pin
    the contract.
    """
    traj_csv = tmp_path / 'traj.csv'
    emis_csv = tmp_path / 'emis.csv'
    _write_csv(
        traj_csv,
        [
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 0.0,
                'alt': 0,
            },
        ],
    )
    _write_csv(
        emis_csv,
        [
            {
                'airportDepart': 'BOS',
                'airportArrive': 'LAX',
                'AC': '738',
                't': 0.0,
                'EI_CO2': 3.16,
            },
        ],
    )
    with pytest.raises(FileNotFoundError, match='Output directory'):
        process_matlab_csvs(traj_csv, emis_csv, tmp_path / 'does_not_exist')
