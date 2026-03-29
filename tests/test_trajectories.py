import numpy as np

import AEIC.trajectories.builders as tb
from AEIC.emissions.emission import compute_emissions
from AEIC.performance.types import ThrustMode
from AEIC.storage import Dimension, FlightPhase
from AEIC.trajectories.trajectory import BASE_FIELDS, Trajectory
from AEIC.types import Species
from AEIC.units import METERS_TO_FEET
from tests.utils import make_test_trajectory


def test_trajectory_comparison(sample_missions, performance_model, fuel):
    # Create a simulated trajectory with emissions data.
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    traj = builder.fly(performance_model, sample_missions[0])
    emissions = compute_emissions(performance_model, fuel, traj)
    traj.add_fields(emissions)

    # Create a clone of the trajectory and check equality and near-equality
    # comparison.
    traj_copy = traj.copy()
    assert traj_copy == traj
    assert traj_copy.approx_eq(traj)

    # All the perturbations here follow the same pattern: make a clone, make a
    # very small change to break exact equality but not approximate equality,
    # then a larger perturbation to break approximate equality as well.
    #
    # The different perturbations are designed to test comparison of different
    # kinds of fields with different dimensions and thus different
    # representations. (The abbreviation for the relevant dimensions is show in
    # parentheses in the comments below.)
    #
    # NOTE: The ThrustModeValues class can be immutable, so we need to copy it
    # rather than trying to mutate it in place.

    # Perturb starting mass (T).
    traj_copy = traj.copy()
    traj_copy.starting_mass *= 1 + 1.0e-7
    assert traj_copy != traj
    assert traj_copy.approx_eq(traj)
    traj_copy.starting_mass *= 1.1
    assert traj_copy != traj
    assert not traj_copy.approx_eq(traj)

    # Perturb one altitude point (TP).
    traj_copy = traj.copy()
    traj_copy.altitude[20] *= 1 + 1.0e-7
    assert traj_copy != traj
    assert traj_copy.approx_eq(traj)
    traj_copy.altitude[22] *= 1.1
    assert traj_copy != traj
    assert not traj_copy.approx_eq(traj)

    # Perturb APU emissions (TS).
    traj_copy = traj.copy()
    traj_copy.apu_emissions[Species.CO2] *= 1 + 1.0e-7
    assert traj_copy != traj
    assert traj_copy.approx_eq(traj)
    traj_copy.apu_emissions[Species.CO2] *= 1.1
    assert traj_copy != traj
    assert not traj_copy.approx_eq(traj)

    # Perturb LTO emissions (TSM).
    traj_copy = traj.copy()
    tmp = traj_copy.lto_emissions[Species.CO2].copy(mutable=True)
    tmp[ThrustMode.TAKEOFF] *= 1 + 1.0e-7
    traj_copy.lto_emissions[Species.CO2] = tmp
    assert traj_copy != traj
    assert traj_copy.approx_eq(traj)
    tmp[ThrustMode.TAKEOFF] *= 1.1
    assert traj_copy != traj
    assert not traj_copy.approx_eq(traj)

    # Perturb trajectory emissions (TSP).
    traj_copy = traj.copy()
    traj_copy.trajectory_emissions[Species.CO2][20] *= 1 + 1.0e-7
    assert traj_copy != traj
    assert traj_copy.approx_eq(traj)
    traj_copy.trajectory_emissions[Species.CO2][20] *= 1.1
    assert traj_copy != traj
    assert not traj_copy.approx_eq(traj)


def test_single_point_field_set():
    base_single = BASE_FIELDS.single_point()
    fields = set(
        [f for f in BASE_FIELDS if Dimension.POINT in BASE_FIELDS[f].dimensions]
    )
    assert fields == set(base_single.fields.keys())
    assert not any(
        Dimension.POINT in base_single.fields[f].dimensions for f in base_single.fields
    )


def test_append_to_trajectory():
    # Create extensible trajectory, append points, check.
    ext_traj = Trajectory()

    def add_points(phase, start_alt, delta_alt):
        ext_traj.set_phase(phase)
        for i in range(10):
            alt = start_alt + i * delta_alt
            p = ext_traj.make_point()
            p.fuel_flow = 1.4
            p.aircraft_mass = 60000 - 1.4 * 60 * i
            p.fuel_mass = 20000 - 1.4 * 60 * i
            p.ground_distance = i * 10000
            p.altitude = alt
            p.flight_level = alt * METERS_TO_FEET / 100
            p.rate_of_climb = delta_alt / 60
            p.flight_time = i * 60
            p.latitude = 41.0 + 0.02 * i
            p.longitude = -60.0 - 0.02 * i
            p.azimuth = 135.0
            p.heading = 135.0
            p.true_airspeed = 240.0
            p.ground_speed = 240.0
            ext_traj.append(p)

    add_points(FlightPhase.CLIMB, 0, 1000)
    add_points(FlightPhase.CRUISE, 10000, 0)
    add_points(FlightPhase.DESCENT, 10000, -1000)
    ext_traj.fix()
    assert len(ext_traj) == 30
    assert ext_traj.n_climb == 10
    assert ext_traj.n_cruise == 10
    assert ext_traj.n_descent == 10


def test_dateline_splitting_no_split(performance_model, fuel):
    # Create trajectory that does not cross the dateline and check that it is
    # not split into sub-trajectories.
    traj = make_test_trajectory(5, 1)
    traj.longitude = np.array([-140.0, -139.5, -139.0, -138.5, -138.0])
    emissions = compute_emissions(performance_model, fuel, traj)
    traj.add_fields(emissions)
    sub_trajs = traj.dateline_split()
    assert len(sub_trajs) == 1
    assert sub_trajs[0].longitude.min() > -180.0


def dateline_check(st, lons, lats, alts, co2):
    assert len(st.longitude) == len(lons)
    assert len(st.latitude) == len(lats)
    assert len(st.altitude) == len(alts)
    assert len(st.trajectory_emissions[Species.CO2]) == len(co2)
    assert np.allclose(st.longitude, lons)
    assert np.allclose(st.latitude, lats)
    assert np.allclose(st.altitude, alts)
    assert np.allclose(st.trajectory_emissions[Species.CO2], co2)


def test_dateline_splitting_two_point_1(performance_model, fuel):
    # Create two-point trajectory that crosses the dateline and check that it
    # is split correctly.
    traj = make_test_trajectory(2, 1)
    traj.longitude = np.array([179.0, -179.0])
    traj.latitude = np.array([40.0, 41.0])
    traj.altitude = np.array([8000.0, 8100.0])
    traj.add_fields(compute_emissions(performance_model, fuel, traj))
    traj.trajectory_emissions[Species.CO2] = np.array([100.0, 0.0])
    sub_trajs = traj.dateline_split()

    assert len(sub_trajs) == 2
    dateline_check(
        sub_trajs[0], [179.0, 180.0], [40.0, 40.5], [8000.0, 8050.0], [50.0, 0.0]
    )
    dateline_check(
        sub_trajs[1], [-180.0, -179.0], [40.5, 41.0], [8050.0, 8100.0], [50.0, 0.0]
    )


def test_dateline_splitting_two_point_2(performance_model, fuel):
    # Create two-point trajectory that crosses the dateline and check that it
    # is split correctly (including correct proportional allocation of
    # emissions).
    traj = make_test_trajectory(2, 1)
    traj.longitude = np.array([179.5, -178.5])
    traj.latitude = np.array([40.0, 41.0])
    traj.altitude = np.array([8000.0, 8100.0])
    traj.add_fields(compute_emissions(performance_model, fuel, traj))
    traj.trajectory_emissions[Species.CO2] = np.array([100.0, 0.0])
    sub_trajs = traj.dateline_split()

    assert len(sub_trajs) == 2
    dateline_check(
        sub_trajs[0], [179.5, 180.0], [40.0, 40.25], [8000.0, 8025.0], [25.0, 0.0]
    )
    dateline_check(
        sub_trajs[1], [-180.0, -178.5], [40.25, 41.0], [8025.0, 8100.0], [75.0, 0.0]
    )


def test_dateline_splitting_degenerate_1(performance_model, fuel):
    # Check on degenerate trajectory with three points where middle point is
    # exactly on the dateline.
    traj = make_test_trajectory(3, 1)
    traj.longitude = np.array([179.0, 180.0, -179.0])
    traj.latitude = np.array([40.0, 40.5, 41.0])
    traj.altitude = np.array([8000.0, 8050.0, 8100.0])
    traj.add_fields(compute_emissions(performance_model, fuel, traj))
    traj.trajectory_emissions[Species.CO2] = np.array([50.0, 50.0, 0.0])
    sub_trajs = traj.dateline_split()

    assert len(sub_trajs) == 2
    dateline_check(
        sub_trajs[0], [179.0, 180.0], [40.0, 40.5], [8000.0, 8050.0], [50.0, 0.0]
    )
    dateline_check(
        sub_trajs[1], [-180.0, -179.0], [40.5, 41.0], [8050.0, 8100.0], [50.0, 0.0]
    )


def test_dateline_splitting_2(performance_model, fuel):
    # Create trajectory that crosses the dateline and check that it is split into
    # two sub-trajectories with points in the correct hemispheres.
    traj = make_test_trajectory(5, 1)
    traj.longitude = np.array([179.0, 179.75, -179.5, -178.75, -178.0])
    emissions = compute_emissions(performance_model, fuel, traj)
    traj.add_fields(emissions)
    sub_trajs = traj.dateline_split()
    assert len(sub_trajs) == 2
    assert np.all(sub_trajs[0].longitude > 0.0)
    assert sub_trajs[0].longitude.max() == 180.0
    assert np.all(sub_trajs[1].longitude < 0.0)
    assert sub_trajs[1].longitude.min() == -180.0


def test_dateline_splitting_4(performance_model, fuel):
    # Create trajectory that repeatedly crosses the dateline and check that it
    # is split into four sub-trajectories with the correct points.
    traj = make_test_trajectory(18, 1)
    traj.latitude = np.array([40 + i * 0.1 for i in range(18)])
    traj.longitude = np.array(
        [
            178.5,
            179.0,
            179.5,
            180.0,
            -179.5,
            -179.0,
            -178.75,
            -178.75,
            -179.0,
            -179.5,
            -180.0,
            -179.5,
            180.0,
            -179.5,
            -179.0,
            -178.5,
            -178.0,
            -177.5,
        ]
    )
    emissions = compute_emissions(performance_model, fuel, traj)
    traj.add_fields(emissions)
    sub_trajs = traj.dateline_split()
    assert len(sub_trajs) == 4
    assert np.all(sub_trajs[0].longitude > 0.0)
    assert sub_trajs[0].longitude.max() == 180.0
    assert np.all(sub_trajs[1].longitude < 0.0)
    assert sub_trajs[1].longitude.min() == -180.0
    assert np.all(sub_trajs[2].longitude > 0.0)
    assert sub_trajs[2].longitude.min() == 180.0
    assert np.all(sub_trajs[3].longitude < 0.0)
    assert sub_trajs[3].longitude.min() == -180.0
