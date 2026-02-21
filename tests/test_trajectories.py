import AEIC.trajectories.builders as tb
from AEIC.emissions.emission import compute_emissions
from AEIC.performance.types import ThrustMode
from AEIC.storage import Dimension, FlightPhase
from AEIC.trajectories.trajectory import BASE_FIELDS, Trajectory
from AEIC.types import Species
from AEIC.units import METERS_TO_FEET


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
