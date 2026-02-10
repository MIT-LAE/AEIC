import AEIC.trajectories.builders as tb
from AEIC.emissions.emission import compute_emissions
from AEIC.performance.types import ThrustMode
from AEIC.types import Species


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
