from abc import ABC, abstractmethod
from dataclasses import dataclass

from AEIC.missions import Mission
from AEIC.performance_model import PerformanceModel

from ..ground_track import GroundTrack
from ..phase import FlightPhase
from ..trajectory import Trajectory


@dataclass
class Options:
    """Common options for trajectory builders."""

    optimize_traj: bool = False
    """(Currently unimplemented) Flag controlling whether the nominal
    trajectory undergoes horizontal, vertical, and speed optimization during
    simulation."""

    iterate_mass: bool = True
    """Flag controlling whether starting mass is iterated on such that the
    remaining fuel (non-reserve) is close to 0 at the arrival airport."""

    use_weather: bool = False
    """Whether to use wind data for ground-speed calculations."""

    max_mass_iters: int = 5
    """Maximum number of mass iterations (if used). Defaults to 5."""

    mass_iter_reltol: float = 1e-2
    """Desired relative tolerance for mass iteration. Defaults to 1e-2."""


@dataclass
class Context:
    """Transient context used during trajectory building.

    An object of this class is created at the beginning of each call to the
    `fly` method of a trajectory builder. Builder-specific context classes are
    derived from this class for each of the concrete trajectory builder
    classes.
    """

    builder: 'Builder'
    """Trajectory builder instance being used for this simulation."""

    ac_performance: PerformanceModel
    """Performance model used for trajectory simulation."""

    mission: Mission
    """Mission data (departure/arrival info, etc.)."""

    ground_track: GroundTrack
    """Ground track for the trajectory."""

    npoints: dict[FlightPhase, int]
    """The number of trajectory points in each flight phase."""

    initial_altitude: float
    """Overall starting altitude for the trajectory."""

    starting_mass: float | None = None
    """Aircraft mass at beginning of trajectory. If not provided, the starting
    mass should be calculated by the trajectory builder."""

    total_fuel_mass: float | None = None
    """Total fuel mass loaded onto the aircraft. Initialized as a non-reserve,
    non-divert/hold fuel mass for mass residual calculation."""


class Builder(ABC):
    """Abstract parent class for all AEIC trajectory builders. Contains overall
    `fly` logic.

    Attributes:
        options (Options): Options for trajectory building.
    """

    CONTEXT_CLASS: type | None = None
    """Class for trajectory building context. Should be derived from base
    Context class."""

    def __init__(self, options: Options = Options(), *args, **kwargs) -> None:
        """Initialize trajectory builder with common options."""

        # Check that the context class has been defined properly in the
        # concrete derived class that we're trying to instantiate.
        if self.CONTEXT_CLASS is None:
            raise ValueError('CONTEXT_CLASS must be set in derived Builder class')

        # Save the options: these are common between all trajectory
        # simulations using this trajectory builder.
        self.options = options

    def __getattr__(self, name: str):
        """Custom attribute getter to allow access to context attributes.

        Allow access to context attributes directly from builder. During
        trajectory simulation, we create a context object (self.ctx), which is
        of a class derived from the base Context class. This object carries all
        of the transient information we need to keep track of during trajectory
        simulation. To avoid having to write `self.ctx` everywhere we want to
        access this information, we redirect attribute accesses to retrieve
        context information as required.
        """
        try:
            # Use __getattribute__ here to avoid infinite recursion.
            ctx = self.__getattribute__('ctx')
            if hasattr(ctx, name):
                return getattr(ctx, name)
        except AttributeError:
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """Custom attribute setter to allow access to context attributes.

        The implementation of `__setattr__` only sets *existing* attributes on
        the context. Adding new attributes to the context has to be done
        explicitly! This seems like the minimally confusing and maximally
        convenient behavior here.
        """
        try:
            # Use __getattribute__ here to avoid mutual recursion with
            # __getattr__.
            ctx = self.__getattribute__('ctx')
            if hasattr(ctx, name):
                return setattr(ctx, name, value)
        except AttributeError:
            pass

        # For attributes not in the context, use default behavior.
        return super().__setattr__(name, value)

    @property
    def n_total(self) -> int:
        """Total number of trajectory points."""

        # Sum counts of trajectory points across all flight phases. Note that
        # npoints is defined in the context object!
        return sum(self.npoints.values())

    # TODO: Define methods for other phases? Or just use self.npoints[phase]?
    # Maybe leave it until we actually have models that model those phases.

    @property
    def n_climb(self) -> int:
        """Number of trajectory points in climb phase."""
        return self.npoints[FlightPhase.CLIMB]

    @property
    def n_cruise(self) -> int:
        """Number of trajectory points in cruise phase."""
        return self.npoints[FlightPhase.CRUISE]

    @property
    def n_descent(self) -> int:
        """Number of trajectory points in descent phase."""
        return self.npoints[FlightPhase.DESCENT]

    def fly(
        self,
        ac_performance: PerformanceModel,
        mission: Mission,
        starting_mass: float | None = None,
        **kwargs,
    ) -> Trajectory:
        """Top-level function that initiates and runs flights.

        As well as the fixed arguments listed below, this can also take
        builder-specific additional keyword arguments.

        Args:
            ac_performance (PerformanceModel): Performance model used for
                trajectory simulation.
            mission (Mission): Data dictating the mission to be flown
                (departure/arrival info, etc.).
            startMass (float, optional): Starting mass of the aircraft; leave as
                default to calculate starting mass during simulation.

        Returns:
            trajectory (Trajectory): Trajectory object.
        """

        # This is a wrapper method to ensure that the simulation context gets
        # initialized and de-initialized properly. The approach used here is
        # predicated on using a trajectory builder instance to process multiple
        # flights in sequence by calling `fly` multiple times. Each call to
        # `fly` is independent, but the builder maintains state throughout the
        # lifetime of the call to `fly` in a transient context variable. The
        # wrapper here manages the lifetime of that context.

        try:
            # Create simulation context. This is of a class derived from the
            # base Context class that depends on the exact trajectory builder
            # being used. Different trajectory builders will have different
            # context requirements.
            assert self.CONTEXT_CLASS is not None
            self.ctx = self.CONTEXT_CLASS(
                builder=self,
                ac_performance=ac_performance,
                mission=mission,
                starting_mass=starting_mass,
                **kwargs,
            )

            # Allow user to specify starting mass if desired, but otherwise let
            # the trajectory builder calculate it.
            if self.starting_mass is None:
                self.starting_mass = self.calc_starting_mass(**kwargs)
            assert self.starting_mass is not None

            # Set up trajectory: all initial values are zero, but that's OK,
            # because we're going to fill those values in before we return from
            # this function, and we're only going to return this trajectory if
            # nothing goes wrong along the way. We give the trajectory a name
            # to identify it in `TrajectoryStore`s and intermediate NetCDF
            # files.
            traj = Trajectory(
                self.n_total,
                name=(
                    f'{mission.origin}_{mission.destination}_{mission.aircraft_type}'
                ),
            )

            # Do the simulation...

            # Trajectory optimization.
            if self.options.optimize_traj:
                raise NotImplementedError(
                    "Trajectory optimization is not yet implemented."
                )

            if self.options.iterate_mass:
                # Iterate on starting mass to minimize mass residual.
                self._iterate_mass(traj, **kwargs)
            else:
                # Otherwise, just fly a single iteration with the given starting
                # mass.
                self._fly_iteration(traj, **kwargs)

            # If everything was OK, we return the filled-in trajectory here,
            # setting up metadata variables before we do.
            traj.starting_mass = self.starting_mass
            traj.total_fuel_mass = self.total_fuel_mass
            if mission.flight_id is not None:
                traj.flight_id = mission.flight_id
            # TODO: This doesn't seem right. Flying the trajectory should
            # result in a trajectory with the right numbers of points in each
            # phase! You shouldn't need to set these things here.
            traj.n_climb = self.n_climb
            traj.n_cruise = self.n_climb
            traj.n_descent = self.n_climb
            return traj
        finally:
            # Remove the context: this only exists during the simulation of a
            # trajectory.
            del self.ctx

    def _iterate_mass(self, traj: Trajectory, **kwargs) -> None:
        """Iterate on starting mass to minimize residual fuel mass."""
        mass_converged = False
        mass_res = 0

        for _ in range(self.options.max_mass_iters):
            mass_res = self._fly_iteration(traj, **kwargs)

            # Keep the calculated trajectory if the mass is sufficiently
            # small.
            if abs(mass_res) < self.options.mass_iter_reltol:
                mass_converged = True
                break

            # Perform a "dumb" correction of the starting mass.
            self.starting_mass = self.starting_mass - (mass_res * self.total_fuel_mass)

        if not mass_converged:
            raise RuntimeError(
                "Mass iteration failed to converge; final residual"
                f"{mass_res * 100}% > {self.options.mass_iter_reltol * 100}%"
            )

    def _fly_iteration(self, traj: Trajectory, **kwargs):
        """Run a single flight iteration. In non-weight-iterating mode, only
        runs once. `kwargs` used to pass in relevent optimization variables in
        applicable cases.

        Args:
            kwargs: Additional parameters needed by the specific type of
                trajectory builder being used.

        Returns:
            (float) Difference in fuel burned and calculated required fuel
                mass.
        """

        self.current_mass = self.starting_mass

        # Set initial values, taking initial position and azimuth from ground
        # track.
        start = self.ground_track[0]
        traj.longitude[0] = start.location.longitude
        traj.latitude[0] = start.location.latitude
        traj.azimuth[0] = start.azimuth
        traj.altitude[0] = self.initial_altitude
        traj.flight_time[0] = 0
        traj.ground_distance[0] = 0
        traj.aircraft_mass[0] = self.starting_mass
        traj.fuel_mass[0] = self.total_fuel_mass

        # Fly the flight segments in order (normally just climb, cruise,
        # descent phases).
        for phase in FlightPhase:
            if hasattr(self, phase.method_name):
                getattr(self, phase.method_name)(traj, **kwargs)

        # Calculate weight residual normalized by total_fuel_mass.
        fuelBurned = self.starting_mass - traj.aircraft_mass[-1]
        mass_residual = (self.total_fuel_mass - fuelBurned) / self.total_fuel_mass

        return mass_residual

    @abstractmethod
    def calc_starting_mass(self, **kwargs) -> float:
        """Calculate the starting mass of the aircraft for the flight."""
        ...
