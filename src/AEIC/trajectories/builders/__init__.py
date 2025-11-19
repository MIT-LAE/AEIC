from abc import ABC, abstractmethod
from dataclasses import dataclass

from AEIC.performance_model import PerformanceModel
from missions import Mission

from .. import Trajectory
from ..ground_track import GroundTrack


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

    NClm: int
    """Number of trajectory points for the climb phase."""

    NCrz: int
    """Number of trajectory points for the cruise phase."""

    NDes: int
    """Number of trajectory points for the descent phase."""

    initial_altitude: float
    """Overall starting altitude for the trajectory."""

    starting_mass: float = -1.0
    """Aircraft mass at beginning of trajectory. Sentinal value (-1.0) is used
    to indicate that the starting mass should be calculated by the trajectory
    builder."""

    fuel_mass: float | None = None
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
    def Ntot(self) -> int:
        """Total number of trajectory points."""

        # Note that NClm, NCrz and NDes are defined in the context object!
        return self.NClm + self.NCrz + self.NDes

    def fly(
        self,
        ac_performance: PerformanceModel,
        mission: Mission,
        startMass: float | None = None,
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
                starting_mass=startMass if startMass is not None else -1.0,
                **kwargs,
            )

            # Allow user to specify starting mass if desired, but otherwise let
            # the trajectory builder calculate it. (Using -1.0 as a sentinel
            # value here is a little ugly, but it allows us to keep the type of
            # starting_mass in the context as float, rather than float | None.)
            if self.starting_mass == -1.0:
                self.starting_mass = self.calc_starting_mass(**kwargs)

            # Set up trajectory: all initial values are zero, but that's OK,
            # because we're going to fill those values in before we return from
            # this function, and we're only going to return this trajectory if
            # nothing goes wrong along the way. We give the trajectory a name
            # to identify it in `TrajectorySets` and intermediate NetCDF files.
            traj = Trajectory(
                self.Ntot,
                name=(
                    f'{mission.origin}_{mission.destination}_{mission.aircraft_type}'
                ),
            )

            # Do the simulation, initializing start location and azimuth from
            # ground track before we start.
            start = self.ground_track[0]
            traj.longitude[0] = start.location.longitude
            traj.latitude[0] = start.location.latitude
            traj.azimuth[0] = start.azimuth
            self._fly(traj, **kwargs)

            # If everything was OK, we return the filled-in trajectory here,
            # setting up metadata variables before we do.
            traj.starting_mass = self.starting_mass
            traj.fuel_mass = self.fuel_mass
            traj.NClm = self.NClm
            traj.NCrz = self.NCrz
            traj.NDes = self.NDes
            return traj
        finally:
            # Remove the context: this only exists during the simulation of a
            # trajectory.
            del self.ctx

    def _fly(self, traj: Trajectory, **kwargs) -> None:
        # Trajectory optimization.
        if self.options.optimize_traj:
            # Will be implemented in a future version.
            pass

        if self.options.iterate_mass:
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
                self.starting_mass = self.starting_mass - (mass_res * self.fuel_mass)

            if not mass_converged:
                raise RuntimeError(
                    "Mass iteration failed to converge; final residual"
                    f"{mass_res * 100}% > {self.options.mass_iter_reltol * 100}%"
                )

        else:
            self._fly_iteration(traj, **kwargs)

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
        traj.flightTime[0] = 0
        traj.acMass[0] = self.starting_mass
        traj.fuelMass[0] = self.fuel_mass
        traj.groundDist[0] = 0
        traj.altitude[0] = self.initial_altitude
        start = self.ground_track[0]
        traj.longitude[0] = start.location.longitude
        traj.latitude[0] = start.location.latitude
        traj.azimuth[0] = start.azimuth

        # Fly the climb, cruise, descent segments in order.
        self.climb(traj, **kwargs)
        self.cruise(traj, **kwargs)
        self.descent(traj, **kwargs)

        # Calculate weight residual normalized by fuel_mass.
        fuelBurned = self.starting_mass - traj.acMass[-1]
        mass_residual = (self.fuel_mass - fuelBurned) / self.fuel_mass

        return mass_residual

    @abstractmethod
    def climb(self, traj: Trajectory, **kwargs) -> None:
        """Simulate the climb phase of the trajectory."""
        ...

    @abstractmethod
    def cruise(self, traj: Trajectory, **kwargs) -> None:
        """Simulate the cruise phase of the trajectory."""
        ...

    @abstractmethod
    def descent(self, traj: Trajectory, **kwargs) -> None:
        """Simulate the descent phase of the trajectory."""
        ...

    @abstractmethod
    def calc_starting_mass(self, **kwargs) -> float:
        """Calculate the starting mass of the aircraft for the flight."""
        ...


from .legacy import LegacyBuilder, LegacyOptions  # noqa
from .tasopt import TASOPTBuilder, TASOPTOptions  # noqa
from .ads_b import ADSBBuilder, ADSBOptions  # noqa
from .dymos import DymosBuilder, DymosOptions  # noqa
