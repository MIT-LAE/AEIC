from dataclasses import dataclass

from AEIC.performance_model import PerformanceModel
from missions import Mission

from .. import Trajectory
from .base import Builder, Context, Options


@dataclass
class ADSBOptions: ...


class ADSBContext(Context):
    def __init__(
        self,
        builder: 'ADSBBuilder',
        ac_performance: PerformanceModel,
        mission: Mission,
        starting_mass: float | None,
    ):
        raise NotImplementedError('ADSBContext is not yet implemented.')


class ADSBBuilder(Builder):
    """Model for determining flight trajectories using ADS-B data."""

    CONTEXT_CLASS = ADSBContext

    def __init__(
        self, options: Options = Options(), tasopt_options: ADSBOptions = ADSBOptions()
    ):
        raise NotImplementedError('ADSBBuilder is not yet implemented.')
        super().__init__(options)

    def calc_starting_mass(self, **kwargs) -> float: ...

    def climb(self, traj: Trajectory, **kwargs) -> None: ...

    def cruise(self, traj: Trajectory, **kwargs) -> None: ...

    def descent(self, traj: Trajectory, **kwargs) -> None: ...
