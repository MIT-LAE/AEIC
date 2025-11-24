from dataclasses import dataclass

from AEIC.performance_model import PerformanceModel
from missions import Mission

from .. import Trajectory
from .base import Builder, Context, Options


@dataclass
class DymosOptions: ...


class DymosContext(Context):
    def __init__(
        self,
        builder: 'DymosBuilder',
        ac_performance: PerformanceModel,
        mission: Mission,
        starting_mass: float | None,
    ):
        raise NotImplementedError('DymosContext is not yet implemented.')


class DymosBuilder(Builder):
    """Model for determining flight trajectories using ADS-B flight data. Can
    be optimized using methods defined by Marek Travnik."""

    CONTEXT_CLASS = DymosContext

    def __init__(
        self,
        options: Options = Options(),
        tasopt_options: DymosOptions = DymosOptions(),
    ):
        raise NotImplementedError('DymosBuilder is not yet implemented.')
        super().__init__(options)

    def calc_starting_mass(self, **kwargs) -> float: ...

    def climb(self, traj: Trajectory, **kwargs) -> None: ...

    def cruise(self, traj: Trajectory, **kwargs) -> None: ...

    def descent(self, traj: Trajectory, **kwargs) -> None: ...
