from dataclasses import dataclass

from AEIC.performance_model import PerformanceModel
from missions import Mission

from .. import Trajectory
from . import Builder, Context, Options


@dataclass
class TASOPTOptions: ...


class TASOPTContext(Context):
    def __init__(
        self,
        builder: 'TASOPTBuilder',
        ac_performance: PerformanceModel,
        mission: Mission,
        starting_mass: float,
    ):
        raise NotImplementedError('TASOPTContext is not yet implemented.')


class TASOPTBuilder(Builder):
    """Model for determining flight trajectories using TASOPT."""

    CONTEXT_CLASS = TASOPTContext

    def __init__(
        self,
        options: Options = Options(),
        tasopt_options: TASOPTOptions = TASOPTOptions(),
    ):
        raise NotImplementedError('TASOPTBuilder is not yet implemented.')
        super().__init__(options)

    def calc_starting_mass(self, **kwargs) -> float: ...

    def climb(self, traj: Trajectory, **kwargs) -> None: ...

    def cruise(self, traj: Trajectory, **kwargs) -> None: ...

    def descent(self, traj: Trajectory, **kwargs) -> None: ...
