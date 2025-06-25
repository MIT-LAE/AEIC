from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory


class ADSBTrajectory(Trajectory):
    '''Model for determining flight trajectories using ADS-B flight data. Can
    be optimized using methods defined by Marek Travnik.
    '''

    def __init__(self, ac_performance: PerformanceModel):
        super().__init__(ac_performance)

    def climb(self):
        pass

    def cruise(self):
        pass

    def descent(self):
        pass
