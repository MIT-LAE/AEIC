import numpy as np
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory

class TASOPTTrajectory(Trajectory):
    '''Model for determining flight trajectories using the legacy method
    from AEIC v2.
    '''
    def __init__(self, ac_performance:PerformanceModel):
        super().__init__(ac_performance)
        
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass