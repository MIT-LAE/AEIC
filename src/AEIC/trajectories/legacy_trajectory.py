import numpy as np
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.utils.helpers import feet_to_meters, meters_to_feet

class LegacyTrajectory(Trajectory):
    '''Model for determining flight trajectories using the legacy method
    from AEIC v2.
    '''
    def __init__(self, ac_performance:PerformanceModel, mission, optimize_traj:bool, iterate_mass:bool, startMass:float=-1):
        super().__init__(ac_performance, mission, optimize_traj, iterate_mass, startMass=startMass)
        
        # Climb defined as starting 3000' above airport
        self.clm_start_altitude = self.dep_lon_lat_alt[-1] + feet_to_meters(3000.0)
        
        # Max alt should be changed to meters
        max_alt = feet_to_meters(ac_performance.model_info['General_Information']['max_alt_ft'])
        
        # Check if starting altitude is above operating ceiling; if true, set start altitude to
        # departure airport altitude
        if self.clm_start_altitude >= max_alt:
            self.clm_start_altitude = self.dep_lon_lat_alt[-1]
            
        # If flying the C172, take smaller climb steps
        if ac_performance.model_info['General_Information']['aircraft_name'] == 'C172':
            self.dAlt_climb = feet_to_meters(50.)
        else:
            self.dAlt_climb = feet_to_meters(1000.)
            
        # Cruise altitude is the operating ceiling - 7000 feet
        self.crz_altitude = max_alt - feet_to_meters(7000.)
        
        # Ensure cruise altitude is above the starting altitude
        if self.crz_altitude < self.clm_start_altitude:
            self.crz_altitude = self.clm_start_altitude
            
        # Prevent flying above A/C ceiling (NOTE: this will only trigger due to random 
        # variables not currently implemented)
        if self.crz_altitude > max_alt:
            self.crz_altitude = max_alt
        
        # Save relevant flight levels
        self.crz_FL = meters_to_feet(self.crz_altitude) / 100
        self.clm_FL = meters_to_feet(self.clm_start_altitude) / 100
        
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass
    
    
    def lto(self):
        pass
    
    
    def calc_starting_mass(self):
        '''Calculates the starting mass using AEIC v2 methods'''
        # Get the two flight levels in data closest to the cruise FL
        FL_ind_high = np.searchsorted(self.ac_performance.performance_table_cols[0], self.crz_FL)
        FL_ind_low = FL_ind_high - 1
        print(self.crz_FL)
        FL_inds = np.array([FL_ind_low, FL_ind_high])
        crz_FLs = np.array(self.ac_performance.performance_table_cols[0])[FL_inds]
        
        # Use the highest value of mass per AEIC v2 method
        mass_ind = [len(self.ac_performance.performance_table_cols[-1])-1]
        crz_mass = np.array(self.ac_performance.performance_table_cols[-1])[mass_ind]
        
        # Assume there is a region in performance data where ROC == 0 
        # indicating AEIC v2-like cruise; mask to that subset
        roc_mask = np.abs(np.array(self.ac_performance.performance_table_cols[2])) < 1e-3
        
        subset_performance = self.ac_performance.performance_table[
            np.ix_(FL_inds,                                                    # axis 0: flight levels
                   np.arange(self.ac_performance.performance_table.shape[1]),  # axis 1: all TAS's
                   np.where(roc_mask)[0],                                      # axis 2: ROC ≈ 0
                   mass_ind,                                                   # axis 3: all mass values
          )
        ]
    
        non_zero_mask = np.any(subset_performance != 0, axis=(0, 2, 3))
        non_zero_perf = subset_performance[:, non_zero_mask, :, :]
        crz_tas = np.array(self.ac_performance.performance_table_cols[1])[non_zero_mask]
    