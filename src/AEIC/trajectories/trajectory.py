import numpy as np
from src.AEIC.performance_model import PerformanceModel
from src.utils.helpers import nautmiles_to_meters

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, mission, optimize_traj:bool, iterate_mass:bool, startMass:float=-1):
        # Save A/C performance model and the mission to be flown
        # NOTE: Currently assume that `mission` comes in as a dictionary with the format of a single flight
        # in `src/missions/sample_missions_10.json`. We also assume that Load Factor for the flight will be
        # included in the mission object.
        self.name = f'{mission["dep_airport"]}_{mission["arr_airport"]}_{mission["ac_code"]}'
        self.ac_performance = ac_performance
        
        # Save airport locations and dep/arr times; lat/long in degrees
        self.dep_lon_lat_alt = mission['dep_location']
        self.arr_lon_lat_alt = mission['arr_location']
        
        self.start_time = mission["dep_datetime"]
        self.end_time = mission["arr_datetime"]
        
        # Convert gc distance to meters
        self.gc_distance = mission["distance_nm"] # FIXME: check to make sure this is changed to meters
    
        # Get load factor from mission object
        self.load_factor = mission["load_factor"]
    
        # Controls whether or not route optimization is performed
        # NOTE: This currently does nothing
        self.optimize_traj = optimize_traj
        
        # Controls whether or not starting mass is iterated on
        self.iter_mass = iterate_mass
        
        # Allow user to specify starting mass if desired
        self.starting_mass = startMass
        
        
    def fly_flight(self):
        pass
        
        
        
    def fly_flight_iteration(self, **kwargs):
        ''' Function for running a single flight iteration. In non-weight-iterating mode,
        only runs once. `kwargs` used to pass in relevent optimization variables in 
        applicable cases.      
        '''
        if self.starting_mass < 0:
            self.calc_starting_mass(**kwargs)
            
        # Create mission state arrays
        # TODO
        
        # Fly the climb, cruise, descent segments in order
        self.climb(**kwargs)
        self.cruise(**kwargs)
        self.descent(**kwargs)
        
        # Run LTO
        self.lto(**kwargs)
        
        # Calculate weight residual
        # TODO
        
        
    
    ############################################################
    # UNIVERSAL TRAJECTORY FUNCTIONS - TO BE DEFINED PER MODEL #
    ############################################################
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass
    
    
    def lto(self):
        pass
    
    
    def calc_starting_mass(self):
        pass