import numpy as np
from src.AEIC.performance_model import PerformanceModel
from src.utils.helpers import nautmiles_to_meters

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, mission, optimize_traj:bool, iterate_mass:bool, startMass:float=-1, 
                 max_mass_iters=5, mass_iter_reltol=1e-2):
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
        
        # Control whether or not starting mass is iterated on
        self.iter_mass = iterate_mass
        self.max_mass_iters = max_mass_iters
        self.mass_iter_reltol = mass_iter_reltol
        self.mass_converged = None
        
        # Allow user to specify starting mass if desired
        self.starting_mass = startMass
        
        # Initialize a non-reserve, non-divert/hold fuel mass for mass residual calculation
        self.fuel_mass = None
        
        # Initialize values for number of simulated points per segment (to be defined in child classes)
        self.NClm = None
        self.NCrz = None
        self.NDes = None
        self.Ntot = None
        
        # Initialize important altitudes (clm = climb, crz = cruise, des = descent)
        self.clm_start_altitude = None
        self.crz_start_altitude = None
        self.des_start_altitude = None
        self.des_end_altitude   = None
        
        
    def fly_flight(self, **kwargs):
        # Initialize data array as numpy structured array
        traj_dtype = [
            ('fuelFlow',   np.float64, self.Ntot),
            ('acMass',     np.float64, self.Ntot),
            ('fuelMass',   np.float64, self.Ntot),
            ('groundDist', np.float64, self.Ntot),
            ('altitude',   np.float64, self.Ntot),
            ('FLs',        np.float64, self.Ntot),
            ('rocs',        np.float64, self.Ntot),
            ('flightTime', np.float64, self.Ntot),
            ('latitude',   np.float64, self.Ntot),
            ('longitude',  np.float64, self.Ntot),
            ('heading',  np.float64, self.Ntot),
            ('tas',        np.float64, self.Ntot),
            ('FL_weight',  np.float64, self.Ntot),
        ]
        self.traj_data = np.empty((), dtype=traj_dtype)
        
        if self.starting_mass < 0:
            self.calc_starting_mass(**kwargs)
        
        # Trajectory optimization
        if self.optimize_traj:
            # Will be implemented in a future version
            pass
        
        if self.iter_mass:
            self.mass_converged = False
            
            for m_iter in range(self.max_mass_iters):
                mass_res = self.fly_flight_iteration(**kwargs)
                
                # Keep the calculated trajectory if the mass is sufficiently small
                if abs(mass_res) < self.mass_iter_reltol:
                    self.mass_converged = True
                    break
                
                # Perform a `dumb` correction of the starting mass
                self.starting_mass = self.starting_mass - (mass_res * self.fuel_mass)
                
            if not self.mass_converged:
                print(f'Mass iteration failed to converge; final residual {mass_res * 100}% > {self.mass_iter_reltol * 100}%')
            
        else:
            self.fly_flight_iteration(**kwargs)
        
        
        
    def fly_flight_iteration(self, **kwargs):
        ''' Function for running a single flight iteration. In non-weight-iterating mode,
        only runs once. `kwargs` used to pass in relevent optimization variables in 
        applicable cases.  
        
        Returns:
            - `mass_residual`: Difference in fuel burned and calculated required fuel mass
        '''
        self.current_mass = self.starting_mass
        
        for field in self.traj_data.dtype.names:
            self.traj_data[field][:] = np.nan
        
        # Set initial values
        self.traj_data['flightTime'][0] = 0
        self.traj_data['acMass'][0] = self.starting_mass
        self.traj_data['fuelMass'][0] = self.fuel_mass
        self.traj_data['groundDist'][0] = 0
        self.traj_data['altitude'][0] = self.clm_start_altitude
        
        # Calculate lat, lon, heading of initial point
        # TODO: Need functions for this 
        
        # Fly the climb, cruise, descent segments in order
        self.climb(**kwargs)
        self.cruise(**kwargs)
        self.descent(**kwargs)
        
        # Run LTO
        self.lto(**kwargs)
        
        # Calculate weight residual normalized by fuel_mass
        fuelBurned = self.starting_mass - self.traj_data['acMass'][-1]
        mass_residual = (self.fuel_mass - fuelBurned) / self.fuel_mass
        
        return mass_residual
        
    
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