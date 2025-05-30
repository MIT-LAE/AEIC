import numpy as np
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.utils.helpers import feet_to_meters, meters_to_feet, nautmiles_to_meters, filter_order_duplicates
import pandas as pd

class LegacyTrajectory(Trajectory):
    '''Model for determining flight trajectories using the legacy method
    from AEIC v2.
    '''
    def __init__(self, ac_performance:PerformanceModel, mission, optimize_traj:bool, iterate_mass:bool, 
                 startMass:float=-1, pctStepClm=0.01, pctStepCrz=0.01, pctStepDes=0.01, fuel_LHV=43.8e6):
        super().__init__(ac_performance, mission, optimize_traj, iterate_mass, startMass=startMass)
        
        # Define discretization of each phase in steps as a percent of the overall distance/altitude change
        self.pctStepClm = pctStepClm
        self.pctStepCrz = pctStepCrz
        self.pctStepDes = pctStepDes
        
        self.NClm = int(1 / self.pctStepClm + 1)
        self.NCrz = int(1 / self.pctStepCrz + 1)
        self.NDes = int(1 / self.pctStepDes + 1)
        self.Ntot = self.NClm + self.NCrz + self.NDes
        
        self.fuel_LHV = fuel_LHV
        
        # Climb defined as starting 3000' above airport
        self.clm_start_altitude = self.dep_lon_lat_alt[-1] + feet_to_meters(3000.0)
        
        # Max alt should be changed to meters
        max_alt = feet_to_meters(ac_performance.model_info['General_Information']['max_alt_ft'])
        
        # Check if starting altitude is above operating ceiling; if true, set start altitude to
        # departure airport altitude
        if self.clm_start_altitude >= max_alt:
            self.clm_start_altitude = self.dep_lon_lat_alt[-1]
            
        # Cruise altitude is the operating ceiling - 7000 feet
        self.crz_start_altitude = max_alt - feet_to_meters(7000.)
        
        # Ensure cruise altitude is above the starting altitude
        if self.crz_start_altitude < self.clm_start_altitude:
            self.crz_start_altitude = self.clm_start_altitude
            
        # Prevent flying above A/C ceiling (NOTE: this will only trigger due to random 
        # variables not currently implemented)
        if self.crz_start_altitude > max_alt:
            self.crz_start_altitude = max_alt
            
        # In legacy trajectory, descent start altitude is equal to cruise altitude
        self.des_start_altitude = self.crz_start_altitude
        
        # Set descent altitude based on 3000' above arrival airport altitude; clamp to A/C operating
        # ceiling if needed
        self.des_end_altitude = self.arr_lon_lat_alt[-1] + feet_to_meters(3000.0)
        if self.des_end_altitude >= max_alt:
            self.des_end_altitude = max_alt
        
        # Save relevant flight levels
        self.crz_FL = meters_to_feet(self.crz_start_altitude) / 100
        self.clm_FL = meters_to_feet(self.clm_start_altitude) / 100
        self.des_FL = meters_to_feet(self.des_start_altitude) / 100
        self.end_FL = meters_to_feet(self.des_end_altitude)   / 100
        
        # Get the relevant bounding flight levels for cruise based on performance data
        self.__calc_crz_FLs()
        
        # Get the indices for 0-ROC performance
        self.__get_zero_roc_index()      
        
    
    def climb(self):
        ''' Function called by `fly_flight_iteration()` to simulate climb '''
        dAlt = (self.crz_start_altitude - self.clm_start_altitude) / (self.NClm - 1)
        if dAlt < 0:
            raise ValueError('Departure airport + 3000ft should not be higher than start of cruise point')
        
        alts = np.linspace(self.clm_start_altitude, self.crz_start_altitude, self.NClm)
        self.traj_data['altitude'][0:self.NClm] = alts
        
        self.__legacy_climb()
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        dAlt = (self.des_end_altitude - self.des_start_altitude) / (self.NDes)
        if dAlt > 0:
            raise ValueError('Arrival airport + 3000ft should not be higher than end of cruise point')
        
        alts = np.linspace(self.clm_start_altitude, self.crz_start_altitude, self.NClm)
        startN = self.NClm + self.NCrz
        endN = startN + self.NDes
        self.traj_data['altitude'][startN:endN] = alts
        
        self.__legacy_descent()
    
    
    def lto(self):
        pass
    
    
    def calc_starting_mass(self):
        ''' Calculates the starting mass using AEIC v2 methods. Sets both starting mass and non-reserve/hold/divert fuel mass '''
        # Use the highest value of mass per AEIC v2 method
        mass_ind = [len(self.ac_performance.performance_table_cols[-1])-1]
        crz_mass = np.array(self.ac_performance.performance_table_cols[-1])[mass_ind]
        
        subset_performance = self.ac_performance.performance_table[
            np.ix_(self.crz_FL_inds,                                           # axis 0: flight levels
                   np.arange(self.ac_performance.performance_table.shape[1]),  # axis 1: all TAS's
                   np.where(self.roc_mask)[0],                                 # axis 2: ROC ≈ 0
                   mass_ind,                                                   # axis 3: high mass value
          )
        ]
    
        non_zero_mask = np.any(subset_performance != 0, axis=(0, 2, 3))
        non_zero_perf = subset_performance[:, non_zero_mask, :, :]
        crz_tas = np.array(self.ac_performance.performance_table_cols[1])[non_zero_mask]
        
        # At this point, we should have a (2, 2, 1, 1)-shape matrix of fuel flow in (FL, TAS, --, --)
        # where there should only be two non-0 values in the FL and TAS dimensions. Isolate this matrix:
        if np.shape(non_zero_perf) != (2,2,1,1):
            raise ValueError('Performance is overdefined for legacy methods')
        
        twoByTwoPerf = non_zero_perf[:,:,0,0]
        ff_mat = twoByTwoPerf[twoByTwoPerf != 0.0]
        if np.shape(ff_mat) != (2,):
            raise ValueError(f'Mass estimation fuel flow matrix does not have the required dimensions (Expected: (2,); Recieved: {np.shape(ff_mat)})')
        
        # Now perform the necessary interpolations in TAS and fuel flow
        FL_weighting = (self.crz_FL - self.crz_FLs[0]) / (self.crz_FLs[1] - self.crz_FLs[0])
        dfuelflow = ff_mat[1] - ff_mat[0]
        dTAS = crz_tas[1] - crz_tas[0]
        
        fuelflow = ff_mat[0] + dfuelflow * FL_weighting
        tas = crz_tas[0] + dTAS * FL_weighting
        
        # Figure out startingMass components per AEIC v2:
        #
        #      |   empty weight 
        #      | + payload weight
        #      | + fuel weight
        #      | + fuel reserves weight
        #      | + fuel divert weight
        #      | + fuel hold weight
        #      | _______________________
        #        = Take-off weight
        
        # Empty mass per BADA-3 (low mass / 1.2)
        emptyMass = self.ac_performance.performance_table_cols[-1][0] / 1.2
        
        # Payload
        payloadMass = self.ac_performance.model_info['General_Information']['max_payload_kg'] * self.load_factor
        
        # Fuel Needed (distance / velocity * fuel flow rate)
        approxTime = self.gc_distance / tas
        fuelMass = approxTime * fuelflow
        
        # Reserve fuel (assumed 5%)
        reserveMass = fuelMass * 0.05
        
        # Diversion fuel per AEIC v2
        if approxTime / 60 > 180: # > 180 minutes
            divertMass = nautmiles_to_meters(200.) / tas * fuelflow
            holdMass = 30 * 60 * tas # 30 min; using cruise ff here
        else:
            divertMass = nautmiles_to_meters(100.) / tas * fuelflow
            holdMass = 45 * 60 * tas # 30 min; using cruise ff here
        
        self.starting_mass = emptyMass + payloadMass + fuelMass + reserveMass + divertMass + holdMass
        
        # Limit to MTOM if overweight
        if self.starting_mass > self.ac_performance.performance_table_cols[-1][-1]:
            self.starting_mass = self.ac_performance.performance_table_cols[-1][-1]        
        
        # Set fuel mass (for weight residual calculation)
        self.fuel_mass = fuelMass
        
        
    ###################
    # PRIVATE METHODS #
    ###################
    def __calc_crz_FLs(self):
        ''' Get the bounding cruise flight levels (for which data exists) '''
        # Get the two flight levels in data closest to the cruise FL
        self.crz_FL_inds = self.__search_flight_levels_ind(self.crz_FL)
        self.crz_FLs = np.array(self.ac_performance.performance_table_cols[0])[self.crz_FL_inds]
        
        
    def __search_flight_levels_ind(self, FL):
        ''' Searches the valid flight levels in the performance model for the indices bounding a
        known FL value.
        '''
        FL_ind_high = np.searchsorted(self.ac_performance.performance_table_cols[0], FL)
        
        if FL_ind_high == 0:
            raise ValueError(f'Aircraft is trying to fly below minimum cruise altitude (FL {FL:.2f})')
        if FL_ind_high == len(self.ac_performance.performance_table_cols[0]):
            raise ValueError(f'Aircraft is trying to fly above maximum cruise altitude (FL {FL:.2f})')
        
        FL_ind_low = FL_ind_high - 1
        FLs = [FL_ind_low, FL_ind_high]
        return FLs
    
    
    def __search_TAS_ind(self, tas):
        ''' Searches the valid tas in the performance model for the indices bounding a
        known tas value.
        '''
        tas_ind_high = np.searchsorted(self.ac_performance.performance_table_cols[1], tas)
        
        if tas_ind_high == 0:
            raise ValueError(f'Aircraft is trying to fly below minimum cruise altitude (FL {tas:.2f})')
        if tas_ind_high == len(self.ac_performance.performance_table_cols[1]):
            raise ValueError(f'Aircraft is trying to fly above maximum cruise altitude (FL {tas:.2f})')
        
        tas_ind_low = tas_ind_high - 1
        tass = [tas_ind_low, tas_ind_high]
        return tass
    
    
    def __search_mass_ind(self, mass):
        ''' Searches the valid mass values in the performance model for the indices bounding a
        known mass value.
        '''
        mass_ind_high = np.searchsorted(self.ac_performance.performance_table_cols[-1], mass)
        
        if mass_ind_high == 0:
            raise ValueError('Aircraft is trying to fly below minimum mass')
        if mass_ind_high == len(self.ac_performance.performance_table_cols[0]):
            raise ValueError('Aircraft is trying to fly above maximum mass')
        
        mass_ind_low = mass_ind_high - 1
        masses = [mass_ind_low, mass_ind_high]
        return masses
        
        
    def __get_zero_roc_index(self, roc_zero_tol=1e-6):
        ''' Get the index along the ROC axis of performance where ROC == 0 '''
        self.roc_mask = np.abs(np.array(self.ac_performance.performance_table_cols[2])) < roc_zero_tol
        
        
    def __calc_FL_interp_vals(self, i, alt, roc_perf):
        ''' Computes the state values that depend only on flight level. These include
        true airspeed (TAS) and fuel flow (ff). Rate of climb (roc) is also only
        dependent on FL in descent.
        '''
        FL = meters_to_feet(alt) / 100
        self.traj_data['FLs'][i] = FL
        FL_inds = self.__search_flight_levels_ind(FL)
        bounding_fls = np.array(self.ac_performance.performance_table_cols[0])[FL_inds]
        
        # Construct interpolation weightings
        fl_weighting = (FL - bounding_fls[0]) / (bounding_fls[1] - bounding_fls[0])
        self.traj_data['FL_weight'][i] = fl_weighting
        
        # Filter to bounding flight levels
        pos_roc_fl_reduced_perf = roc_perf[
            np.ix_( FL_inds,                          # axis 0: flight levels
                    np.arange(roc_perf.shape[1]), # axis 1: all TAS's
                    np.arange(roc_perf.shape[2]), # axis 2: all positive ROC
                    np.arange(roc_perf.shape[3]), # axis 3: mass value
            )
        ]
        
        # The the collapsed indices and values of all non-0 fuel flow and TAS values in the filtered performance data
        non_zero_ff_inds = np.nonzero(pos_roc_fl_reduced_perf)
        non_zero_ff_vals = pos_roc_fl_reduced_perf[non_zero_ff_inds]
        
        non_zero_tas_inds = non_zero_ff_inds[1]
        non_zero_tas_vals = np.array(self.ac_performance.performance_table_cols[1])[non_zero_tas_inds]
        
        # ROC will only be valid in descent
        non_zero_roc_inds = non_zero_ff_inds[2]
        non_zero_roc_vals = np.array(self.ac_performance.performance_table_cols[2])[non_zero_roc_inds]
        
        # Remove duplicate entries; we should have 2 entries in each corresponding to the two bounding flight levels
        tas_vals = filter_order_duplicates(non_zero_tas_vals)
        ff_vals = filter_order_duplicates(non_zero_ff_vals)
        
        roc_vals = filter_order_duplicates(non_zero_roc_vals)
        
        # Interpolate to get TAS and fuel flow
        if len(tas_vals) == 1:
            tas_interp = tas_vals[0]
        else:
            tas_interp = tas_vals[0] + fl_weighting * (tas_vals[1] - tas_vals[0])
            
        if len(ff_vals) == 1:
            ff_interp = ff_vals[0]
        else:
            ff_interp  = ff_vals[0]  + fl_weighting * (ff_vals[1]  - ff_vals[0])
        
        if len(roc_vals) == 1:
            roc_interp = roc_vals[0]
        else:
            roc_interp  = roc_vals[0]  + fl_weighting * (roc_vals[1]  - roc_vals[0])
        
        return tas_interp, ff_interp, roc_interp
        
        
    def __calc_roc_climb(self, i, FL, seg_start_mass, roc_perf, rocs):
        ''' Calculates rate of climb (roc) given flight level and mass given a 
        subset of overall performance data (limited to roc > 0 or roc < 0 in 
        `roc_perf`)
        '''
        # Get bounding flight levels
        FL_inds = self.__search_flight_levels_ind(FL)
        bounding_fls = np.array(self.ac_performance.performance_table_cols[0])[FL_inds]
        
        # Get bounding mass values
        mass_inds = self.__search_mass_ind(seg_start_mass)
        bounding_mass = np.array(self.ac_performance.performance_table_cols[3])[mass_inds]
        
        # Filter to bounding values
        pos_roc_reduced_perf = roc_perf[
            np.ix_( FL_inds,                          # axis 0: flight levels
                    np.arange(roc_perf.shape[1]), # axis 1: all TAS's
                    np.arange(roc_perf.shape[2]), # axis 2: all positive ROC
                    mass_inds,                        # axis 3: mass value
            )
        ]
        
        non_zero_ff_inds = np.nonzero(pos_roc_reduced_perf)
        rocs = rocs[non_zero_ff_inds[2]]
        fls = bounding_fls[non_zero_ff_inds[0]]
        masses = bounding_mass[non_zero_ff_inds[3]]
        
        # Prepare ROC matrix
        roc_mat = np.full((len(bounding_fls), len(bounding_mass)), np.nan)
        
        # Fill ROC matrix
        for kk in range(len(rocs)):
            ii = np.where(bounding_fls == fls[kk])[0][0]
            jj = np.where(bounding_mass == masses[kk])[0][0]
            roc_mat[ii, jj] = rocs[kk]
            
        fl_weight = self.traj_data['FL_weight'][i]
        mass_weight = (seg_start_mass - bounding_mass[0]) / (bounding_mass[1] - bounding_mass[0])
            
        roc_1 = roc_mat[0, 0] + (roc_mat[1,0] - roc_mat[0,0]) * fl_weight
        roc_2 = roc_mat[0, 1] + (roc_mat[1,1] - roc_mat[0,1]) * fl_weight
        
        roc = roc_1 + (roc_2 - roc_1) * mass_weight
        return roc
        
        
    def __legacy_climb(self):
        ''' Computes state over the climb segment using AEIC v2 methods based on BADA-3 formulas '''
        # Create a mask for ROC limiting to only positive values (climb)
        pos_roc_mask = np.array(self.ac_performance.performance_table_cols[2]) > 0
        
        # Convert ROC mask to the indices of positive ROC
        roc_inds = np.where(pos_roc_mask)[0]
        pos_rocs = np.array(self.ac_performance.performance_table_cols[2])[roc_inds]
        
        # Filter performance data to positive ROC
        pos_roc_perf = self.ac_performance.performance_table[
            np.ix_( np.arange(self.ac_performance.performance_table.shape[0]), # axis 0: flight levels
                    np.arange(self.ac_performance.performance_table.shape[1]), # axis 1: all TAS's
                    np.where(pos_roc_mask)[0],                                 # axis 2: all positive ROC
                    np.arange(self.ac_performance.performance_table.shape[3]), # axis 3: mass value
            )
        ]
        
        # We first compute the instantaneous data at each flight level to avoid repeat calculations. 
        # In AEIC v2 fuel flow and TAS are only dependent on flight level in climb.
        for i in range(0, self.NClm):
            alt = self.traj_data['altitude'][i]
            tas_interp, ff_interp, _ = self.__calc_FL_interp_vals(i, alt, pos_roc_perf)
            
            self.traj_data['fuelFlow'][i] = ff_interp
            self.traj_data['tas'][i] = tas_interp
            
        # Now we get rate of climb by running the flight
        for i in range(0, self.NClm-1):
            FL = self.traj_data['FLs'][i]
            tas = self.traj_data['tas'][i]
            ff = self.traj_data['fuelFlow'][i]
            seg_start_mass = self.traj_data['acMass'][i]
            
            # Calculate rate of climb
            roc = self.__calc_roc_climb(i, FL, seg_start_mass, pos_roc_perf, pos_rocs)
            self.traj_data['rocs'][i] = roc
            
            # Calculate the forward true airspeed (will be used for ground speed)
            fwd_tas = np.sqrt(tas**2 - roc**2)
            
            # Get time to complete alititude change segment and total fuel burned
            segment_time = (self.traj_data['altitude'][i+1] - self.traj_data['altitude'][i]) / roc
            segment_fuel = ff * segment_time
            
            # Get ground speed using weather module
            # TODO: Need functions for this 
            self.traj_data['latitude'] = 0
            self.traj_data['longitude'] = 0
            self.traj_data['heading'] = 0
            gspd = fwd_tas # temp value set to tas; no wind
            
            # Calculate distance along route travelled
            dist = gspd * segment_time
            
            # Account for acceleration/deceleration over the segment using end-of-segment tas
            tas_end = self.traj_data['tas'][i+1]
            kinetic_energy_chg = 1/2 * seg_start_mass * (tas_end**2 - tas**2)
            
            # Calculate fuel required for acceleration
            # NOTE: I have no idea where AEIC v2 got the efficiency of 0.15 from
            accel_fuel = kinetic_energy_chg / (self.fuel_LHV) / 0.15
            
            segment_fuel += accel_fuel
            
            # Update the state vector
            self.traj_data['fuelMass'][i+1] = self.traj_data['fuelMass'][i] - segment_fuel
            self.traj_data['acMass'][i+1] = self.traj_data['acMass'][i] - segment_fuel
            self.traj_data['groundDist'][i+1] = self.traj_data['groundDist'][i] + dist
            self.traj_data['flightTime'][i+1] = self.traj_data['flightTime'][i] + segment_time           
            
            
    def __legacy_descent(self):
        ''' Computes state over the descent segment using AEIC v2 methods based on BADA-3 formulas '''
        startN = self.NClm + self.NCrz
        endN = startN + self.NDes
        
        # Create a mask for ROC limiting to only positive values (climb)
        neg_roc_mask = np.array(self.ac_performance.performance_table_cols[2]) < 0
        
        # Convert ROC mask to the indices of positive ROC
        roc_inds = np.where(neg_roc_mask)[0]
        neg_rocs = np.array(self.ac_performance.performance_table_cols[2])[roc_inds]
        
        # Filter performance data to positive ROC
        neg_roc_perf = self.ac_performance.performance_table[
            np.ix_( np.arange(self.ac_performance.performance_table.shape[0]), # axis 0: flight levels
                    np.arange(self.ac_performance.performance_table.shape[1]), # axis 1: all TAS's
                    np.where(neg_roc_mask)[0],                                 # axis 2: all positive ROC
                    np.arange(self.ac_performance.performance_table.shape[3]), # axis 3: mass value
            )
        ]
        
        # We first compute the instantaneous data at each flight level to avoid repeat calculations. 
        # In AEIC v2 fuel flow and TAS are only dependent on flight level.
        for i in range(startN, endN):
            alt = self.traj_data['altitude'][i]
            tas_interp, ff_interp, roc_interp = self.__calc_FL_interp_vals(i, alt, neg_roc_perf)
            
            self.traj_data['fuelFlow'][i] = ff_interp
            self.traj_data['tas'][i] = tas_interp
            self.traj_data['rocs'][i] = roc_interp
                
        # Now we calculate segment level info by running the flight
        for i in range(startN, endN-1):
            tas = self.traj_data['tas'][i]
            ff = self.traj_data['fuelFlow'][i]
            roc = self.traj_data['rocs'][i]
            seg_start_mass = self.traj_data['acMass'][i]
            
            # Calculate the forward true airspeed (will be used for ground speed)
            fwd_tas = np.sqrt(tas**2 - roc**2)
            
            # Get time to complete alititude change segment and total fuel burned
            segment_time = (self.traj_data['altitude'][i+1] - self.traj_data['altitude'][i]) / roc
            segment_fuel = ff * segment_time
            
            # Get ground speed using weather module
            # TODO: Need functions for this 
            self.traj_data['latitude'] = 0
            self.traj_data['longitude'] = 0
            self.traj_data['heading'] = 0
            gspd = fwd_tas # temp value set to tas; no wind
            
            # Calculate distance along route travelled
            dist = gspd * segment_time
            
            # Account for acceleration/deceleration over the segment using end-of-segment tas
            tas_end = self.traj_data['tas'][i+1]
            kinetic_energy_chg = 1/2 * seg_start_mass * (tas_end**2 - tas**2)
            
            # Calculate fuel required for acceleration
            # NOTE: I have no idea where AEIC v2 got the efficiency of 0.15 from
            accel_fuel = kinetic_energy_chg / (self.fuel_LHV) / 0.15
            
            segment_fuel += accel_fuel
            
            # Update the state vector
            self.traj_data['fuelMass'][i+1] = self.traj_data['fuelMass'][i] - segment_fuel
            self.traj_data['acMass'][i+1] = self.traj_data['acMass'][i] - segment_fuel
            self.traj_data['groundDist'][i+1] = self.traj_data['groundDist'][i] + dist
            self.traj_data['flightTime'][i+1] = self.traj_data['flightTime'][i] + segment_time  