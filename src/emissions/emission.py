# Emissions class
import numpy as np
import tomllib
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.emissions.EI_CO2 import EI_CO2
from src.emissions.EI_H2O import EI_H2O
from src.emissions.EI_SOx import EI_SOx
from src.emissions.EI_NOx import EI_NOx

class Emission:
    '''Model for determining flight emissions.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, trajectory:Trajectory, mission,
                 fuel_file:str = "./fuels/conventional_jetA.toml"):
        
        with open(fuel_file, 'rb') as f:
            self.fuel = tomllib.load(f)

        self.Ntot = trajectory.Ntot
        EIs_dtype = [
            ('EI_CO2',   np.float64, self.Ntot),
            ('EI_H2O',     np.float64, self.Ntot),
            ('EI_HCCO',   np.float64, self.Ntot),
            ('EI_NOx', np.float64, self.Ntot),
            ('EI_NO', np.float64, self.Ntot),
            ('EI_NO2', np.float64, self.Ntot),
            ('EI_HONO', np.float64, self.Ntot),
            ('EI_PMnvol',  np.float64, self.Ntot),
            ('EI_PMvol',   np.float64, self.Ntot),
            ('EI_SO2',   np.float64, self.Ntot),
            ('EI_SO4',   np.float64, self.Ntot)
        ]
        self.emission_indices = np.empty((), dtype=EIs_dtype)

        emissions_dtype = [
            ('CO2',   np.float64, self.Ntot),
            ('H2O',     np.float64, self.Ntot),
            ('HCCO',   np.float64, self.Ntot),
            ('NOx', np.float64, self.Ntot),
            ('NO', np.float64, self.Ntot),
            ('NO2', np.float64, self.Ntot),
            ('HONO', np.float64, self.Ntot),
            ('PMnvol',   np.float64, self.Ntot),
            ('PMvol',   np.float64, self.Ntot),
            ('SO2',  np.float64, self.Ntot),
            ('SO4',  np.float64, self.Ntot)
        ]
        self.emission_g = np.empty((), dtype=emissions_dtype)


        fuel_mass = trajectory.traj_data['fuelMass']
        fuel_burn = np.zeros_like(fuel_mass)    # initialize an array of zeros, same shape as fuel_mass
        fuel_burn[1:] = fuel_mass[:-1] - fuel_mass[1:]
        self.fuel_burn_per_segment = fuel_burn

        # CO2
        self.emission_indices['EI_CO2'],_ = EI_CO2(self.fuel)
        self.emission_g['CO2'] = self.emission_indices['EI_CO2'] * self.fuel_burn_per_segment

        # H20
        self.emission_indices['EI_H2O'] = EI_H2O(self.fuel)
        self.emission_g['H2O'] = self.emission_indices['EI_H2O'] * self.fuel_burn_per_segment

        # SOx
        self.emission_indices['EI_SO2'],self.emission_indices['EI_SO4'] = EI_SOx(self.fuel)
        self.emission_g['SO2'],self.emission_g['SO4'] = (self.emission_indices['EI_SO2'] * self.fuel_burn_per_segment),\
                                                (self.emission_indices['EI_SO4'] * self.fuel_burn_per_segment)
        
        # NOx
        self.emission_indices['EI_NOx'], self.emission_indices['EI_NO'], \
        self.emission_indices['EI_NO2'], self.emission_indices['EI_HONO'], \
        noProp, no2Prop, honoProp = \
            EI_NOx(trajectory.traj_data['fuelFlow'],
                   ac_performance.fuel_flow_values, ac_performance.EI_NOx_values,
                   )
        
        self.emission_g['NOx'] = self.emission_indices['EI_NOx'] * self.fuel_burn_per_segment
        self.emission_g['NO'] = self.emission_indices['EI_NO'] * self.fuel_burn_per_segment
        self.emission_g['NO2'] = self.emission_indices['EI_NO2'] * self.fuel_burn_per_segment
        self.emission_g['HONO'] = self.emission_indices['EI_HONO'] * self.fuel_burn_per_segment


