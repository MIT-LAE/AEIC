# Emissions class
import numpy as np
import tomllib
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.emissions.EI_CO2 import EI_CO2
from src.emissions.EI_H2O import EI_H2O
from src.emissions.EI_SOx import EI_SOx
from src.emissions.EI_NOx import EI_NOx
from src.emissions.EI_HCCO import hccoEIsFunc
from src.utils.standard_atmosphere import temperature_at_altitude_isa_bada4,pressure_at_altitude_isa_bada4
from src.utils.consts import kappa, R_air
class Emission:
    '''Model for determining flight emissions.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, trajectory:Trajectory, mission,
                 EDB_data:bool = True, fuel_file:str = "./fuels/conventional_jetA.toml"):
        
        with open(fuel_file, 'rb') as f:
            self.fuel = tomllib.load(f)

        self.Ntot = trajectory.Ntot
        EIs_dtype = [
            ('EI_CO2',   np.float64, self.Ntot),
            ('EI_H2O',     np.float64, self.Ntot),
            ('EI_HC',   np.float64, self.Ntot),
            ('EI_CO',   np.float64, self.Ntot),
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
            ('HC',   np.float64, self.Ntot),
            ('CO',   np.float64, self.Ntot),
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

        # Temperature and Pressures during flight
        flight_temps = temperature_at_altitude_isa_bada4(trajectory.traj_data['altitude'])
        flight_pressures = pressure_at_altitude_isa_bada4(trajectory.traj_data['altitude'])

        # Mach number
        mach_number = trajectory.traj_data['tas'] / np.sqrt(kappa * R_air * flight_temps)

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
                   Pamb=flight_pressures, Tamb=flight_temps, mach_number=mach_number
                   )
        
        self.emission_g['NOx'] = self.emission_indices['EI_NOx'] * self.fuel_burn_per_segment
        self.emission_g['NO'] = self.emission_indices['EI_NO'] * self.fuel_burn_per_segment
        self.emission_g['NO2'] = self.emission_indices['EI_NO2'] * self.fuel_burn_per_segment
        self.emission_g['HONO'] = self.emission_indices['EI_HONO'] * self.fuel_burn_per_segment

        # HC, CO
        if EDB_data:
            lto_co_ei_array = np.array(ac_performance.EDB_data['CO_EI_matrix'])
            lto_hc_ei_array = np.array(ac_performance.EDB_data['HC_EI_matrix'])
            lto_ff_array = np.array(ac_performance.EDB_data['fuelflow_KGperS'])
        else:
            lto_co_ei_array = np.array([mode['CO_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_hc_ei_array = np.array([mode['HC_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_ff_array = np.array([mode['FUEL_KGs'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            

        self.emission_indices['EI_HC'] = hccoEIsFunc(trajectory.traj_data['fuelFlow'],lto_hc_ei_array,lto_ff_array)
        self.emission_indices['EI_CO'] = hccoEIsFunc(trajectory.traj_data['fuelFlow'],lto_co_ei_array,lto_ff_array)

        self.emission_g['HC'] = self.emission_indices['EI_HC'] * self.fuel_burn_per_segment
        self.emission_g['CO'] = self.emission_indices['EI_CO'] * self.fuel_burn_per_segment

        # PMvol

        # PMnvol
#         self.emission_indices['EI_PMnvol'] = EI_PMnvol(trajectory.traj_data['fuelFlow'],lto_ff_array
#     fuel_flow_flight: np.ndarray,
#     fuel_flow_performance_model: np.ndarray,
#     PMnvolEI_ICAOthrust: np.ndarray
# ) -> np.ndarray





