# Emissions class
import numpy as np
import tomllib
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.emissions.EI_CO2 import EI_CO2
from src.emissions.EI_H2O import EI_H2O
from src.emissions.EI_SOx import EI_SOx
from src.emissions.EI_NOx import BFFM2_EINOx,NOx_speciation
from src.emissions.EI_HCCO import hccoEIsFunc
from src.emissions.EI_PMvol import EI_PMvol_NEW
from src.emissions.EI_PMnvol import PMnvol_MEEM
from src.emissions.APU_emissions import get_APU_emissions
from src.utils.standard_atmosphere import temperature_at_altitude_isa_bada4,pressure_at_altitude_isa_bada4
from src.utils.helpers import meters_to_feet
from src.utils.standard_fuel import get_thrust_cat
from src.utils.consts import kappa, R_air
class Emission:
    '''Model for determining flight emissions.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, trajectory:Trajectory,
                 EDB_data:bool = True, fuel_file:str = "./fuels/conventional_jetA.toml"):
        
        with open(fuel_file, 'rb') as f:
            self.fuel = tomllib.load(f)

        self.Ntot, self.NClm, self.NCrz, self.NDes = \
            trajectory.Ntot, trajectory.NClm, trajectory.NCrz, trajectory.NDes
        
        self.emission_indices = np.empty((), dtype=self.__emission_dtype(self.Ntot))

        traj_emissions_all = ac_performance.config['climb_descent_usage']

        # If takeoff, climb and approach calculated via performance model and NOT LTO
        # Then only do LTO calculations for taxi 
        if traj_emissions_all:
            self.LTO_emission_indices = np.empty((), dtype=self.__emission_dtype(1))
            self.LTO_emissions_g = np.empty((), dtype=self.__emission_dtype(1))
        else:
            self.LTO_emission_indices = np.empty((), dtype=self.__emission_dtype(4))
            self.LTO_emissions_g = np.empty((), dtype=self.__emission_dtype(4))

        self.APU_emission_indices = np.empty((), dtype=self.__emission_dtype(1))
        self.APU_emissions_g = np.empty((), dtype=self.__emission_dtype(1))

        self.GSE_emissions_g = np.empty((), dtype=self.__emission_dtype(1))

        self.pointwise_emissions_g = np.empty((), dtype=self.__emission_dtype(self.Ntot))
        self.summed_emission_g = np.empty((), dtype=self.__emission_dtype(1))
        
        # Fuel burn per segment
        fuel_mass = trajectory.traj_data['fuelMass']
        fuel_burn = np.zeros_like(fuel_mass)    
        fuel_burn[1:] = fuel_mass[:-1] - fuel_mass[1:]
        self.fuel_burn_per_segment = fuel_burn

        # Calculate cruise emission indices
        self.get_trajectory_emissions(trajectory, ac_performance, EDB_data=EDB_data, traj_emissions_all=traj_emissions_all)

        # Calculate LTO emissions
        self.get_LTO_emissions(ac_performance, traj_emissions_all=traj_emissions_all, pmnvol_switch_lc=ac_performance.config['pmnvol_switch_lc'])

        # Calculate APU emissions
        self.APU_emission_indices, self.APU_emissions_g = \
            get_APU_emissions(self.APU_emission_indices, self.APU_emissions_g, 
                                   self.LTO_emission_indices, ac_performance.EDB_data,
                                   self.LTO_noProp, self.LTO_no2Prop, self.LTO_honoProp)

        # Calculate GSE emissions
        self.get_GSE_emissions(ac_performance.EDB_data['WNSF'])

        # Grid emissions (todo in v1)

        # Calculate total emissions Cruise + LTO + APU + GSE 
        self.sum_total_emissions()

        # Calculate lifecycle emissions
        self.get_lifecycle_emissions(self.fuel,trajectory)
        
    def sum_total_emissions(self):

        for field in self.summed_emission_g.dtype.names:
            self.summed_emission_g[field] = np.sum(self.emission_indices[field] * self.fuel_burn_per_segment) 
            # +\
            #                     np.sum(self.LTO_emissions_g[field]) + self.APU_emissions_g[field] + self.GSE_emissions_g[field]


    def get_trajectory_emissions(self, trajectory, ac_performance, EDB_data = True, traj_emissions_all = True):

        (i_start,i_end) = (0,self.Ntot) if traj_emissions_all else (self.NClm,-self.NDes)
        # CO2
        self.emission_indices['CO2'][i_start:i_end],nvolCarbCont = EI_CO2(self.fuel)
        
        # H20
        self.emission_indices['H2O'][i_start:i_end] = EI_H2O(self.fuel)

        # SOx
        self.emission_indices['SO2'][i_start:i_end],\
        self.emission_indices['SO4'][i_start:i_end] = EI_SOx(self.fuel)
        
        # If using EDB data
        if EDB_data:
            lto_co_ei_array = np.array(ac_performance.EDB_data['CO_EI_matrix'])
            lto_hc_ei_array = np.array(ac_performance.EDB_data['HC_EI_matrix'])
            lto_nox_ei_array = np.array(ac_performance.EDB_data['NOX_EI_matrix'])
            lto_ff_array = np.array(ac_performance.EDB_data['fuelflow_KGperS'])
        else:
            # If using LTO data
            lto_co_ei_array = np.array([mode['CO_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_hc_ei_array = np.array([mode['HC_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_nox_ei_array = np.array([mode['NOX_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_ff_array = np.array([mode['FUEL_KGs'] for mode in ac_performance.LTO_data['thrust_settings'].values()])

        # NOx
        # Temperature and Pressures during flight
        flight_temps = temperature_at_altitude_isa_bada4(trajectory.traj_data['altitude'][i_start:i_end])
        flight_pressures = pressure_at_altitude_isa_bada4(trajectory.traj_data['altitude'][i_start:i_end])
        # Mach number
        mach_number = trajectory.traj_data['tas'][i_start:i_end] / np.sqrt(kappa * R_air * flight_temps)

        self.emission_indices['NOx'][i_start:i_end],self.emission_indices['NO'][i_start:i_end],\
            self.emission_indices['NO2'][i_start:i_end],self.emission_indices['HONO'][i_start:i_end],\
                _, _, _ = \
            BFFM2_EINOx(fuelflow_trajectory=trajectory.traj_data['fuelFlow'][i_start:i_end],
                   NOX_EI_matrix=lto_nox_ei_array,
                   fuelflow_performance=lto_ff_array, 
                   Pamb=flight_pressures, Tamb=flight_temps
                   )

        # HC, CO
        self.emission_indices['HC'][i_start:i_end] = hccoEIsFunc(trajectory.traj_data['fuelFlow'][i_start:i_end],lto_hc_ei_array,lto_ff_array)
        self.emission_indices['CO'][i_start:i_end] = hccoEIsFunc(trajectory.traj_data['fuelFlow'][i_start:i_end],lto_co_ei_array,lto_ff_array)

        # OC/PMvolo EIs
        thrustCat = get_thrust_cat(trajectory.traj_data['fuelFlow'][i_start:i_end],
                                    ac_performance.fuel_flow_values, cruiseCalc=True)
        self.emission_indices['PMvol'][i_start:i_end],\
        self.emission_indices['OCic'][i_start:i_end] = EI_PMvol_NEW(
                trajectory.traj_data['fuelFlow'][i_start:i_end], thrustCat
            )
        
        # BC EIs
        self.emission_indices['PMnvolGMD'][i_start:i_end],\
        self.emission_indices['PMnvol'][i_start:i_end],\
        self.emission_indices['PMnvolN'][i_start:i_end] = PMnvol_MEEM(ac_performance.EDB_data,
                            meters_to_feet(trajectory.traj_data['altitude'][i_start:i_end]),
                            flight_temps, flight_pressures, mach_number,
                            trajectory.traj_data['fuelFlow'][i_start:i_end])
        
        for field in self.pointwise_emissions_g.dtype.names:
            self.pointwise_emissions_g[field] = self.emission_indices[field] * self.fuel_burn_per_segment

    def get_LTO_emissions(self, ac_performance, traj_emissions_all = False, EDB_LTO = True, pmnvol_switch_lc = "SCOPE11"):
        
        (i_start,i_end) = (0,1) if traj_emissions_all else (0,4)
        
        # LTO Time in modes (TIM) 
        # https://www.icao.int/environmental-protection/Documents/EnvironmentalReports/2016/ENVReport2016_pg73-74.pdf 
        TIM_TakeOff = 0.7 * 60
        TIM_Climb = 2.2 * 60
        TIM_Approach = 4.0 * 60
        TIM_Taxi = 26.0 * 60
        
        if traj_emissions_all:
            TIM_LTO = np.array([TIM_Taxi])
        else:
            TIM_LTO = np.array([TIM_Taxi,TIM_Approach,TIM_Climb,TIM_TakeOff])
        fuel_flows_LTO = np.array(ac_performance.EDB_data['fuelflow_KGperS']) if EDB_LTO else np.array([mode['FUEL_KGs'] for mode in ac_performance.LTO_data['thrust_settings'].values()])

        # Calculate EIs at each LTO point

        # CO2
        self.LTO_emission_indices['CO2'],nvolCarbCont = EI_CO2(self.fuel)
        
        # H20
        self.LTO_emission_indices['H2O'] = EI_H2O(self.fuel)

        # SOx
        self.LTO_emission_indices['SO2'],\
        self.LTO_emission_indices['SO4'] = EI_SOx(self.fuel)
        
        if EDB_LTO:
            # NOx
            self.LTO_emission_indices['NOx'] = np.array(ac_performance.EDB_data['NOX_EI_matrix'][i_start:i_end])
            # HC
            self.LTO_emission_indices['HC'] = np.array(ac_performance.EDB_data['HC_EI_matrix'][i_start:i_end])
            # CO
            self.LTO_emission_indices['CO'] = np.array(ac_performance.EDB_data['CO_EI_matrix'][i_start:i_end])
        else:
            self.LTO_emission_indices['NOx'] = np.array([mode['EI_NOx'] for mode in ac_performance.LTO_data['thrust_settings'].values()][i_start:i_end])

            self.LTO_emission_indices['HC'] = np.array([mode['EI_HC'] for mode in ac_performance.LTO_data['thrust_settings'].values()][i_start:i_end])

            self.LTO_emission_indices['CO'] = np.array([mode['EI_CO'] for mode in ac_performance.LTO_data['thrust_settings'].values()][i_start:i_end])


        # OC/PMvolo EIs
        thrustCat = get_thrust_cat(fuel_flows_LTO,
                                    None, cruiseCalc=False)
        
        self.LTO_noProp, self.LTO_no2Prop, self.LTO_honoProp = NOx_speciation(thrustCat)
        LTO_PMvol, LTO_OCic = EI_PMvol_NEW(
                fuel_flows_LTO, thrustCat
            )
        self.LTO_emission_indices['PMvol'] = LTO_PMvol[i_start:i_end]
        self.LTO_emission_indices['OCic'] = LTO_OCic[i_start:i_end]

        # BC EIs

        if pmnvol_switch_lc in ('foa3', 'newsnci'):
            PMnvolEI_ICAOthrust = ac_performance.EDB_data['PMnvolEI_best_ICAOthrust']

        elif pmnvol_switch_lc in ('fox', 'dop', 'sst'):
            PMnvolEI_ICAOthrust = ac_performance.EDB_data['PMnvolEI_new_ICAOthrust']

        elif pmnvol_switch_lc == 'SCOPE11':
            PMnvolEI_ICAOthrust     = ac_performance.EDB_data['PMnvolEI_best_ICAOthrust']
            PMnvolEIN_ICAOthrust    = ac_performance.EDB_data['PMnvolEIN_best_ICAOthrust']
            PMnvolEI_lo_ICAOthrust  = ac_performance.EDB_data['PMnvolEI_lower_ICAOthrust']
            PMnvolEIN_lo_ICAOthrust = ac_performance.EDB_data['PMnvolEIN_lower_ICAOthrust']
            PMnvolEI_hi_ICAOthrust  = ac_performance.EDB_data['PMnvolEI_upper_ICAOthrust']
            PMnvolEIN_hi_ICAOthrust = ac_performance.EDB_data['PMnvolEIN_upper_ICAOthrust']

        else:
            raise ValueError(f"Re-define PMnvol estimation method: pmnvolSwitch = {pmnvol_switch_lc}")

        self.LTO_emission_indices['PMnvol'] = PMnvolEI_ICAOthrust[1:][i_start:i_end]

        if pmnvol_switch_lc == 'SCOPE11':
            self.LTO_emission_indices['PMnvol_lo'] = PMnvolEI_lo_ICAOthrust[1:][i_start:i_end]
            self.LTO_emission_indices['PMnvol_hi'] = PMnvolEI_hi_ICAOthrust[1:][i_start:i_end]

            # For number-based EI
            self.LTO_emission_indices['PMnvolN']     = PMnvolEIN_ICAOthrust[1:][i_start:i_end]
            self.LTO_emission_indices['PMnvolN_lo']  = PMnvolEIN_lo_ICAOthrust[1:][i_start:i_end]
            self.LTO_emission_indices['PMnvolN_hi']  = PMnvolEIN_hi_ICAOthrust[1:][i_start:i_end]

        # LTO Emission (g)

        # Get fuel burn at each mode
        LTO_fuel_burn = np.array([(TIM_LTO * fuel_flows_LTO)[i_start:i_end]])

        for field in self.LTO_emission_indices.dtype.names:
            self.LTO_emissions_g[field] = self.LTO_emission_indices[field] * LTO_fuel_burn

    
    def get_GSE_emissions(self, wnsf):
        # Map letter → index into nominal lists
        mapping = {'w': 0, 'n': 1, 's': 2, 'f': 3}
        i = mapping.get(wnsf.lower())
        if i is None:
            raise ValueError("Invalid WNSF code; must be one of 'w','n','s','f'")

        # Nominal EIs by category [w, n, s, f]
        CO2_nom = [58e3,   18e3,  10e3, 58e3]    # g/cycle
        NOx_nom = [0.9e3,  0.4e3, 0.3e3, 0.9e3]   # g/cycle
        HC_nom  = [0.07e3, 0.04e3,0.03e3,0.07e3]  # g/cycle (NMVOC)
        CO_nom  = [0.3e3,  0.15e3,0.1e3, 0.3e3]   # g/cycle
        PM10_nom= [0.055e3,0.025e3,0.020e3,0.055e3]# g/cycle (≈PM2.5)

        # Pick out the scalar values
        self.GSE_emissions_g['CO2'] = CO2_nom[i]
        self.GSE_emissions_g['NOx'] = NOx_nom[i]
        self.GSE_emissions_g['HC']  = HC_nom[i]
        self.GSE_emissions_g['CO']  = CO_nom[i]
        pm_core = PM10_nom[i]

        # Fuel (kg/cycle) from CO2:
        #   EI_CO2 = fuel * 3.16 * 1000  ⇒  fuel = EI_CO2/(3.16*1000)
        gse_fuel = self.GSE_emissions_g['CO2'] / (3.16 * 1000.0)

        # NOx split
        self.GSE_emissions_g['NO']   = self.GSE_emissions_g['NOx'] * 0.90
        self.GSE_emissions_g['NO2']  = self.GSE_emissions_g['NOx'] * 0.09
        self.GSE_emissions_g['HONO'] = self.GSE_emissions_g['NOx'] * 0.01

        # Sulfate / SO2 fraction (independent of WNSF)
        GSE_FSC = 5.0    # fuel‐sulfur concentration (ppm)
        GSE_EPS = 0.02   # fraction → sulfate
        # g SO4 per kg fuel:
        self.GSE_emissions_g['SO4'] = (GSE_FSC / 1e6) * 1000.0 * GSE_EPS * (96.0/32.0)
        # g SO2 per kg fuel:
        self.GSE_emissions_g['SO2'] = (GSE_FSC / 1e6) * 1000.0 * (1.0 - GSE_EPS) * (64.0/32.0)

        # Subtract sulfate from the core PM₁₀ then split 50:50
        pm_minus_so4 = pm_core - self.GSE_emissions_g['SO4']
        self.GSE_emissions_g['PMvol']   = pm_minus_so4 * 0.5
        self.GSE_emissions_g['PMnvol']   = pm_minus_so4 * 0.5

    def get_lifecycle_emissions(self, fuel, traj):
        # add lifecycle CO2 emissions for climate model run
        lc_CO2 = (fuel['LC_CO2'] * (traj.fuel_mass * fuel['Energy_MJ_per_kg'])) - self.summed_emission_g['CO2']
        self.summed_emission_g['CO2'] += lc_CO2

    ###################
    # PRIVATE METHODS #
    ###################
    def __emission_dtype(self, shape, scope11 = True):
        n = (shape,)
        return [
            ('CO2',   np.float64, n),
            ('H2O',     np.float64, n),
            ('HC',   np.float64, n),
            ('CO',   np.float64, n),
            ('NOx', np.float64, n),
            ('NO', np.float64, n),
            ('NO2', np.float64, n),
            ('HONO', np.float64, n),
            ('PMnvol',   np.float64, n),
            ('PMnvol_lo',   np.float64, n),
            ('PMnvol_hi',   np.float64, n),
            ('PMnvolN',   np.float64, n),
            ('PMnvolN_lo',   np.float64, n),
            ('PMnvolN_hi',   np.float64, n),
            ('PMnvolGMD',   np.float64, n),
            ('PMvol',   np.float64, n),
            ('OCic',   np.float64, n),
            ('SO2',  np.float64, n),
            ('SO4',  np.float64, n)
        ] if scope11 else [
            ('CO2',   np.float64, n),
            ('H2O',     np.float64, n),
            ('HC',   np.float64, n),
            ('CO',   np.float64, n),
            ('NOx', np.float64, n),
            ('NO', np.float64, n),
            ('NO2', np.float64, n),
            ('HONO', np.float64, n),
            ('PMnvol',   np.float64, n),
            ('PMnvolGMD',   np.float64, n),
            ('PMvol',   np.float64, n),
            ('OCic',   np.float64, n),
            ('SO2',  np.float64, n),
            ('SO4',  np.float64, n)
        ]

    # def __nvPM_processing(self,EDB_data,altitudes,Tamb_cruise,Pamb_cruise,machFlight,fuel_flow_cruise,pmnvolSwitch_cruise = 'SCOPE11'):
        
