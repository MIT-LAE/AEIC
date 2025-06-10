# Emissions class
import numpy as np
import tomllib
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.emissions.EI_CO2 import EI_CO2
from src.emissions.EI_H2O import EI_H2O
from src.emissions.EI_SOx import EI_SOx
from src.emissions.EI_NOx import EI_NOx,BFFM2_EINOx
from src.emissions.EI_HCCO import hccoEIsFunc
from src.emissions.EI_PMvol import EI_PMvol_NEW, EI_PMvol_FOA3
from src.emissions.EI_PMnvol import EI_PMnvol, EI_PMnvolN
from src.utils.standard_atmosphere import temperature_at_altitude_isa_bada4,pressure_at_altitude_isa_bada4
from src.utils.helpers import meters_to_feet
from src.utils.standard_fuel import get_fuel_factor, get_thrust_cat
from src.utils.consts import kappa, R_air
class Emission:
    '''Model for determining flight emissions.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, trajectory:Trajectory, mission,
                 EDB_data:bool = True, fuel_file:str = "./fuels/conventional_jetA.toml"):
        
        with open(fuel_file, 'rb') as f:
            self.fuel = tomllib.load(f)

        self.Ntot, self.NClm, self.NCrz, self.NDes = \
            trajectory.Ntot, trajectory.NClm, trajectory.NCrz, trajectory.NDes
        
        self.emission_indices = np.empty((), dtype=self.__emission_dtype(self.Ntot))

        self.LTO_emission_indices = np.empty((), dtype=self.__emission_dtype(11))

        self.APU_emission_indices = np.empty((), dtype=self.__emission_dtype(1))
        self.APU_emissions_g = np.empty((), dtype=self.__emission_dtype(1))

        self.GSE_emissions_g = np.empty((), dtype=self.__emission_dtype(1))

        self.pointwise_emissions_g = np.empty((), dtype=self.__emission_dtype(self.Ntot))

        self.emission_g = np.empty((), dtype=self.__emission_dtype(1))
        
        # Fuel burn per segment
        fuel_mass = trajectory.traj_data['fuelMass']
        fuel_burn = np.zeros_like(fuel_mass)    
        fuel_burn[1:] = fuel_mass[:-1] - fuel_mass[1:]
        self.fuel_burn_per_segment = fuel_burn

        # Calculate cruise emission indices
        self.cruise_emissions(trajectory, ac_performance)

        # Calculate LTO emissions
        self.LTO_emissions(ac_performance)

        # Calculate APU emissions
        self.APU_emissions(ac_performance.EDB_data)

        # Calculate GSE emissions
        self.GSE_emissions(ac_performance.EDB_data['WNSF'])

        # Grid emissions (todo in v1)

        # Calculate total emissions (For now just cruise) Will add LTO + APU + GSE when TIMs resolved
        self.total_sum_emissions()

        # Calculate lifecycle emissions
        self.lifecycle_emissions(self.fuel,trajectory)
        
    def total_sum_emissions(self, pmnvolSwitch = "SCOPE11"):
        # CO2
        self.emission_g['CO2'] = np.sum(self.emission_indices['CO2'] * self.fuel_burn_per_segment)

        # H20
        self.emission_g['H2O'] = np.sum(self.emission_indices['H2O'] * self.fuel_burn_per_segment)

        # SOx
        self.emission_g['SO2'],self.emission_g['SO4'] = np.sum(self.emission_indices['SO2'] * self.fuel_burn_per_segment),\
                                                np.sum(self.emission_indices['SO4'] * self.fuel_burn_per_segment)
        
        # NOx
        self.emission_g['NOx'] = np.sum(self.emission_indices['NOx'] * self.fuel_burn_per_segment)
        self.emission_g['NO'] = np.sum(self.emission_indices['NO'] * self.fuel_burn_per_segment)
        self.emission_g['NO2'] = np.sum(self.emission_indices['NO2'] * self.fuel_burn_per_segment)
        self.emission_g['HONO'] = np.sum(self.emission_indices['HONO'] * self.fuel_burn_per_segment)

        # HC, CO
        self.emission_g['HC'] = np.sum(self.emission_indices['HC'] * self.fuel_burn_per_segment)
        self.emission_g['CO'] = np.sum(self.emission_indices['CO'] * self.fuel_burn_per_segment)

        # PMvol, OCic
        self.emission_g['PMvol'] = np.sum(self.emission_indices['PMvol'] * self.fuel_burn_per_segment)
        self.emission_g['OCic'] = np.sum(self.emission_indices['OCic'] * self.fuel_burn_per_segment)

        # PMnvol, add PMnvolN, PMnvolGMD if SCOPE11
        self.emission_g['PMnvol'] = np.sum(self.emission_indices['PMnvol'] * self.fuel_burn_per_segment)
        if pmnvolSwitch == "SCOPE11":
            self.emission_g['PMnvolN'] = np.sum(self.emission_indices['PMnvolN'] * self.fuel_burn_per_segment)
            self.emission_g['PMnvolGMD'] = np.sum(self.emission_indices['PMnvolGMD'] * self.fuel_burn_per_segment)


    def cruise_emissions(self, trajectory, ac_performance, EDB_data = True):

        # CO2
        self.emission_indices['CO2'][self.NClm:-self.NDes],nvolCarbCont = EI_CO2(self.fuel)
        
        # H20
        self.emission_indices['H2O'][self.NClm:-self.NDes] = EI_H2O(self.fuel)

        # SOx
        self.emission_indices['SO2'][self.NClm:-self.NDes],\
        self.emission_indices['SO4'][self.NClm:-self.NDes] = EI_SOx(self.fuel)
        
        # NOx
        # Temperature and Pressures during flight
        flight_temps = temperature_at_altitude_isa_bada4(trajectory.traj_data['altitude'][self.NClm:-self.NDes])
        flight_pressures = pressure_at_altitude_isa_bada4(trajectory.traj_data['altitude'][self.NClm:-self.NDes])
        # Mach number
        mach_number = trajectory.traj_data['tas'][self.NClm:-self.NDes] / np.sqrt(kappa * R_air * flight_temps)

        # TODO: Check fuel factor ... and also check EDB ff if only 1 engine or both
        self.emission_indices['NOx'][self.NClm:-self.NDes],self.emission_indices['NO'][self.NClm:-self.NDes],\
            self.emission_indices['NO2'][self.NClm:-self.NDes],self.emission_indices['HONO'][self.NClm:-self.NDes],\
                noProp, no2Prop, honoProp = \
            BFFM2_EINOx(fuelflow_trajectory=trajectory.traj_data['fuelFlow'][self.NClm:-self.NDes],
                   NOX_EI_matrix=np.array(ac_performance.EDB_data['NOX_EI_matrix']),
                   fuelflow_performance=np.array(ac_performance.EDB_data['fuelflow_KGperS']), 
                   Pamb=flight_pressures, Tamb=flight_temps
                   )

        # HC, CO
        # If using EDB data
        if EDB_data:
            lto_co_ei_array = np.array(ac_performance.EDB_data['CO_EI_matrix'])
            lto_hc_ei_array = np.array(ac_performance.EDB_data['HC_EI_matrix'])
            lto_ff_array = np.array(ac_performance.EDB_data['fuelflow_KGperS'])
        else:
            # If using LTO data
            lto_co_ei_array = np.array([mode['CO_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_hc_ei_array = np.array([mode['HC_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_ff_array = np.array([mode['FUEL_KGs'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            

        self.emission_indices['HC'][self.NClm:-self.NDes] = hccoEIsFunc(trajectory.traj_data['fuelFlow'][self.NClm:-self.NDes],lto_hc_ei_array,lto_ff_array)
        self.emission_indices['CO'][self.NClm:-self.NDes] = hccoEIsFunc(trajectory.traj_data['fuelFlow'][self.NClm:-self.NDes],lto_co_ei_array,lto_ff_array)

        # OC/PMvolo EIs
        thrustCat = get_thrust_cat(trajectory.traj_data['fuelFlow'][self.NClm:-self.NDes],
                                    ac_performance.fuel_flow_values, cruiseCalc=True)
        self.emission_indices['PMvol'][self.NClm:-self.NDes],\
        self.emission_indices['OCic'][self.NClm:-self.NDes] = EI_PMvol_NEW(
                trajectory.traj_data['fuelFlow'][self.NClm:-self.NDes], thrustCat
            )
        
        # BC EIs
        self.emission_indices['PMnvolGMD'][self.NClm:-self.NDes],\
        self.emission_indices['PMnvol'][self.NClm:-self.NDes],\
        self.emission_indices['PMnvolN'][self.NClm:-self.NDes] = self.__nvPM_processing(ac_performance.EDB_data,
                            meters_to_feet(trajectory.traj_data['altitude'][self.NClm:-self.NDes]),
                            flight_temps, flight_pressures, mach_number,
                            trajectory.traj_data['fuelFlow'][self.NClm:-self.NDes])

    def LTO_emissions(self, ac_performance, EDB_LTO = True, pmnvol_switch_lc = "SCOPE11"):
        NATSGRP = np.array(ac_performance.EDB_data['NATSGRP'])
        N = NATSGRP.size
        # Get thrust levels at 11 LTO points
        taxi_in   = 5.5
        taxi_out  = taxi_in
        hold      = taxi_out
        landing   = taxi_in
        taxi_acc  = 10.0
        takeoff   = 100.0
        inclimb   = takeoff
        climb_out = 85.0
        approach  = 30.0

        #TODO Lookup reverse thrust per NATS group
        reverse = 0.50#np.asarray(thrusts_reverseNats)[NATSGRP]

        # Build the base thrust matrix (N rows × 11 modes)
        thrusts = np.column_stack([
            np.full(N, taxi_out),    # 0: TaxiOut
            np.full(N, taxi_acc),    # 1: TaxiAccDecel
            np.full(N, hold),        # 2: Hold
            np.full(N, takeoff),     # 3: TakeOff
            np.full(N, inclimb),     # 4: InClimb
            np.full(N, climb_out),   # 5: ClimbOut
            np.full(N, approach),    # 6: Approach
            np.full(N, landing),     # 7: Landing
            np.full(N, reverse),     # 8: Reverse
            np.full(N, taxi_in),     # 9: TaxiIn
            np.full(N, taxi_acc),    # 10: TaxiAccel
        ])

        if NATSGRP >=7:
            # For “large jets” (NATSGRP >= 7), enforce maximum thrust in T/O & InClimb,
            # and reset ClimbOut to 85%
            mask = (NATSGRP >= 7)
            thrusts[mask, 3] = 100.0              # TakeOff → 100%
            thrusts[mask, 4] = thrusts[mask, 3]   # InClimb = same as TakeOff
            thrusts[mask, 5] = 85.0               # ClimbOut → 85%


        # Linerarly interpolate to get fuel flows at LTO points
        thurst_levels_4 = np.array([7, 30, 85, 100])
        fuel_flows_4 = np.array(ac_performance.EDB_data['fuelflow_KGperS']) if EDB_LTO else np.array([mode['FUEL_KGs'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
        fuel_flows_LTO = np.interp(thrusts, thurst_levels_4, fuel_flows_4)[0]

        # Calculate EIs at each LTO point

        # CO2
        self.LTO_emission_indices['CO2'],nvolCarbCont = EI_CO2(self.fuel)
        
        # H20
        self.LTO_emission_indices['H2O'] = EI_H2O(self.fuel)

        # SOx
        self.LTO_emission_indices['SO2'],\
        self.LTO_emission_indices['SO4'] = EI_SOx(self.fuel)
        
        # NOx

        # TODO: Check fuel factor ... and also check EDB ff if only 1 engine or both
        self.LTO_emission_indices['NOx'],self.LTO_emission_indices['NO'],\
            self.LTO_emission_indices['NO2'],self.LTO_emission_indices['HONO'],\
                self.LTO_noProp, self.LTO_no2Prop, self.LTO_honoProp = \
            BFFM2_EINOx(fuelflow_trajectory=fuel_flows_LTO,
                   NOX_EI_matrix=np.array(ac_performance.EDB_data['NOX_EI_matrix']),
                   fuelflow_performance=np.array(ac_performance.EDB_data['fuelflow_KGperS']), 
                   Pamb=np.empty_like(fuel_flows_LTO), Tamb=np.empty_like(fuel_flows_LTO), cruiseCalc=False
                   )
        # HC, CO
        # If using EDB data
        if EDB_LTO:
            lto_co_ei_array = np.array(ac_performance.EDB_data['CO_EI_matrix'])
            lto_hc_ei_array = np.array(ac_performance.EDB_data['HC_EI_matrix'])
            lto_ff_array = np.array(ac_performance.EDB_data['fuelflow_KGperS'])
        else:
            # If using LTO data
            lto_co_ei_array = np.array([mode['CO_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_hc_ei_array = np.array([mode['HC_EI'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            lto_ff_array = np.array([mode['FUEL_KGs'] for mode in ac_performance.LTO_data['thrust_settings'].values()])
            

        self.LTO_emission_indices['HC'] = hccoEIsFunc(fuel_flows_LTO,lto_hc_ei_array,lto_ff_array,cruiseCalc=False)
        self.LTO_emission_indices['CO'] = hccoEIsFunc(fuel_flows_LTO,lto_co_ei_array,lto_ff_array,cruiseCalc=False)

        # OC/PMvolo EIs
        thrustCat = get_thrust_cat(fuel_flows_LTO,
                                    ac_performance.fuel_flow_values, cruiseCalc=False)
        self.LTO_emission_indices['PMvol'],\
        self.LTO_emission_indices['OCic'] = EI_PMvol_NEW(
                fuel_flows_LTO, thrustCat
            )
        
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

        self.LTO_emission_indices['PMnvol'] = EI_PMnvol(
            fuel_flows_LTO,
            fuel_flows_4,
            PMnvolEI_ICAOthrust[1:],
        )

        if pmnvol_switch_lc == 'SCOPE11':
            self.LTO_emission_indices['PMnvol_lo'] = EI_PMnvol(
                fuel_flows_LTO,
                fuel_flows_4,
                PMnvolEI_lo_ICAOthrust[1:]
            )
            self.LTO_emission_indices['PMnvol_hi'] = EI_PMnvol(
                fuel_flows_LTO,
                fuel_flows_4,
                PMnvolEI_hi_ICAOthrust[1:]
            )
            # For number-based EI
            self.LTO_emission_indices['PMnvolN']     = EI_PMnvolN(thrusts, PMnvolEIN_ICAOthrust[1:])
            self.LTO_emission_indices['PMnvolN_lo']  = EI_PMnvolN(thrusts, PMnvolEIN_lo_ICAOthrust[1:])
            self.LTO_emission_indices['PMnvolN_hi']  = EI_PMnvolN(thrusts, PMnvolEIN_hi_ICAOthrust[1:])
    
    def APU_emissions(self, EDB_data, apu_tim_arr=1050, apu_tim_dep=1804):
        mask = (EDB_data['APU_fuelflow_ref'] != 0.0)

        # SOx
        self.APU_emission_indices['SO2'] = self.LTO_emission_indices['SO2'][0] if mask else 0.0
        self.APU_emission_indices['SO4'] = self.LTO_emission_indices['SO4'][0] if mask else 0.0

        # Particulate‐matter breakdown (deterministic BC fraction of 0.95)
        APU_PM10 = max(EDB_data['APU_PM10EI_ref'] - self.APU_emission_indices['SO4'], 0.0)
        bc_prop = 0.95
        self.APU_emission_indices['PMnvol'] = APU_PM10 * bc_prop
        self.APU_emission_indices['PMvol'] = APU_PM10 - self.APU_emission_indices['PMnvol']

        # NO/NO2/HONO speciation
        self.APU_emission_indices['NO']   = EDB_data['APU_PM10EI_ref'][0] * self.LTO_noProp[0]
        self.APU_emission_indices['NO2']  = EDB_data['APU_PM10EI_ref'][0] * self.LTO_no2Prop[0]
        self.APU_emission_indices['HONO'] = EDB_data['APU_PM10EI_ref'][0] * self.LTO_honoProp[0]

        self.APU_emission_indices['NOx'] = EDB_data['APU_NOxEI_ref'][0]
        self.APU_emission_indices['HC'] = EDB_data['APU_HCEI_ref'][0]
        self.APU_emission_indices['CO'] = EDB_data['APU_COEI_ref'][0]

        # CO2 via mass balance
        if mask:
            co2_ei_nom = 3160
            nvol_carb_cont = 0.95

            co2 = co2_ei_nom
            co2 -= (44/28)     * self.APU_emission_indices['CO']
            co2 -= (44/(82/5)) * self.APU_emission_indices['HC']
            co2 -= (44/(55/4)) * self.APU_emission_indices['PMvol']
            co2 -= (44/12)     * nvol_carb_cont * self.APU_emission_indices['PMnvol']
            self.APU_emission_indices['CO2'] = co2
        else:
            self.APU_emission_indices['CO2'] = 0.0

    def GSE_emissions(self, wnsf):
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
        # TODO: add to total fuel burn when APU TIMs done
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

    def lifecycle_emissions(self, fuel, traj):
        # add lifecycle CO2 emissions for climate model run
        lc_CO2 = (fuel['LC_CO2'] * (traj.fuel_mass * fuel['Energy_MJ_per_kg'])) - self.emission_g['CO2']
        self.emission_g['CO2'] += lc_CO2

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

    def __nvPM_processing(self,EDB_data,altitudes,Tamb_cruise,Pamb_cruise,machFlight,fuel_flow_cruise,pmnvolSwitch_cruise = 'SCOPE11'):
        # nvPM processing function written by Carla
        # This implementation follows the implementation by Ahrens et al (2022)
        # which is based on SCOPE11, and Peck et al (2013)  

        # -----------------------------
        # 0. Get fuel factor (sea-level equivalent fuel flow):
        # -----------------------------
        fuelFactor = get_fuel_factor(fuel_flow_cruise, Pamb_cruise, machFlight)

        # [idle, approach, climb-out, take-off]
        # geometric mean diameter (nm)
        GMD_mode = np.array([20, 20, 40, 40])

        # Step 0: find EI from SN if necessary
        # Check if EI measurements for this aircraft exist
        SN_mat_tmp = EDB_data['SN_matrix']
        ismeasure = False

        if np.min(EDB_data['nvPM_mass_matrix']) < 0:
            # ground level SN to use
            if np.min(SN_mat_tmp) < 0:
                # use max SN
                SN_ref = np.full_like(SN_mat_tmp, np.max(SN_mat_tmp))
            else:
                # else use SN matrix
                SN_ref = SN_mat_tmp.copy()

            # apply the SMOKE11 correlation to find the unadjusted CImass (mg/m^3)
            CI_mass = np.zeros(4)
            for idx in range(4):
                CI_mass[idx] = (
                    0.6484 * np.exp(0.0766 * SN_ref[idx])
                    / (1 + np.exp(-1.098 * (SN_ref[idx] - 3.064)))
                )  # mg/m^3

            # define the air fuel ratio (AFR) according to Wayson (ratio)
            # [idle, approach, climb-out, take-off]
            AFR_mode = np.array([106, 83, 51, 45])

            # find appropriate bypass ratio for the engine
            eng_type = EDB_data['ENGINE_TYPE']
            if eng_type == 'MTF':
                bypass_ratio = EDB_data['BP_Ratio']
            elif eng_type == 'TF':
                bypass_ratio = 0.0
            else:
                bypass_ratio = 0.0

            # Find the thr volumetric flow rate (m^3/kg)
            Q_mode = 0.776 * AFR_mode * (1 + bypass_ratio) + 0.767

            # Find the adjustment factor kslm
            # CImass adjusted from mg/kg to microgram/kg in this calculation
            if (eng_type == 'MTF') or (eng_type == 'TF'):
                kslm = np.log(
                    (3.219 * CI_mass * (1 + bypass_ratio) * 1000 + 312.5)
                    / (CI_mass * (1 + bypass_ratio) * 1000 + 42.6)
                )
            else:
                kslm = np.zeros(4)

            # replace the calculated values into the measurements matrix
            # nvPM_mass_mode in mg/kg
            nvPM_mass_mode = CI_mass * Q_mode * kslm  # mg/kg
            ismeasure = True

        else:
            # EI from measurements (mg/kg)
            nvPM_mass_mode = EDB_data['nvPM_mass_matrix'].astype(float)


        # if not defined, do the number calculation
        if np.min(EDB_data['nvPM_num_matrix']) < 0:
            # do num calculation
            # find number emissions #/kg fuel
            nvPM_num_mode = (
                6.0
                * nvPM_mass_mode
                / (
                    np.pi
                    * 1e9
                    * ((GMD_mode * 1.00e-9) ** 3)
                    * np.exp(4.5 * (np.log(1.8) ** 2))
                )
            )
        else:
            nvPM_num_mode = EDB_data['nvPM_num_matrix'].astype(float)


        # Step 1: Determine In-flight thermodynamic conditions
        max_pressure_ratio = np.array(EDB_data['PR'])[0]
        gamma = kappa

        # first find combustor efficiencies as defined in Ahrens et al
        # Find if aircraft is in climb, cruise, or descent
        # altitudes   # 1D array
        alt_change = np.concatenate((np.diff(altitudes), np.zeros(1)))
        eta_comp = np.zeros_like(alt_change)
        eta_comp[alt_change >= 0] = 0.88
        eta_comp[alt_change < 0] = 0.7

        # find pressure coefficient variable
        pressure_coef = np.zeros_like(alt_change)
        max_alt = np.max(altitudes)
        lin_vary_alt = (altitudes - 3000) / (max_alt - 3000)
        pressure_coef[alt_change > 0] = 0.85 + (1.15 - 0.85) * lin_vary_alt[alt_change > 0]
        pressure_coef[alt_change == 0] = 0.95
        pressure_coef[alt_change < 0] = 0.12

        # These variables define the temperature and pressure at altitude
        # Tamb_cruise and Pamb_cruise are arrays defined elsewhere (BADA standard atmosphere)
        Ttamb_total = Tamb_cruise #* (1 + ((gamma - 1) / 2) * machFlight ** 2)
        Ptamb_total = Pamb_cruise #* (1 + ((gamma - 1) / 2) * machFlight ** 2) ** (gamma / (gamma - 1))

        # find conditions at the combustor inlet
        P3 = Ptamb_total * (1 + pressure_coef * (max_pressure_ratio - 1.0))
        T3 = Ttamb_total * (1 + (1.0 / eta_comp) * ((P3 / Ptamb_total) ** ((gamma - 1) / gamma) - 1))

        # Step 2: Find conditions at the ground/reference state
        P_ground = 101325.0  # Pa
        T_ground = 288.15  # K

        # reference values
        T3_ref = T3.copy()  # assume same as T3
        P3_ref = P_ground * (1 + eta_comp * (T3_ref / T_ground - 1)) ** (gamma / (gamma - 1))

        # find F_Foo_ref
        FG_over_Foo = (P3_ref / P_ground - 1) / (max_pressure_ratio - 1)


        # Step 3: Interpolation
        # Mass:
        if (np.isnan(EDB_data['EImass_max_thrust'])) or (EDB_data['EImass_max_thrust'] < 0):
            # Use four-point interpolation
            EI_nvPM_mass_interp = np.concatenate(
                ([nvPM_mass_mode[0]], nvPM_mass_mode, [nvPM_mass_mode[-1]])
            )
            thrust_EImass_interp = np.array([-10, 0.07, 0.3, 0.85, 1, 100])
        else:
            if EDB_data['EImass_max_thrust'] == 0.575:
                EI_nvPM_mass_interp = np.concatenate(
                    (
                        [nvPM_mass_mode[0]],
                        nvPM_mass_mode[0:2],
                        [EDB_data['EImass_max']],
                        nvPM_mass_mode[2:],
                        [nvPM_mass_mode[-1]],
                    )
                )
                thrust_EImass_interp = np.array([-10, 0.07, 0.3, 0.575, 0.85, 1, 100])
            elif EDB_data['EImass_max_thrust'] == 0.925:
                EI_nvPM_mass_interp = np.concatenate(
                    (
                        [nvPM_mass_mode[0]],
                        nvPM_mass_mode[0:3],
                        [EDB_data['EImass_max']],
                        [nvPM_mass_mode[3]],
                        [nvPM_mass_mode[-1]],
                    )
                )
                thrust_EImass_interp = np.array([-10, 0.07, 0.3, 0.85, 0.925, 1, 100])
            else:
                raise ValueError(
                    f" EImass_max_thrust not recognized"
                )
            
        # Num:
        if (np.isnan(EDB_data['EInum_max_thrust'])) or (EDB_data['EInum_max_thrust'] < 0):
            # Use four-point interpolation
            EI_nvPM_num_interp = np.concatenate(
                ([nvPM_num_mode[0]], nvPM_num_mode, [nvPM_num_mode[-1]])
            )
            thrust_EInum_interp = np.array([-10, 0.07, 0.3, 0.85, 1, 100])
        else:
            if EDB_data['EInum_max_thrust'] == 0.575:
                EI_nvPM_num_interp = np.concatenate(
                    (
                        [nvPM_num_mode[0]],
                        nvPM_num_mode[0:2],
                        [EDB_data['EInum_max']],
                        nvPM_num_mode[2:],
                        [nvPM_num_mode[-1]],
                    )
                )
                thrust_EInum_interp = np.array([-10, 0.07, 0.3, 0.575, 0.85, 1, 100])
            elif EDB_data['EInum_max_thrust'] == 0.925:
                EI_nvPM_num_interp = np.concatenate(
                    (
                        [nvPM_num_mode[0]],
                        nvPM_num_mode[0:3],
                        [EDB_data['EInum_max']],
                        [nvPM_num_mode[3]],
                        [nvPM_num_mode[-1]],
                    )
                )
                thrust_EInum_interp = np.array([-10, 0.07, 0.3, 0.85, 0.925, 1, 100])
            else:
                raise ValueError(
                    f"cruiseEmissions_byFlight: EInum_max_thrust not recognized"
                )
            
        # prepare GMD for interpolation
        thrust_GMD_interp = np.array([-10, 0.07, 0.3, 0.85, 1, 100])
        GMD_interp = np.concatenate(([GMD_mode[0]], GMD_mode, [GMD_mode[-1]]))

        # now interpolate (mg/kg or #/kg)
        fg_over = FG_over_Foo  
        EI_ref_mass = np.interp(fg_over, thrust_EImass_interp, EI_nvPM_mass_interp)
        EI_ref_num = np.interp(fg_over, thrust_EInum_interp, EI_nvPM_num_interp)
        GMD_ref = np.interp(fg_over, thrust_GMD_interp, GMD_interp)

        EI_PMnvol_GMD = GMD_ref

        # Step 4: Now apply DoppelHeuer-Lecht to altitude (g/kg & #/kg)
        # also convert mg/kg to g/kg
        EI_PMnvol = (
            1e-3 * EI_ref_mass * (P3 / P3_ref) ** 1.35 * (1.1 ** 2.5)
        )
        EI_PMnvolN = EI_ref_num * EI_PMnvol / (1e-3 * EI_ref_mass)

        # correct the shape to fit with previous definition (transpose)
        EI_PMnvol = EI_PMnvol.T
        EI_PMnvol_GMD = EI_PMnvol_GMD.T
        EI_PMnvolN = EI_PMnvolN.T

        # do final checks and replace if necessary
        if np.max(SN_mat_tmp) < 0:
            print(f"SN definition for AC not sufficient {SN_mat_tmp}")
            EI_PMnvol = np.zeros_like(fuelFactor)
            if pmnvolSwitch_cruise == 'SCOPE11':
                EI_PMnvol_GMD = np.zeros_like(fuelFactor.T)
                EI_PMnvol_GMD = np.zeros_like(fuelFactor.T)

        if np.sum(EI_PMnvol < 0) > 1:
            print("Warning! Negative EI(BC) at cruise")
            EI_PMnvol[EI_PMnvol < 0] = 0.0

        return EI_PMnvol_GMD,EI_PMnvol,EI_PMnvolN
