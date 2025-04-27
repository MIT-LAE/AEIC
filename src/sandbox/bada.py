import parser as pars
import numpy as np
def badaClimbfuel(start_alt, end_alt, currentMass, currentLat, currentLon):
    # This function calculates fuel burn based on an aircraft config, starting altitude, enfing altitude and mass
    # for the climb segment
    
    # Transform to flight levels
    FL = start_alt/100
    FL_end = end_alt/100
    
    # Retrieve the climb PTF data here
    file_path = "data/B738__.PTF"
    climb_data = pars.parse_climb_data(file_path)
    flight_levels = list(climb_data.keys())

    print("Lowest flight level at climb segment: ", flight_levels[0])
    print("Highest flight level for climb segment: ", flight_levels[-1])
    
    # Ensure that the current FL is higher than the lowest climb FL for the aircraft
    if (FL < flight_levels[0]):
        raise ValueError("Current FL is lower than lowest climb FL for aircraft!")
    
    # Ensure that the current FL is lower than the highest climb FL for the aircraft
    if (FL > flight_levels[-1]):
        raise ValueError("Current FL is higher than the highest climb FL for aircraft!")
    
    
    # Retrieve the mass levels from aircraft PTF
    mass_levels = pars.parse_mass_levels(file_path)
    
    # Ensure that the current mass is within the bounds of allowble mass from PTF
    if (currentMass < mass_levels["Low Mass"]):
        raise ValueError("Current mass is lower than the lowest mass from PTF")
    
    if (currentMass > mass_levels["High Mass"]):
        raise ValueError("Current mass exceeds the maximum mass from aircraft PTF")
    
    # Ensure that the altitude in the current climb segmnet is valid
    if (end_alt < start_alt):
        raise ValueError("Ending altitude i lower than starting altitude")
    
    ###### RATE OF CLIMB ######
    
    # Get the FL index from the PTF closest to starting FL in current climb segment
    # Convert FLs (dict keys) to a NumPy array
    Climb_FLs = np.array(list(climb_data.keys()), dtype=float)
    print(Climb_FLs)
    # Find the index of the climb FL closest to the aircraft FL
    FL_index1 = np.argmin(np.abs(Climb_FLs - FL))
   
    print("Current FL_index:", FL_index1)
    # See if the aircraft FL is either greater or lesser than than th first
    # PTF climb FL and select the next flight level accordingly
    
    if FL > FL_index1:
        FL_index2 = FL_index1 + 1
        
    elif FL < FL_index1:
        FL_index2 = FL_index1 - 1
        
    elif FL == FL_index1:
        FL_index2 = FL_index1
        
    else:
        raise ValueError("Could not set climb FL index from PTF") 
    
    
    # Get climb segment flight leveeks based on PTF
    FL_1 = Climb_FLs[FL_index1]
    FL_2 = Climb_FLs[FL_index2]
    
    

    # Setup the mass matrix
    MassMat = [mass_levels["Low Mass"], mass_levels["Nominal Mass"], mass_levels["High Mass"]]
    MassMat = np.array(MassMat)

    # Find index of the closest value in MassMat to Mass
    Mass_index1 = np.argmin(np.abs(currentMass - MassMat))
    
    # Set the mass index based on the three mass levels
    
    if currentMass > MassMat[Mass_index1]:
        Mass_index2 = Mass_index1 + 1
    
    elif currentMass < MassMat[Mass_index1]:
        Mass_index2 = Mass_index1 - 1
        
    elif currentMass == MassMat[Mass_index1]:
        Mass_index2 = Mass_index1
    
    else:
        raise ValueError("Could not set mass index from PTF")
    
    # Get mass levels based on PTF indices
    Mass_1 = MassMat[Mass_index1]
    Mass_2 = MassMat[Mass_index2]
    
    
    
    # Generate the stencil for bilinear interpolation
    
    print("FL index 1:", FL_index1)
    print("FL index 2:", FL_index2)
    
    print("Mass index 1:", Mass_index1)
    print("Mass index 2:", Mass_index2)
    
    print("Flight level PTF 1: ", Climb_FLs[FL_index1])
    print("Flight level PTF 2: ", Climb_FLs[FL_index2])
    
    # Extract ROCS based on PTF flight level and mass indices
    ROC_11 = list(climb_data[FL_1]['ROCD_fpm'].values())[Mass_index1]
    ROC_12 = list(climb_data[FL_1]['ROCD_fpm'].values())[Mass_index2]
    ROC_21 = list(climb_data[FL_2]['ROCD_fpm'].values())[Mass_index1]
    ROC_22 = list(climb_data[FL_2]['ROCD_fpm'].values())[Mass_index2]
    
    
    # Interpolate in FL dimension
    
    if FL_1 == FL_2:
        ROC_1 = ROC_11
        ROC_2 = ROC_12
        
    else:
        ROC_1 = ROC_11 + (ROC_21 - ROC_11) / (FL_2 - FL_1) * (FL - FL_1)
        ROC_2 = ROC_12 + (ROC_22 - ROC_12) / (FL_2 - FL_1) * (FL - FL_1)
    
    #print(climb_data)
    