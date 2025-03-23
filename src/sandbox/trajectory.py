import numpy as np
import math
import bada as bd

def alt_change(altStart, altEnd, altStep, currentMass, fuelConsumed, distanceTraveled, elapsedTime,currentLat, currentLong, currentAz, endLat, endLong):
    
    # Initialzie some variables
    climbEndedEarly = 0
    currentAltitude = altStart
    iAlt = 1
    
    # Make sure altStep > 0
    
    if altStep == 0:
        raise ValueError("altStep is zero! Cannot proceed with the mission analysis")
    #end
    
    # Identify climb or descent phase
    if altEnd >= altStart:
        altChangeType = 'Climb'
        if altStep < 0:
            raise ValueError("altStep is negative for climb!")
    
    elif altEnd < altStart:
        altChangeType = 'Descent'
        if altStep > 0:
            raise ValueError("altStep is positive for descent!")
    #end
    
    # Check if altStep is too large compared to the climb being executed
    if (altChangeType == 'Climb' and
    (altEnd - altStart) < altStep and
    altEnd > altStart):
        altStep = altEnd - altStart
    #end
    
    # Initilize climb matrices (matrix Columns  = total number of steps)
    
    # Number of steps
    stepsleg = (altEnd - altStart)/altStep
    
    # Round the number of stepsleg and store in matrixColumns
    matrixColumns = math.ceil(stepsleg)
    
    print("Initializing missio arrays.....")
    # --- Preallocate arrays (add +1 for starting point) ---
    fuelLeg     = np.zeros(matrixColumns + 1)
    NOxLeg      = np.zeros(matrixColumns + 1)
    massLeg     = np.zeros(matrixColumns + 1)
    distanceLeg = np.zeros(matrixColumns + 1)
    altsLeg     = np.zeros(matrixColumns + 1)
    timeLeg     = np.zeros(matrixColumns + 1)
    latLeg      = np.zeros(matrixColumns + 1)
    longLeg     = np.zeros(matrixColumns + 1)
    azLeg       = np.zeros(matrixColumns + 1)
    fuelFlowLeg = np.zeros(matrixColumns + 1)
    #NOxFlowLeg  = np.zeros(matrixColumns + 1)
    TAS_Leg     = np.zeros(matrixColumns + 1)
    
    # Initialize first index
    iAlt = 0
    fuelLeg[iAlt]     = fuelConsumed
    #NOxLeg[iAlt]      = NOxEmitted
    massLeg[iAlt]     = currentMass
    distanceLeg[iAlt] = distanceTraveled
    altsLeg[iAlt]     = currentAltitude
    timeLeg[iAlt]     = elapsedTime
    latLeg[iAlt]      = currentLat
    longLeg[iAlt]     = currentLong
    azLeg[iAlt]       = currentAz
    
    
    # Account for precision issue when stepping though altitude in fixed
    # -size increments
    
    if matrixColumns > stepsleg:
        altEnd_1st = altStart + altStep * (matrixColumns - 1)
    else:
        altEnd_1st = altEnd
    
    
    # Make sure not to overshoot trajectory due to float error
    
    altEnd_1st -= altStep / 10.0
    
    ########## BEGIN RUNNING ALTITUDE CHANGE ###########
    altitudes = np.arange(altStart, altEnd_1st + altStep/2, altStep)
    
    for altitudeNow in altitudes:
       
       # Advance altitude index
       iAlt + 1
       
       # Determine current and next altitude
       alt1 = altitudeNow
       alt2 = alt1 + altStep
       
       if altChangeType == 'Climb':
           print("We are in climb mode! at altitude:", altitudeNow)
           
        # Call bada rountines here
       bd.badaClimbfuel(alt1, alt2, currentMass, currentLat, currentLong)
    
    # Determine current and next altitude
    
    print("done!")
    
    
    