import numpy as np
import math

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
    NOxFlowLeg  = np.zeros(matrixColumns + 1)
    TAS_Leg     = np.zeros(matrixColumns + 1)
    
    # Initialize first index