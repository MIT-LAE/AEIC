import numpy as np

# from numpy import ndarray as NDArray
from numpy.typing import NDArray

# create a type for Union[float, NDArray]
FloatOrNDArray = float | NDArray[np.float64]
