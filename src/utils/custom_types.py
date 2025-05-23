from typing import Union
import numpy as np

# from numpy import ndarray as NDArray
from numpy.typing import NDArray

# create a type for Union[float, NDArray]
FloatOrNDArray = Union[float, NDArray[np.float64]]
