from typing import Literal

from AEIC.performance.types import SimpleFlightRules

from .base import BasePerformanceModel


class TASOPTPerformanceModel(BasePerformanceModel[SimpleFlightRules]):
    model_type: Literal['TASOPT']
    ...
