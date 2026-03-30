from typing import Literal

from AEIC.performance.types import SimpleFlightRules

from .base import BasePerformanceModel


class BADAPerformanceModel(BasePerformanceModel[SimpleFlightRules]):
    model_type: Literal['BADA']
    ...
