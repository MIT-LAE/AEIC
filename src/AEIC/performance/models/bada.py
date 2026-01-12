from typing import Literal

from .base import BasePerformanceModel


class BADAPerformanceModel(BasePerformanceModel):
    model_type: Literal['BADA']
    ...
