from typing import Literal

from .base import BasePerformanceModel


class TASOPTPerformanceModel(BasePerformanceModel):
    model_type: Literal['TASOPT']
    ...
