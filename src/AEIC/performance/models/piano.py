from typing import Literal

from .base import BasePerformanceModel


class PianoPerformanceModel(BasePerformanceModel):
    model_type: Literal['Piano']
    ...
