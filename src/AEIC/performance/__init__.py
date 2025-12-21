import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, RootModel, model_validator

from .bada import BADAPerformanceModel
from .piano import PianoPerformanceModel
from .table import TablePerformanceModel
from .tasopt import TASOPTPerformanceModel

PerformanceModelUnion = Annotated[
    (
        BADAPerformanceModel
        | TASOPTPerformanceModel
        | PianoPerformanceModel
        | TablePerformanceModel
    ),
    Field(discriminator='model_type'),
]


class PerformanceModel(RootModel[PerformanceModelUnion]):
    """Wrapper class to make model_type field case-insensitive."""

    @model_validator(mode='before')
    @classmethod
    def normalize_model_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'model_type' in data:
            data = {**data, 'model_type': data['model_type'].lower()}
        return data

    @classmethod
    def load(cls, path: str | Path) -> PerformanceModelUnion:
        with open(path, 'rb') as f:
            return cls.model_validate(tomllib.load(f)).root

    @classmethod
    def from_data(cls, data: dict) -> PerformanceModelUnion:
        return cls.model_validate(data).root
