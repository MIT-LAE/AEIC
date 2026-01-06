import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, RootModel, model_validator

from .bada import BADAPerformanceModel
from .base import BasePerformanceModel as BasePerformanceModel
from .piano import PianoPerformanceModel
from .table import TablePerformanceModel
from .tasopt import TASOPTPerformanceModel

# TODO: Document what's going on here.
PerformanceModelUnion = Annotated[
    (
        BADAPerformanceModel
        | TASOPTPerformanceModel
        | PianoPerformanceModel
        | TablePerformanceModel
    ),
    Field(discriminator='model_type'),
]


# TODO: Better docstrings.
class PerformanceModel(RootModel[PerformanceModelUnion]):
    """Wrapper class to make the ``model_type`` field case-insensitive and to
    implement loading of performance models from TOML data."""

    @model_validator(mode='before')
    @classmethod
    def normalize_model_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'model_type' in data:
            data = {**data, 'model_type': data['model_type'].lower()}
        return data

    @classmethod
    def load(cls, path: str | Path) -> PerformanceModelUnion:
        """Load a performance model from a TOML file.

        The exact performance model type is determined by the 'model_type'
        field in the TOML data."""
        with open(path, 'rb') as f:
            return cls.model_validate(tomllib.load(f)).root

    @classmethod
    def from_data(cls, data: dict) -> PerformanceModelUnion:
        """Initialize a performance model from a dictionary."""
        return cls.model_validate(data).root
