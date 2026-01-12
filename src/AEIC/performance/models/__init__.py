import tomllib
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, RootModel, model_validator

from .bada import BADAPerformanceModel
from .base import BasePerformanceModel as BasePerformanceModel
from .legacy import LegacyPerformanceModel
from .piano import PianoPerformanceModel
from .tasopt import TASOPTPerformanceModel

PerformanceModelUnion = Annotated[
    (
        BADAPerformanceModel
        | TASOPTPerformanceModel
        | PianoPerformanceModel
        | LegacyPerformanceModel
    ),
    Field(discriminator='model_type'),
]
"""Union type representing all supported performance model types. This is a
Pydantic discriminated union, using the ``model_type`` field to guide the
actual type of model instantiated when loading models from TOML files."""


class PerformanceModel(RootModel[PerformanceModelUnion]):
    """Performance model loader.

    This is a wrapper class to implement loading of performance models from
    TOML data, and additionally to make the ``model_type`` field used for
    discriminating model types case-insensitive."""

    @model_validator(mode='before')
    @classmethod
    def normalize_model_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'model_type' in data:
            data = {**data, 'model_type': data['model_type'].lower()}
        return data

    @classmethod
    def load(cls, path: str | Path) -> PerformanceModelUnion:
        """Load a performance model from a TOML file.

        The exact performance model type is determined by the ``model_type``
        field in the TOML data."""
        with open(path, 'rb') as f:
            return cls.model_validate(tomllib.load(f)).root

    @classmethod
    def from_data(cls, data: dict) -> PerformanceModelUnion:
        """Initialize a performance model from a dictionary."""
        return cls.model_validate(data).root
