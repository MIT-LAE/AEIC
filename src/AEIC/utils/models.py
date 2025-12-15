from enum import StrEnum
from typing import Any, get_args, get_origin

from pydantic import BaseModel, model_validator


class CIStrEnum(StrEnum):
    """Case-insensitive string enumeration."""

    def __str__(self):
        """Normalize on output."""
        return self.value.lower()

    @classmethod
    def _missing_(cls, value):
        """Normalize on input."""
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value.lower() == value:
                    return member
        return None


class CIBaseModel(BaseModel):
    """Pydantic base model that recursively normalizes input keys to match
    lower-case model field names."""

    @classmethod
    def _normalize_dict(cls, values: dict) -> dict:
        """Recursively normalize keys of a dict to match model fields."""
        normalized = {}
        field_map = {f.lower(): f for f in cls.model_fields}

        for k, v in values.items():
            field_name = field_map.get(k.lower(), k)
            if field_name is None:
                continue
            field_info = cls.model_fields.get(field_name)

            # If the field is itself a Pydantic model, recurse
            if field_info is not None:
                field_type = field_info.annotation
                origin = get_origin(field_type)
                args = get_args(field_type)

                # Nested BaseModel
                if (
                    isinstance(v, dict)
                    and isinstance(field_type, type)
                    and issubclass(field_type, BaseModel)
                ):
                    v = field_type.model_validate(cls._normalize_dict(v))

                # List of BaseModels
                elif (
                    origin is list
                    and args
                    and issubclass(args[0], BaseModel)
                    and isinstance(v, list)
                ):
                    v = [
                        args[0].model_validate(cls._normalize_dict(item))
                        if isinstance(item, dict)
                        else item
                        for item in v
                    ]

            normalized[field_name] = v

        return normalized

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return cls._normalize_dict(values)
        elif isinstance(values, list):
            # Normalize each item if it's a dict
            return [
                cls._normalize_dict(v) if isinstance(v, dict) else v for v in values
            ]
        else:
            return values
