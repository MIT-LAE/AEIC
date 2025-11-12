from collections.abc import Mapping
from typing import Any


def as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Unable to coerce '{value}' to bool")


def require_str(mapping: Mapping[str, Any], key: str) -> str:
    raw = mapping.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Configuration option '{key}' must be a non-empty string.")
    return raw
