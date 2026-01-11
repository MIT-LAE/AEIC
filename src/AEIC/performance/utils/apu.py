# TODO: Remove this when we move to Python 3.14.
from __future__ import annotations

import tomllib
import warnings

from pydantic import TypeAdapter

from AEIC.config import config
from AEIC.utils.models import CIBaseModel


class APU(CIBaseModel):
    """Pydantic model representing APU data."""

    name: str
    """APU name."""

    defra: str
    """DEFRA code."""

    fuel_kg_per_s: float
    """Fuel consumption in kilograms per second."""

    NOx_g_per_kg: float
    """NOx emissions in grams per kilogram of fuel."""

    CO_g_per_kg: float
    """CO emissions in grams per kilogram of fuel."""

    HC_g_per_kg: float
    """HC emissions in grams per kilogram of fuel."""

    PM10_g_per_kg: float
    """PM10 emissions in grams per kilogram of fuel."""

    @classmethod
    def unknown(cls, name: str) -> APU:
        return cls(
            name=name,
            defra='0000',
            fuel_kg_per_s=0.0,
            NOx_g_per_kg=0.0,
            CO_g_per_kg=0.0,
            HC_g_per_kg=0.0,
            PM10_g_per_kg=0.0,
        )


APUList = TypeAdapter(list[APU])
"""Pydantic type adapter for a list of APUs."""


def lookup_apu(apu_name: str) -> APU | None:
    """Look up APU data by name."""

    with open(config.file_location('engines/APU_data.toml'), 'rb') as fp:
        data = tomllib.load(fp)
    apus = APUList.validate_python(data.get('APU', []))
    for apu in apus:
        if apu.name.lower() == apu_name.lower():
            return apu
    return None


def find_apu(apu_name: str) -> APU:
    """Find APU data by name, returning "unknown" APU for unknown name."""

    apu = lookup_apu(apu_name)
    if apu is not None:
        return apu
    warnings.warn(
        f'APU "{apu_name}" not found in APU database. '
        'Using "unknown" APU with zero emissions and fuel consumption.'
    )
    return APU.unknown(apu_name)
