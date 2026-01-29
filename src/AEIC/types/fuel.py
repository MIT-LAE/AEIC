from pydantic import ConfigDict

from AEIC.utils.models import CIBaseModel


class Fuel(CIBaseModel):
    model_config = ConfigDict(frozen=True)
    """Configuration is frozen after creation."""

    name: str
    """Fuel name."""

    energy_MJ_per_kg: float
    """Fuel energy content in MJ/kg."""

    EI_H2O: float
    """Emission index for water vapor (g H2O / kg fuel)."""

    EI_CO2: float
    """Emission index for carbon dioxide (g CO2 / kg fuel)."""

    non_volatile_carbon_content: float
    # TODO: Is this the right name?
    # TODO: Units?
    """Non-volatile carbon content (???)."""

    lifecycle_CO2: float | None = None
    """Life-cycle CO2 emissions (g CO2e / MJ fuel)."""

    fuel_sulfur_content_nom: float
    """Fuel sulfur content (nominal) in ppm by weight (mg S / kg fuel)."""

    sulfate_yield_nom: float
    """Sulfate yield (nominal) as fraction of fuel sulfur content converted to
    sulfate."""
