"""Typed data rows of the PIANO climb, descent and cruise file formats.

These models represent the layout of a row and the `COLUMNS` tuple
of a model defines the order of the columns in the file.

The readers convert these rows to the `TableRow` tuples of the output tables.
"""

# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

# The value in the fourth cruise column of a row from the Mach sweep.
_SWEPT_DISCRIMINATOR = '|'

# PIANO cruise reference Mach specific points mapped to their column name
_REFERENCE_MACH_LABELS = {
    'maxSAR': 'max_sar',
    '99%SAR': 'sar_99',
    'maxLim': 'max_lim',
}


class UnexpectedDiscriminatorError(Exception):
    """A cruise row is composed of numbers, except for the fourth column
    which corresponds to the discriminator between reference/swept rows.

    If the discriminator is not one of the 4 expected values, then the
    file convention is not the one expected by the parser and we throw
    this error to stop parsing (different than a Pydantic validation error)

    Pydantic wraps a `ValueError` from a validator in a `ValidationError`, which
    `_PianoRow` reads as a malformed row. This class derives from `Exception` to
    pass through Pydantic unchanged.
    """


class _PianoRow(BaseModel):
    """One data row of a PIANO table."""

    model_config = ConfigDict(frozen=True)

    COLUMNS: ClassVar[tuple[str, ...]] = ()
    """Names of the columns for every row in file order.
    This defines how a row is parsed
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a subclass if `COLUMNS` does not name each required field once.

        Raises:
            TypeError: If `COLUMNS` holds a duplicate, a name that is not a
                required field, or does not hold a required field.
        """
        super().__pydantic_init_subclass__(**kwargs)
        required = {
            name for name, field in cls.model_fields.items() if field.is_required()
        }
        if len(set(cls.COLUMNS)) != len(cls.COLUMNS) or set(cls.COLUMNS) != required:
            raise TypeError(
                f'{cls.__name__}.COLUMNS must name each required field once. '
                f'COLUMNS: {cls.COLUMNS}. Required fields: {sorted(required)}'
            )

    @classmethod
    def columns(cls) -> list[str]:
        """Names of the columns that every row holds, in file order."""
        return list(cls.COLUMNS)

    @classmethod
    def _from_columns(cls, columns: list[str], tokens: list[str]) -> Self | None:
        """Build a row with one token for each column, or return None if the
        tokens are malformed."""
        if len(tokens) != len(columns):
            return None
        try:
            return cls.model_validate(dict(zip(columns, tokens, strict=True)))
        except ValidationError:
            return None

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> Self | None:
        """Parse the tokens of a data line, or return None if they are malformed.

        Call this only on a line that `_is_data_row` accepts. None then means
        that the row is malformed, rather than that the line is not a data row.
        """
        return cls._from_columns(cls.columns(), tokens)


class PianoClimbRow(_PianoRow):
    """A row of a PIANO climb table."""

    COLUMNS: ClassVar[tuple[str, ...]] = (
        'altitude_ft',
        'time_s',
        'distance_nm',
        'burn_lb',
        'fn_per_engine_lbf',
        'rocd_fpm',
        'drag_lbf',
    )

    altitude_ft: float
    time_s: float
    """Cumulative time."""
    distance_nm: float
    """Cumulative distance."""
    burn_lb: float
    """Cumulative fuel burn."""
    fn_per_engine_lbf: float
    rocd_fpm: float
    drag_lbf: float
    nox: float | None = None
    """NOx column. Only some exports write it, and its position varies."""

    @classmethod
    def columns(cls, nox_col: int | None = None) -> list[str]:
        """Names of the columns of a row, in file order.

        Set `nox_col` to the index of the NOx column if the file has one.
        """
        columns = super().columns()
        if nox_col is not None:
            columns.insert(nox_col, 'nox')
        return columns

    @classmethod
    def from_tokens(
        cls, tokens: list[str], nox_col: int | None = None
    ) -> PianoClimbRow | None:
        """Parse the tokens of a data line, or return None if they are malformed.

        Set `nox_col` to the index of the NOx column if the file writes one.
        """
        return cls._from_columns(cls.columns(nox_col), tokens)


class PianoDescentRow(_PianoRow):
    """A row of a PIANO descent table."""

    COLUMNS: ClassVar[tuple[str, ...]] = (
        'altitude_ft',
        'time_s',
        'distance_nm',
        'burn_lb',
        'rod_fpm',
        'fn_per_engine_lbf',
    )

    altitude_ft: float
    time_s: float
    """Cumulative time."""
    distance_nm: float
    """Cumulative distance."""
    burn_lb: float
    """Cumulative fuel burn."""
    rod_fpm: float
    """Rate of descent, positive when the aircraft descends."""
    fn_per_engine_lbf: float


class PianoCruiseRow(_PianoRow):
    """A row of a PIANO cruise table.

    PIANO writes more columns after these (buffet onset and emissions). The
    reader does not use them, so `from_tokens` ignores them.

    Raises:
        UnexpectedDiscriminatorError: If the numeric columns are valid, but
            `discriminator` is neither "|" nor a reference Mach label.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        'mass_lb',
        'altitude_ft',
        'mach',
        'discriminator',
        'tas_kts',
        'cas_kts',
        'drag_lbf',
        'mcr_pct',
        'lift_to_drag',
        'fuel_flow_lb_per_h',
        'sfc_lb_per_h_per_lbf',
        'sar_nm_per_lb',
        'mcl_avail_per_engine_lbf',
        'rocd_mcl_fix_mach_fpm',
        'rocd_mcl_fix_cas_fpm',
    )

    mass_lb: float
    altitude_ft: float
    mach: float
    discriminator: str
    """"|" for a swept row, or a reference Mach label."""
    tas_kts: float
    cas_kts: float
    drag_lbf: float
    mcr_pct: float
    """Percent of maximum cruise thrust."""
    lift_to_drag: float
    fuel_flow_lb_per_h: float
    sfc_lb_per_h_per_lbf: float
    sar_nm_per_lb: float
    mcl_avail_per_engine_lbf: float
    rocd_mcl_fix_mach_fpm: float
    """Signed rate of climb at MCL thrust and fixed Mach."""
    rocd_mcl_fix_cas_fpm: float
    """Signed rate of climb at MCL thrust and fixed CAS."""

    @model_validator(mode='after')
    def _check_discriminator(self) -> Self:
        """Pydantic runs this only if all of the columns are valid, so a row that
        is malformed and has unknown discriminator is marked as a malformed."""
        if not self.is_swept and self.reference_mach_column is None:
            labels = ', '.join(f'"{label}"' for label in _REFERENCE_MACH_LABELS)
            raise UnexpectedDiscriminatorError(
                f'has "{self.discriminator}" where "{_SWEPT_DISCRIMINATOR}" or a '
                f'reference Mach label ({labels}) is expected, so the row does not '
                'match the expected PIANO cruise layout'
            )
        return self

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> PianoCruiseRow | None:
        """Parse the tokens of a data line, or return None if they are malformed.

        Raises:
            UnexpectedDiscriminatorError: See the class documentation.
        """
        columns = cls.columns()
        return cls._from_columns(columns, tokens[: len(columns)])

    @property
    def is_swept(self) -> bool:
        """Whether the row comes from the Mach sweep."""
        return self.discriminator == _SWEPT_DISCRIMINATOR

    @property
    def reference_mach_column(self) -> str | None:
        """The reference Mach table column of a labelled row, or None for a
        swept row."""
        return _REFERENCE_MACH_LABELS.get(self.discriminator)
